"""进程公共件（兼容 facade）：配置/传输/心跳三层的聚合入口。

compute / g2p / funasr_ts / supervisor / gateway 仍从本模块 import 全部
公共件；实现分布：
  - config.py          可调参数注册表、env 解析、日志初始化（叶子模块）
  - redis_transport.py redis-server 自举、两种队列、取消标记、mailbox 统计
  - heartbeat.py       worker 心跳落盘线程
本文件保留进程级小件：BusyWatch 看门狗状态、retry_forever 加载重试骨架、
shm_usage（/dev/shm 观测）。

入口进程（gateway.py）有自己的心跳实现（快照字段更丰富），不复用 heartbeat。
"""
import os
import time

# 先导入 config：HF_HUB_OFFLINE 强制与 HEALTH_DIR 解析必须在任何项目模块
# 使用它们之前发生（原 proc_common 模块级副作用的保持）
from config import (HEALTH_DIR, TUNABLE_PARAMS, dump_config, env_bool,
                    env_float, env_int, env_str, log_level, log_level_name,
                    setup_logging)
from heartbeat import (_heartbeat_once, heartbeat_loop, start_heartbeat,
                       write_heartbeat)
from redis_transport import (_CLAIM_HEADER, ASR_MAX_GROUP_SEC,
                             IPC_REDIS_SOCKET,
                             IPC_REDIS_URL, QUEUE_INDEX,
                             TTS_BATCH_BUCKET_CHARS, InstrumentedQueue,
                             TtsClaimQueue, cancel_add, claim_queue_pressure,
                             decode_member, mailbox_depths, shared_redis,
                             start_redis_server)


class BusyWatch:
    """推理卡死看门狗的工人侧状态。

    任务从队列取出时 enter()、处理完毕 exit()；心跳快照读 stuck_s() 上报
    "busy 已持续多久"。非 busy 恒 0——闲置工人绝不会被误判；busy 期间
    心跳线程照常落盘（mtime 新鲜），但 stuck_s 持续增长，supervisor 据此
    识别"心跳活着、主循环 hang 死"（NPU aicore 挂起、FunASR 推理卡死等
    mtime 检测盲区）。monotonic 计时不受系统时钟跳变影响。"""

    __slots__ = ("_since",)

    def __init__(self):
        self._since = None

    def enter(self) -> None:
        self._since = time.monotonic()

    def exit(self) -> None:
        self._since = None

    def stuck_s(self) -> float:
        since = self._since
        return round(time.monotonic() - since, 1) if since is not None else 0.0


# ── 传输层：Redis 队列（唯一传输，redis-server 由 supervisor 子进程托管）──
# 三段事故史，为什么最后落在 Redis：
#   1) mp.Queue：锁是 POSIX 信号量（非 robust），进程持锁瞬间被 SIGKILL
#      （OOM killer/看门狗强杀）→ 信号量永不释放 → 全员 get() 永久阻塞；
#      feeder 线程被杀还会撕裂帧。锁在 C 层共享内存里，Python 层无解
#      （20261108 生产事故）。
#   2) ZeroMQ PUSH/PULL：broker 按 round-robin 预分配消息给 worker 管道，
#      消费者 RCVHWM=1 时轮询落到忙 worker 的满管道 → 阻塞单线程转发，
#      其余空闲 worker 饿死（队头阻塞，实测吞吐 -30%）；且计数器跨进程
#      RMW 丢更新 → depth/in/out 幻影（20261110：57/113 条"幽灵积压"）。
#   3) Redis（现状）：消息常驻服务端（worker 死亡队列零丢失，其他 worker
#      接手）、LPOP 拉取 = 真 demand-pull（谁空闲谁取，无预分配无队头


def shm_usage() -> dict | None:
    """/dev/shm 使用情况（MB + 水位 pct + total_bytes）；不可用（非 POSIX/
    Windows 开发机）返回 None。gateway /health·水位告警与 supervisor 启动
    自检共用——statvfs 全仓库唯一一处。"""
    try:
        st = os.statvfs("/dev/shm")
        frsize = st.f_frsize
        total = frsize * st.f_blocks
        if total <= 0:
            return None
        used = frsize * (st.f_blocks - st.f_bavail)
        return {"total_mb": round(total / 2**20, 1),
                "used_mb": round(used / 2**20, 1),
                "total_bytes": total,
                "pct": round(used / total * 100, 1)}
    except (OSError, AttributeError):
        return None


def retry_forever(do_init, warm_key: str, warm: dict, what: str,
                  log_fn) -> object:
    """worker 模型加载重试骨架（tts KModel / asr FunASR / g2p 引擎共用）：
    指数退避 5s→120s 驻留重试；失败置 warm[key]="fail"（/health 503 →
    编排层重启容器）。

    看门狗 SIGKILL 前代工人后，NPU 显存回收有延迟、aicore 可能短暂坏态
    ——立即重生再初始化会失败。原"失败永久驻留不重试"会让一次坏态变成
    整池永久死亡（生产事故 20261108：6/6 tts 全部 warm=fail 僵尸、零容量，
    认领队列顶满全服务堵塞）。**不降级**（fail-fast）：持续失败保持
    warm=fail，编排层重启容器换取全新 NPU 上下文；悄悄降级 CPU 会把
    "卡坏了"伪装成"还能跑"，掩盖故障。"""
    delay = 5.0
    while True:
        try:
            return do_init()
        except Exception as exc:  # noqa: BLE001
            warm[warm_key] = "fail"
            log_fn.error("%s init failed, retry in %.0fs (warm=fail → /health"
                         " 503 → orchestration restarts container): %s",
                         what, delay, str(exc)[:200])

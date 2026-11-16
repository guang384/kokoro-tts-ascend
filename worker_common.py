"""worker 公共件：两推理角色（tts/asr）共享的状态与脚手架。

compute 拆分出的公共层——看门狗、stats 账本、warm 态、心跳快照骨架、
NPU 内存采样、OOM 识别、chunk_ts 回投。角色入口在 tts_worker/asr_worker；
supervisor 仍经 compute（facade）取 tts_main/asr_main。

注意：心跳线程运行的快照路径绝不能调 torch.npu API——设备坏态时副线程
碰 NPU 会 SIGBUS（生产事故 20261109）；采样只由主循环写 _stats 缓存。
"""
import logging
import re
import time

import proc_common
import protocol

_mem_log = logging.getLogger("kokoro.compute")

_started_at = time.time()
SAMPLE_RATE = proc_common.env_int("KOKORO_SAMPLE_RATE", minimum=8000)
_MAX_BATCH = proc_common.env_int("TTS_MAX_BATCH_SIZE", minimum=1)
_ASR_MAX_BATCH = proc_common.env_int("ASR_MAX_BATCH_SIZE", minimum=1)  # asr 批认领上限（显存预算 npu_ops._asr_auto_budget_mb 随它缩放）
_busy = proc_common.BusyWatch()  # 推理卡死看门狗：任务取出→enter，处理完→exit
# stats 每进程仅一个角色在跑，共用一份无妨。**全部键必须预置**：心跳线程
# 用 dict(_stats) 快照，主循环中途首插新键会触发 "dictionary changed size
# during iteration" → 心跳线程死亡 → supervisor 30s 后误杀健康 worker。
# jobs：tts=前向批数 / asr=对齐作业数（同 dict 但每进程只跑一个角色，无冲突）
_stats = {"jobs": 0, "forward_ms": 0.0, "audio_s": 0.0,
          "forward_errors": 0, "job_drops": 0, "fp16_jobs": 0,
          "asr_ms": 0.0, "job_errors": 0, "npu_mem": None,
          "budget_error": None, "cancelled_chunks": 0,
          "batches": 0, "batch_max": 0}
_tts_warm = {"kokoro": "pending"}   # tts 池预热项
_asr_warm = {"funasr": "pending"}   # asr 池预热项


def group_by_request(chunks: list) -> dict:
    """批内 chunk 按请求分组、组内按句序排序（纯函数，供测试锁定混批语义）。

    返回 {req_id: [(chunk_idx, batch_pos, chunk), ...]}——batch_pos 是该
    chunk 在批中的位置（对应 synth_batch 返回音频的下标）。"""
    by_req: dict = {}
    for pos, c in enumerate(chunks):
        by_req.setdefault(c["req_id"], []).append((int(c["idx"]), pos, c))
    for rid in by_req:
        by_req[rid].sort(key=lambda t: t[0])
    return by_req


def _with_busy(fn):
    """看门狗覆盖段：NPU 初始化（设备探测/KModel 加载/冒烟前向/FunASR
    推理）存在「不抛错、永久阻塞」的坏态（aicore 挂起）。不进 BusyWatch
    的话心跳照常新鲜、stuck_s 恒 0——僵尸 worker 永远占位，产能静默损失
    （串行启动闸超时放行后无人再管它）。进入本段后 supervisor 按
    STUCK_KILL_S（默认 300s >> 正常加载 60-90s）击杀重生。"""
    _busy.enter()
    try:
        return fn()
    finally:
        _busy.exit()


def _snapshot_base(role: str, device, ready: bool, warm: dict,
                   extra_stats: dict) -> dict:
    """心跳快照公共骨架（tts/asr 共用）：device/warm/stats + NPU 占用 +
    看门狗（/health 诊断用）。

    注意：本函数在心跳线程运行，**绝不能调 torch.npu API**——设备坏态时
    副线程碰 NPU 会被 SIGBUS 直接杀进程（生产事故 20261109：5 个空闲
    tts worker 的心跳线程相继被带走）。占用数据由主循环每次前向后采样
    写入 _stats["npu_mem"]，这里只读缓存值（非实时，5s 心跳粒度）。"""
    return {"role": role, "device": device, "ready": ready,
            "uptime_s": round(time.time() - _started_at, 1),
            "stuck_s": _busy.stuck_s(),
            "warm": dict(warm),
            "stats": {**dict(_stats), "ipc_dropped": protocol.drop_count(),
                      **extra_stats},
            "npu_mem": _stats.get("npu_mem")}


def _log_mem_sample(mem: dict, prev: dict | None) -> None:
    """实际 npu-smi 内存变化时打一行 INFO（30s 缓存天然限频）。

    metrics 是连续观测；日志给真机排障一个可直接 grep 的锚点：
    actual 与 planned 的差就是公式偏差。"""
    if not mem or mem.get("process_mb") is None:
        return
    if prev and prev.get("process_mb") == mem.get("process_mb"):
        return
    planned = (f"{mem['planned_mb']:.0f}"
               if mem.get("planned_mb") is not None else "n/a")
    delta = (f"{mem['delta_mb']:+.0f}"
             if mem.get("delta_mb") is not None else "n/a")
    ctx_actual = (f"{mem['context_actual_mb']:.0f}"
                  if mem.get("context_actual_mb") is not None else "n/a")
    _mem_log.info(
        "npu mem actual=%.0fMB reserved=%.0fMB estimated=%.0fMB "
        "planned=%s delta=%s context_actual=%s",
        mem["process_mb"], mem.get("reserved_mb", 0.0),
        mem.get("estimated_mb", 0.0), planned, delta, ctx_actual)


def _sample_worker_npu_mem(dev: str, budget, context_mb: float,
                           budget_err_fn) -> None:
    """主循环采样 worker 的 NPU 内存进 _stats（tts/asr 同一估算口径）——
    只有主线程碰 NPU API，心跳线程只读 _stats["npu_mem"]。

    估算口径由 npu_ops.estimate_worker_mem 统一：reserved（torch 缓存池）
    + CANN 上下文常量（tts 450 / asr 153），并给出预算/整卡占比。

    **仅 NPU 设备启用**：非 NPU（显式 DEVICE=cpu/测试替身/无驱动容器）直接跳过
    ——torch_npu 在无驱动环境调用会在 C 层段错误，except 拦不住
    （20261109 镜像内验证 Segmentation fault 的根因）。"""
    if not str(dev or "").startswith("npu"):
        return
    try:
        import npu_ops

        prev = _stats.get("npu_mem")
        mem = npu_ops.npu_mem_estimate(str(dev), budget, context_mb)
        if mem is not None:
            _stats["npu_mem"] = mem
            _log_mem_sample(mem, prev)
        err = budget_err_fn()
        if err:
            _stats["budget_error"] = err
    except Exception:  # noqa: BLE001  采样失败不影响服务
        pass


def _chunk_key(x: dict) -> str:
    """chunk 级关联键：{req_id}_{need}_{idx}——TTS chunk 行与 ASR job 行
    同值（同一块音频的两个阶段），grep 它直接对齐两段。"""
    return f"{x.get('req_id')}_{x.get('need')}_{x.get('idx')}"


def _queue_ms(x: dict) -> float:
    """认领等待毫秒：claim − enqueue（chunk/job 通用；缺戳按 0）。"""
    enq, claim = x.get("_q_enq"), x.get("_q_claim")
    if isinstance(enq, (int, float)) and isinstance(claim, (int, float)):
        return max(0.0, (claim - enq) * 1000)
    return 0.0


def _reqs_list(by_req: dict) -> str:
    """批次归属清单：rid(chunk 数) 用 + 连接（forward 失败行的关联字段）。"""
    return "+".join(f"{rid}({len(items)})" for rid, items in by_req.items())


_OOM_RE = re.compile(
    r"(?i)\bout of memory\b|\boom(?:[-_ ]?kill(?:ed)?)?\b"
    r"|hbm[^\n]{0,40}(?:alloc|memory)")


def _looks_like_oom(msg: str) -> bool:
    """识别显存耗尽类异常（torch "out of memory"、独立缩写 OOM、昇腾 HBM
    分配/内存文案）——"SharedMemory" 等普通错误不误触发自愈。

    不能用裸子串：`"oom" in msg` 会命中 room/zoom/bloom，误判会把
    effective_batch 永久减半（只降不升）。"""
    return bool(_OOM_RE.search(str(msg)))


def _reply_chunk_ts(job: dict, timestamp: list, error,
                    gateway_queues, log, bucket_ts=None) -> None:
    """chunk_ts 回投 origin 网关（成功/失败统一路径）。回投满计
    job_drops——网关 fut_ts 超时 504 兜底（at-most-once 设计内）。

    请求级墙钟跨度全在网关按戳计算：q_asr = min(首 job 入队 _q_enq)→
    max(末 job 出队 _q_claim)；asr = min(认领)→max(末桶完成 asr_done)；
    asr_bucket = 桶起始时刻，仅进过推理桶的 job 携带——网关按同请求
    去重计数 asr_fw（桶数）。"""
    timings = {"q_asr_first": job.get("_q_enq"),
               "q_asr_last": job.get("_q_claim"),
               "asr_done": time.time()}
    if bucket_ts is not None:
        timings["asr_bucket"] = bucket_ts
    ok = protocol.put_result(
        gateway_queues[int(job.get("origin", 0))],
        protocol.chunk_ts(job["req_id"], job["idx"], job["need"],
                          job["piece"]["samples"], timestamp,
                          error=error,
                          timings=timings), log)
    if not ok:
        _stats["job_drops"] += 1


def _reply_chunk_ts_safe(job: dict, timestamp: list, error,
                         gateway_queues, log, warn: str,
                         bucket_ts=None) -> None:
    """_reply_chunk_ts 的防御壳：畸形载荷/回投异常只废自家（计 job_drops +
    warning），绝不炸整批处理循环。"""
    try:
        _reply_chunk_ts(job, timestamp, error, gateway_queues, log,
                        bucket_ts=bucket_ts)
    except Exception as exc:  # noqa: BLE001  畸形载荷只废自家回投
        _stats["job_drops"] += 1
        log.warning("%s req=%s idx=%s: %s", warn, job.get("req_id"),
                    job.get("idx"), str(exc)[:160])

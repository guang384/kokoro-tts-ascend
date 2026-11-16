#!/usr/bin/env python3
"""supervisor：Actor 风格流水线的监督者（进程编排父进程）。

进程拓扑：
    gateway×E  HTTP 网关：接入/校验/聚合（synth_part 按 req_id 收齐）/编码
    g2p×G      Stage1 文本池：切句/G2P
    tts×T      TTS 前向：无状态批工人（每段直投 origin 网关）
    asr×S      FunASR 时间戳推理

分发：入口进程各自以 SO_REUSEPORT 绑定同一端口，内核按四元组哈希把新
连接均匀分发到各入口；内容级并行由统一队列完成。

兜底：任一子进程退出 → 指数退避后同角色拉起（稳定 60s 清零）；心跳停更
（HEARTBEAT_STALE_S）或 busy 卡死（stuck_s > STUCK_KILL_S）→ 强杀重启；
全部角色就绪后连续 WEDGE_RESTART_S 零成功且仍有超时 → exit(1) 交容器
重启（覆盖历史 mp.Queue 信号量泄漏 / redis-server 死亡等进程还活着的
假死）；NPU 角色串行启动（上一个 ready 再拉下一个）。SIGTERM 全组退出。

env：GATEWAY_WORKERS / G2P_WORKERS / TTS_WORKERS / ASR_WORKERS /
PORT / KOKORO_HEALTH_DIR / HEARTBEAT_STALE_S 等，
全量见 proc_common.TUNABLE_PARAMS（启动时逐项打印）。
"""
import json
import multiprocessing as mp
import os
import signal
import sys
import time

import proc_common

log = proc_common.setup_logging("supervisor")   # 尊重 LOG_LEVEL，格式与其余角色统一

GATEWAY_WORKERS_N = proc_common.env_int("GATEWAY_WORKERS", 2, minimum=1)  # 网关进程数
G2P_WORKERS_N = proc_common.env_int("G2P_WORKERS", 2, minimum=1)  # Stage 1 音素化进程数
TTS_N = proc_common.env_int("TTS_WORKERS", 4, minimum=1)    # TTS 前向进程数
ASR_N = proc_common.env_int("ASR_WORKERS", 4, minimum=1)    # ASR 对齐推理进程数（0/负数=配置错误，钳 1 并告警）
PORT = proc_common.env_int("PORT", 5174, minimum=1)
HEALTH_DIR = os.getenv("KOKORO_HEALTH_DIR", "/tmp/kokoro-health")
MAX_PENDING_REQUESTS = proc_common.env_int("MAX_PENDING_REQUESTS", 200, minimum=1)  # 受理级闸门（请求数）
WORKER_MAILBOX_SIZE = proc_common.env_int("WORKER_MAILBOX_SIZE", 1024, minimum=1)   # Mailbox 级容量（消息数）
MAX_INFLIGHT_SENTENCES = proc_common.env_int("MAX_INFLIGHT_SENTENCES", 512, minimum=1)  # 在途级闸门（句子数）
HEARTBEAT_STALE_S = proc_common.env_float("HEARTBEAT_STALE_S", 30.0, minimum=15.0)  # 心跳停更判定线（心跳 5s 一写；下限 15s=3×周期，防止误杀健康 worker）
STUCK_KILL_S = proc_common.env_float("STUCK_KILL_S", 300.0, minimum=30.0)  # busy 卡死击杀线（模型推理 hang）
SPAWN_READY_TIMEOUT_S = proc_common.env_float("SPAWN_READY_TIMEOUT_S", 360.0, minimum=0.0)  # NPU 角色串行启动的就绪等待上限；0=禁用（并行启动，旧行为）
WEDGE_RESTART_S = proc_common.env_float("WEDGE_RESTART_S", 900.0, minimum=0.0)  # 端到端零成功判死线；0=禁用
# 毒块防护：同一 chunk 累计重派超过该次数 → 放弃（受影响请求 504），
# 不再无限循环炸 worker
REDISPATCH_MAX = proc_common.env_int("REDISPATCH_MAX", 1, minimum=0)
# TTL 兜底线（2×看门狗击杀线）：死亡重派走 requeue_worker 秒级路径，
# 此值只兜「worker 活着但永远不 ack」与 supervisor 漏检的极端
SWEEP_TTL_S = 2 * STUCK_KILL_S

ctx = mp.get_context("spawn")
procs = {}  # (role, i) → Process
# 串行启动：NPU 角色（tts/asr）同类 worker 一个 ready 后才拉起下一个——
# 并行 init 的 6 路抢卡是 20261109 卡坏态雪崩（SIGBUS→500001 全池起不来）的放大器；
# CPU 角色不抢卡，照旧并行秒起
_SERIAL_ROLES = {"tts", "asr"}
starting = {}  # (role, i) → 拉起时刻（仅 _SERIAL_ROLES 登记）
mailboxes = {}  # 名字 → 队列对象（Redis 键名元数据；心跳深度探针用）
_spawn_ts = {}     # (role, i) → 最近拉起时刻（稳定判定）
_fail_count = {}   # (role, i) → 连续崩溃次数（指数退避）
_next_ok = {}      # (role, i) → 最早允许重启时刻
_crash_total = {}  # (role, i) → 累计崩溃次数（/health workers[].crashes 展示）

EXPECTED_ROLES = {"gateway": GATEWAY_WORKERS_N, "g2p": G2P_WORKERS_N,
                  "tts": TTS_N, "asr": ASR_N}


def _wtag(role: str, i: int) -> str:
    """worker 心跳/日志标签（单一映射）：gateway 用 gw-N（1 起），其余
    role-i。crash 计数与 gateway._worker_row 的 wtag 必须同源，否则
    /health 的 crashes 永远为空、/metrics 的 worker 标签分裂。"""
    return f"gw-{i + 1}" if role == "gateway" else f"{role}-{i}"


def _as_int(v) -> int:
    """心跳字段容错：坏值按 0（监控循环不能因一条坏快照停摆）。"""
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def spawn_one(role: str, i: int) -> None:
    """拉起角色进程；各角色只接收自己消费/投递的 mailbox 句柄（最小权限）。

    mailbox 句柄从模块级 mailboxes 字典取——队列对象经 spawn pickle 只传
    键名元数据（子进程惰性建连），__main__ 块已填充完毕。"""
    def mb(name):
        return mailboxes[name]

    gqs = mailboxes["gateway_queues"]  # 全部网关 mailbox：按 origin 回投的角色需要完整列表
    if role == "g2p":
        import g2p

        target, args, tag = g2p.main, (i, mb("g2p_queue"), mb("tts_queue"), gqs), f"g2p-{i}"
    elif role == "gateway":
        import gateway

        target, args, tag = gateway.gateway_main, (i, PORT, mb("g2p_queue"),
                                                   gqs[i]), f"gw-{i + 1}"
    elif role == "asr":
        import compute

        # chunk 自带 origin → chunk_ts 直投对应网关
        target, args, tag = compute.asr_main, (i, mb("asr_queue"), gqs), f"asr-{i}"
    else:  # tts
        import compute

        target, args, tag = compute.tts_main, (i, mb("tts_queue"),
                                               mb("asr_queue"), gqs), f"tts-{i}"
    p = ctx.Process(target=target, args=args, name=tag)
    # 先登记后 start：信号处理器在主线程字节码间隙执行，start 与登记之间
    # 被打断的话 shutdown 遍历不到该子进程 → 孤儿持有 NPU 显存/监听端口。
    # 未 start 的 Process 被 terminate/is_alive 时抛的异常均有调用
    # 方 try 兜底（窗口仅数条字节码，monitor 每 2s 一轮不可能落在其间）。
    procs[(role, i)] = p
    p.start()
    _spawn_ts[(role, i)] = time.time()
    _crash_total.setdefault((role, i), 0)  # 预置：心跳线程遍历该 dict 期间不得增键
    if SPAWN_READY_TIMEOUT_S and role in _SERIAL_ROLES:
        starting[(role, i)] = time.time()  # 占串行启动闸，ready/死亡/超时后放行
    log.info("spawned [%s] pid=%s", tag, p.pid)


def shutdown(signum, frame):
    log.info("exit signal received, shutting down all children")
    for p in procs.values():
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    for p in procs.values():  # 限时等待子进程退出，避免 in-flight 响应被硬切
        try:
            p.join(timeout=5)
            if p.is_alive():  # 对 SIGTERM 装死的升级 SIGKILL（不能留孤儿进程）
                p.kill()
                p.join(timeout=2)
        except Exception:  # noqa: BLE001
            pass
    if _redis_proc is not None:
        try:
            _redis_proc.terminate()
            _redis_proc.join(timeout=2)
        except Exception:  # noqa: BLE001
            pass
    _cleanup_own_heartbeat()
    sys.exit(0)


def _cleanup_own_heartbeat() -> None:
    try:
        os.unlink(os.path.join(HEALTH_DIR, f"worker-{os.getpid()}.json"))
    except OSError:
        pass


def _clean_stale_heartbeats() -> None:
    """启动时无条件清掉上一代进程遗留的心跳文件。

    不能按 mtime 阈值宽容：docker restart 保留容器 fs，PID namespace 重置
    后新 worker 大概率撞上旧代同号 pid 的残留快照——旧 ready=true 会让串行
    启动闸提前放行（NPU 并行 init 抢卡，正是雪崩放大器），旧 stuck_s>0 会让
    reap_hung 按文件名 pid 误杀新 worker。此刻本代 worker 尚未拉起，全删安全。"""
    try:
        for name in os.listdir(HEALTH_DIR):
            if name.startswith(("worker-", ".tmp-")):
                try:
                    os.unlink(os.path.join(HEALTH_DIR, name))
                except OSError:
                    continue
    except OSError:
        pass


def _worker_ready(pid: int) -> bool:
    """心跳文件里该 pid 是否已 ready（读不了按未就绪，下轮再查）。"""
    try:
        with open(os.path.join(HEALTH_DIR, f"worker-{pid}.json"),
                  encoding="utf-8") as f:
            return bool(json.load(f).get("ready"))
    except (OSError, ValueError):
        return False


def _prune_starting(now: float) -> None:
    """放行启动闸：worker ready / 死亡 / 超时——同角色下一个可拉起。"""
    for key in list(starting):
        p = procs.get(key)
        if p is None or not p.is_alive():
            starting.pop(key, None)  # 启动期死亡：交回重生路径，闸立即放行
        elif _worker_ready(p.pid):
            log.info("[%s-%d] started (ready in %.0fs)",
                     key[0], key[1], now - starting[key])
            starting.pop(key, None)
        elif SPAWN_READY_TIMEOUT_S and now - starting[key] > SPAWN_READY_TIMEOUT_S:
            log.warning("[%s-%d] not ready after %.0fs, releasing start gate",
                        key[0], key[1], now - starting[key])
            starting.pop(key, None)


def _role_start_busy(role: str) -> bool:
    """该角色还有未就绪的新拉起工人 → 暂缓拉起下一个（串行启动）。"""
    return any(k[0] == role for k in starting)


def _wait_started(key) -> None:
    """冷启动串行：阻塞等刚拉起的 NPU worker ready/死亡/超时。
    SPAWN_READY_TIMEOUT_S=0 时直接返回（并行启动，旧行为）。"""
    if not SPAWN_READY_TIMEOUT_S:
        return
    deadline = time.time() + SPAWN_READY_TIMEOUT_S
    while time.time() < deadline:
        # 串行启动期也必须跑看门狗/Redis 守护：原实现把主线程阻塞在这里，
        # 直到最后一个 worker 拉起才进 monitor_once——hung NPU init 的
        # stuck_s 无人读，SPAWN_READY_TIMEOUT 后下一个 worker 会和它一起
        # 抢卡（正是串行启动要避免的雪崩）。reap_hung 到线即杀，_prune_starting
        # 随即放行闸门。
        try:
            _check_redis()
        except Exception:  # noqa: BLE001  Redis 重启失败：继续等/下一轮再试
            log.exception("redis guardian failed during startup (continue)")
        try:
            reap_hung()
        except Exception:  # noqa: BLE001  监控异常不能阻塞启动闸
            log.exception("reap_hung failed during startup (continue)")
        _prune_starting(time.time())
        if key not in starting:
            return
        time.sleep(0.5)


def _shm_total_bytes() -> int:
    """/dev/shm 总容量（字节）；读不了返回 -1。"""
    try:
        st = os.statvfs("/dev/shm")   # POSIX；Windows 开发机无此 API → 跳过
        return st.f_frsize * st.f_blocks
    except (OSError, AttributeError):
        return -1


def check_shm() -> None:
    """启动自检：/dev/shm 过小 → 长文本 /tts 的音频 SHM 块会写满 tmpfs
    触发 SIGBUS（生产事故 20261109：docker 默认 64MB，/tts 长文本必崩
    而 /v1 短文本幸存）。docker run 必须带 --shm-size 2g 起步。"""
    total = _shm_total_bytes()
    if total < 0:
        log.warning("/dev/shm 容量探测失败（跳过 SHM 自检）")
        return
    gb = total / 2**30
    if total < 1 * 2**30:
        log.error("/dev/shm 只有 %.0fMB —— 长文本 /tts 的音频 SHM 会写满触发"
                  " SIGBUS（症状：/v1 正常、/tts 大批全崩 rc=-7）。修复："
                  "docker run 加 --shm-size 2g 并重启容器", total / 2**20)
    else:
        log.info("/dev/shm %.1fGB (SHM 自检通过)", gb)


def classify_heartbeat(path: str, now: float) -> tuple:
    """心跳判决：(verdict, detail)，verdict ∈ {"ok", "stale", "stuck"}。

    stale = 文件停更 ≥HEARTBEAT_STALE_S（心跳线程死亡/整机卡死）；
    stuck = 心跳新鲜但工人上报 busy 卡住 >STUCK_KILL_S——模型推理 hang
    （NPU aicore 挂起、FunASR 卡死等）时心跳线程照常落盘，光看 mtime
    是盲区，必须读快照里的 stuck_s（worker 侧 BusyWatch 上报）。
    文件读不了按 ok 宽容处理（与既有语义一致）。"""
    try:
        if now - os.path.getmtime(path) > HEARTBEAT_STALE_S:
            return "stale", f"heartbeat stale >= {HEARTBEAT_STALE_S}s"
    except OSError:
        return "ok", ""
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
    except (OSError, ValueError):
        return "ok", ""  # 半截/损坏文件：真死会走 stale 判定，这里宽容
    try:
        stuck = float(snap.get("stuck_s") or 0)
    except (TypeError, ValueError):
        return "ok", ""  # 坏字段不让监控循环抛错；真死走 stale
    if stuck > STUCK_KILL_S:
        return "stuck", f"busy stuck {stuck:.0f}s > {STUCK_KILL_S:.0f}s (inference hang)"
    return "ok", ""


def reap_hung() -> None:
    """hang 清扫：心跳停更（stale）或推理卡死（stuck，看门狗上报）→ 强杀。

    文件不存在不算（启动初期尚未写首份心跳，宽容）。仅强杀自己的子进程。"""
    try:
        now = time.time()
        for name in os.listdir(HEALTH_DIR):
            if not (name.startswith("worker-") and name.endswith(".json")):
                continue
            pid_str = name[len("worker-"):-5]
            if not pid_str.isdigit():
                continue
            path = os.path.join(HEALTH_DIR, name)
            verdict, detail = classify_heartbeat(path, now)
            if verdict == "ok":
                continue
            pid = int(pid_str)
            for key, p in procs.items():
                if p.pid == pid:
                    log.warning("[%s-%s] %s, judged hung, force restart",
                                key[0], key[1], detail)
                    try:
                        p.kill()
                    except Exception:  # noqa: BLE001
                        pass
                    break
    except OSError:
        pass


_wedge_state = {"last_res": None, "last_res_ts": 0.0,
                "last_tmo": None, "last_tmo_ts": 0.0,
                "last_5xx": None, "last_5xx_ts": 0.0}
_redis_proc = None  # redis-server 子进程（main 拉起、_check_redis 守护）
_tts_queue = None   # 认领队列实例（main 里自 mailboxes 赋值，sweeper 用）
_asr_queue = None   # asr 同款认领队列（chunk 级对齐 job 的 in-flight 追踪）
_redis_client = None        # sweeper 用 redis 客户端（redis 重启时随 _check_redis 重建）
_sweep_totals = {"redispatched": 0, "gaveup": 0}
_bad_seen = {"n": None}     # 上轮坏 member 累计（增量告警基线）


def _watch_queue_bad() -> None:
    """坏 member 增量告警（不依赖 METRICS_ENABLED）：每轮汇总 tts/asr 两条
    认领队列的累计计数，增量出现即 WARNING——滚动升级混跑/载荷损坏唯一的
    主动信号；首轮只记基线（容器刚起时计数可能已非零）。"""
    if _tts_queue is None and _asr_queue is None:
        return
    total = 0
    for q in (_tts_queue, _asr_queue):
        if q is None:
            continue
        try:
            total += int((q.stats() or {}).get("bad") or 0)
        except Exception:  # noqa: BLE001  redis 瞬断：下轮再读
            return
    prev = _bad_seen["n"]
    if prev is not None and total > prev:
        log.warning("queue bad members discarded: +%d (total %d) — "
                    "payload corruption or version-mixed members",
                    total - prev, total)
    _bad_seen["n"] = total


def _sweep_inflight() -> None:
    """在途重派扫描：per-worker 在途账本按 worker 死亡精确重派 →
    重派给其他 worker；重派超上限放弃。仅告警不重启——单条重派=某 worker
    在途死亡，属既有自愈闭环。"""
    global _redis_client
    if _redis_client is None:
        _redis_client = proc_common._make_redis_client()
    try:
        # 存活 pid 集合（BRPOPLPUSH 级追踪）：重派条件是「worker 已死」
        # 而非「TTL 超时」——慢 worker 不被误重派（重复前向/部分重叠的
        # 根因消除）；TTL 只兜 supervisor 漏检极端
        # 必须用 is_alive()：procs 里的死进程要等本函数之后的 respawn
        # 逻辑才 pop（串行闸/退避期间可能滞留多轮），仅看 pid 会让 sweep
        # 把死者的在途账本当活人，重派被延迟到闸门释放。
        alive = set()
        for p in procs.values():
            if not p.pid:
                continue
            try:
                if p.is_alive():
                    alive.add(p.pid)
            except Exception:  # noqa: BLE001  未 start/已关闭的 Process
                continue
        r = _tts_queue.sweep(alive, time.time(), SWEEP_TTL_S, REDISPATCH_MAX)
        for q in (_asr_queue,):
            if q is not None:
                r2 = q.sweep(alive, time.time(), SWEEP_TTL_S, REDISPATCH_MAX)
                r["redispatched"] += r2["redispatched"]
                r["gaveup"] += r2["gaveup"]
    except Exception as exc:  # noqa: BLE001  redis 瞬断：下轮再试
        log.warning("inflight sweep failed (retry next cycle): %s", exc)
        return
    _sweep_totals["redispatched"] += r["redispatched"]
    _sweep_totals["gaveup"] += r["gaveup"]
    if r["redispatched"]:
        log.warning("inflight redispatched %d (dead worker precise requeue"
                    "; TTL 兜底=%.0fs) total=%d gaveup=%d",
                    r["redispatched"], SWEEP_TTL_S,
                    _sweep_totals["redispatched"], _sweep_totals["gaveup"])
    if r["gaveup"]:
        log.error("inflight discarded %d chunks (bad member / redispatch "
                  "over limit %d; affected requests → 504)",
                  r["gaveup"], REDISPATCH_MAX)


def _check_redis() -> None:
    """redis-server 守护：死亡 = 队列与在途**一起丢失**（服务端内存态，
    无持久化——设计如此），损失有界：受影响请求走 WAIT_TIMEOUT 504 兜底。
    自动重启，新实例队列从零起。注意：redis 是本进程子进程，容器级重启
    （K8s/docker）场景全进程组同亡——本守护只覆盖 supervisor 存活而
    redis 单独死的窄窗口。"""
    global _redis_proc, _redis_client
    if _redis_proc is None or _redis_proc.poll() is None:
        return
    log.critical("redis-server died (rc=%s), restarting "
                 "(queues+in-flight lost, requests fall back to WAIT 504)",
                 _redis_proc.returncode)
    _redis_proc = proc_common.start_redis_server()
    _redis_client = None                      # 旧实例连接已死，下轮重建


def _gateway_activity(now: float) -> tuple:
    """聚合网关心跳：(requests, results, timeouts, server_5xx, role_ready)。

    role_ready 用于排除“某角色还没 ready（启动/热身）”的误判窗口。文件
    过旧/损坏/字段类型异常一律容错（按 0）；无网关心跳返回 None。"""
    req = res = tmo = s5 = 0
    n_gw = 0
    role_ready = {r: 0 for r in EXPECTED_ROLES}
    try:
        names = os.listdir(HEALTH_DIR)
    except OSError:
        return None
    for name in names:
        if not (name.startswith("worker-") and name.endswith(".json")):
            continue
        path = os.path.join(HEALTH_DIR, name)
        try:
            if now - os.path.getmtime(path) > HEARTBEAT_STALE_S:
                continue
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        role = d.get("role")
        if d.get("ready") and role in role_ready:
            role_ready[role] += 1
        if role != "gateway" or not d.get("ready"):
            continue
        n_gw += 1
        s = d.get("stats") or {}
        req += _as_int(s.get("requests"))
        res += _as_int(s.get("results"))
        tmo += _as_int(s.get("timeouts"))
        s5 += _as_int(s.get("server_5xx"))
    if n_gw == 0:
        return None
    return req, res, tmo, s5, role_ready


def check_wedge(now: float) -> None:
    """端到端假死兜底（transport 无关）：全部角色就绪后，连续
    WEDGE_RESTART_S 零成功、且最近 WEDGE_RESTART_S/2 内仍有超时或
    服务端 5xx 增长 → os._exit(1) 交容器重启。

    为什么需要：历史 mp.Queue 的 POSIX semaphore 被 SIGKILL 持有者永久
    泄漏会让所有存取方静默卡死（SND/RCV 线程 wchan=futex_wait），心跳线程
    与 BusyWatch 都还活着——只能由 supervisor 从“请求进得去、结果出不来”
    这一服务级事实判死；redis-server 死亡/上游生产卡死是同一症状。
    用请求级计数而不是队列深度判定：避免依赖任何传输的队列内部状态。"""
    if WEDGE_RESTART_S <= 0:
        return
    act = _gateway_activity(now)
    if act is None:
        _wedge_state.update(last_res=None, last_tmo=None, last_5xx=None)
        return
    req, res, tmo, s5, role_ready = act
    if any(role_ready.get(r, 0) < 1 for r in EXPECTED_ROLES):
        # 启动/热身期（tts/asr 未 ready）不算：此窗口的 504 是预期行为
        _wedge_state.update(last_res=None, last_tmo=None, last_5xx=None)
        return
    st = _wedge_state
    if st["last_res"] is None:  # 首次全角色就绪：以当前计数为基线
        st.update(last_res=res, last_res_ts=now, last_tmo=tmo,
                  last_5xx=s5, last_5xx_ts=now)
        return
    # 计数变化即刷新时间戳（增长=结果推进；下降=网关成员变化——部分网关
    # 重生后聚合累计和跌破陈旧基线，若不重置 quiet 会从崩溃前起算，恢复期
    # 一次普通 504 就能在"结果明明在流"时误判 exit(1)）。重置只会延迟检测，
    # 真楔死时计数恒不变，quiet 照常涨满。
    if res != st["last_res"]:
        st.update(last_res=res, last_res_ts=now)
    if tmo != st["last_tmo"]:
        st.update(last_tmo=tmo, last_tmo_ts=now)
    if s5 != st["last_5xx"]:
        st.update(last_5xx=s5, last_5xx_ts=now)
    quiet = now - st["last_res_ts"]
    recent_timeout = now - st["last_tmo_ts"] <= WEDGE_RESTART_S / 2
    # Redis 不可达等快速 5xx 不产生超时：只看 timeouts 会让"请求全 500、
    # 零结果"永远不触发 wedge。5xx 增长同样视为近期失败信号。
    recent_5xx = s5 > 0 and now - st["last_5xx_ts"] <= WEDGE_RESTART_S / 2
    if quiet >= WEDGE_RESTART_S and (recent_timeout or recent_5xx) and req > 0:
        log.critical("end-to-end wedge: no successful result for %.0fs "
                     "(requests=%d timeouts=%d server_5xx=%d, all roles ready)"
                     " — exiting for container restart", quiet, req, tmo, s5)
        for h in log.handlers:
            try:
                h.flush()
            except Exception:  # noqa: BLE001
                pass
        os._exit(1)


def supervisor_snapshot() -> dict:
    return {"role": "supervisor", "ready": True,
            "uptime_s": round(time.time() - _started_at, 1),
            "warm": {},
            "stats": {"crashes": {_wtag(r, i): n for (r, i), n in
                                  list(_crash_total.items())}},
            "expected": dict(EXPECTED_ROLES),
            "mailboxes": proc_common.mailbox_depths(mailboxes)}


_started_at = time.time()


def _safe_step(name: str, fn, *args):
    """监控子步骤独立兜底：一项失败不能连带跳过其余项。

    典型：Redis 重启失败曾让 _check_redis 抛错直接跳出 monitor_once，
    reap_hung/check_wedge/重生循环全部不执行，而心跳线程还在写——
    supervisor 看起来健康、实际编排停摆。"""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("monitor step %s failed (continue)", name)
        return None


def monitor_once() -> None:
    """单轮监控：redis 守护 + hang 清扫 + 端到端假死判死 + 退出重启
    （指数退避，稳定 60s 清零）。"""
    _safe_step("redis", _check_redis)
    _safe_step("sweep_inflight", _sweep_inflight)
    _safe_step("queue_bad", _watch_queue_bad)
    _safe_step("reap_hung", reap_hung)
    now = time.time()
    _safe_step("wedge", check_wedge, now)
    _safe_step("prune_starting", _prune_starting, now)
    for (role, i), p in list(procs.items()):
        try:
            alive = p.is_alive()   # mp.Process：非阻塞存活探测
        except Exception:  # noqa: BLE001  未 start/已关闭的 Process
            alive = False
        if alive:
            if now - _spawn_ts.get((role, i), now) > 60:
                _fail_count[(role, i)] = 0  # 稳定运行满 60s：退避清零
            continue
        # 死亡收口必须立即执行（不能等串行闸/退避）：
        #  - 心跳文件：否则 mtime 过期前 /health 仍把幽灵算进 role 计数
        #  - 在途重派：否则请求 120s 先 504，重派要等闸门释放/下一轮
        pid = p.pid
        try:
            os.unlink(os.path.join(HEALTH_DIR, f"worker-{pid}.json"))
        except OSError:
            pass
        # BRPOPLPUSH 级追踪：死者在途立即精确重派（其他 worker 接手，
        # 恢复时效秒级——不再等 TTL 猜）。tts/asr 认领队列持有在途账本。
        for q in (_tts_queue, _asr_queue):
            if q is not None:
                try:
                    n = q.requeue_worker(f"{role}-{i}-{pid}",
                                         max_redispatch=REDISPATCH_MAX)
                    if n:
                        log.warning("[%s-%d] requeued %d in-flight from dead "
                                    "worker (over-limit chunks discarded)",
                                    role, i, n)
                except Exception:  # noqa: BLE001  redis 瞬断：sweep 兜底
                    pass
        # 再生决策（串行启动闸/指数退避）
        if role in _SERIAL_ROLES and _role_start_busy(role):
            continue
        n = _fail_count.get((role, i), 0)
        if now < _next_ok.get((role, i), 0):
            continue  # 退避等待期
        delay = min(30.0, 0.5 * (2 ** n))
        _fail_count[(role, i)] = n + 1
        _crash_total[(role, i)] = _crash_total.get((role, i), 0) + 1
        _next_ok[(role, i)] = now + delay
        rc = p.exitcode if isinstance(p.exitcode, int) else -1
        log.warning("[%s-%d] exited(rc=%s), respawn in %.1fs (consecutive #%d)",
                    role, i, rc, delay, n + 1)
        procs.pop((role, i), None)
        try:
            spawn_one(role, i)
        except Exception as exc:  # noqa: BLE001  拉起失败不杀 supervisor 自身
            log.error("[%s-%d] respawn failed: %s", role, i, exc)
            # 槽位放回：否则该 key 已不在 procs，后续循环永远不会再重试，
            # 角色永久缺员（/health 报 missing 但 supervisor 不修）。
            procs.setdefault((role, i), p)


def _monitor_step() -> None:
    """主循环单步兜底：monitor_once 的残余异常不能杀死 supervisor 主线程。

    原实现主循环裸调 monitor_once——任一异常（Redis 重启失败、坏心跳
    数值、Process 状态异常）会退出 main；子进程非 daemon，multiprocessing
    atexit 会 join 无限运行的 worker，进程卡死不退出，而 daemon 心跳线程
    仍在写 /health（看起来健康、实际编排已停）。"""
    try:
        monitor_once()
    except Exception:  # noqa: BLE001
        log.exception("monitor cycle failed (supervisor kept alive)")


def create_mailboxes() -> None:
    """Mailbox 创建（全部有界，Actor 背压）：下游满 → 上游停 → 更上游拒。
    /health 的 queues 段展示各阶段入队/出队/最长排队。

    独立成函数：布线一致性可被测试覆盖。历史事故（随 aligner 删除已消解）：
    asr 曾按自己的序号取 per-aligner 结果队列，池尺寸不等时 KeyError 崩掉
    编排。"""
    # 每个邮箱分配固定序号（Redis 键 kokoro:q:{idx}:{name} 由序号决定，
    # supervisor 与子进程必须一致）
    idx = 0

    def _mb(name, cap):
        nonlocal idx
        q = proc_common.InstrumentedQueue(cap, index=idx, name=name)
        idx += 1
        return q

    mailboxes["g2p_queue"] = _mb("g2p_queue", MAX_PENDING_REQUESTS)      # Stage1：原始请求（gateway → g2p 池）
    mailboxes["tts_queue"] = proc_common.TtsClaimQueue(
        MAX_INFLIGHT_SENTENCES, index=idx, name="tts")  # Stage2：chunk 认领式队列
    idx += 1
    mailboxes["asr_queue"] = proc_common.TtsClaimQueue(
        WORKER_MAILBOX_SIZE, index=idx, name="asr")  # Stage3：对齐 job 认领式（in-flight 追踪同款）
    idx += 1
    mailboxes["gateway_queues"] = [_mb(f"gateway_queue_{i}", WORKER_MAILBOX_SIZE)
                                   for i in range(GATEWAY_WORKERS_N)]  # 终态回投（按 origin 路由）


def spawn_all() -> None:
    """按角色配置拉起全部子进程（独立成函数供布线测试复用）。"""
    for i in range(G2P_WORKERS_N):
        spawn_one("g2p", i)
    for i in range(GATEWAY_WORKERS_N):
        spawn_one("gateway", i)
    # 能力优先（每角色 ≥1 后 /v1、/tts 即完整可用），再扩容余量：
    #   阶段1：tts-0 → asr-0（能力就绪：/v1、/tts 均可服务）
    #   阶段2：剩余 tts/asr 交替串行扩容（单路 init 不抢卡）。
    #          NPU 串行原则不变（并行 init 抢卡曾放大坏态雪崩）
    spawn_one("tts", 0)
    _wait_started(("tts", 0))
    if ASR_N:
        spawn_one("asr", 0)
        _wait_started(("asr", 0))
    from itertools import zip_longest
    for tts_i, asr_i in zip_longest(range(1, TTS_N), range(1, ASR_N)):
        if tts_i is not None:
            spawn_one("tts", tts_i)
            _wait_started(("tts", tts_i))
        if asr_i is not None:
            spawn_one("asr", asr_i)
            _wait_started(("asr", asr_i))


def main() -> None:
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    check_shm()
    _clean_stale_heartbeats()
    create_mailboxes()
    global _redis_proc, _tts_queue, _asr_queue
    _redis_proc = proc_common.start_redis_server()  # 队列服务端：单容器自治，fail-fast
    _tts_queue = mailboxes["tts_queue"]             # sweeper 持队列实例（键由它决定）
    _asr_queue = mailboxes["asr_queue"]
    if _redis_proc is None:   # IPC_REDIS_URL 外部模式
        log.info("external redis ready (url=%s)", proc_common.IPC_REDIS_URL)
    else:
        log.info("redis-server ready (unix socket=%s, pid=%s)",
                 proc_common.IPC_REDIS_SOCKET, _redis_proc.pid)

    proc_common.start_heartbeat(tag="supervisor", pid=os.getpid(),
                                stats_fn=supervisor_snapshot)
    proc_common.dump_config(log.info)
    # 启动容量自检（仅 npu；非 npu/驱动不可用静默跳过）：默认配置按 64GB
    # 整卡设计，旧部署沿用旧 env 时可能超卡——超 95% 记 ERROR 但不拒绝启动。
    try:
        import pipeline

        cap = pipeline.capacity_check(TTS_N, ASR_N)
    except Exception:  # noqa: BLE001  自检失败不能阻塞编排
        cap = None
    if cap is not None:
        total_mb, phys_mb, over = cap
        if over:
            log.error("NPU capacity check: tts x%d + asr x%d ≈ %.1fGB > 95%% of "
                      "%.1fGB — reduce TTS_WORKERS/TTS_MAX_BATCH_SIZE or pin "
                      "explicit per-pool values", TTS_N, ASR_N,
                      phys_mb / 1024, total_mb / 1024)
        else:
            log.info("NPU capacity check: tts x%d + asr x%d ≈ %.1fGB / %.1fGB"
                     " (%.1f%%)", TTS_N, ASR_N, phys_mb / 1024, total_mb / 1024,
                     100.0 * phys_mb / max(total_mb, 1.0))
    spawn_all()
    log.info("orchestration complete gateway x%d + g2p x%d + tts x%d"
             " + asr x%d, listening :%d (reuseport hashed dispatch)",
             GATEWAY_WORKERS_N, G2P_WORKERS_N, TTS_N, ASR_N, PORT)

    while True:
        time.sleep(2)
        _monitor_step()


if __name__ == "__main__":
    main()

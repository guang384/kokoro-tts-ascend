"""gateway 观测层：心跳/僵尸清理、/health 全量聚合、Prometheus 指标数据面。

从 gateway 拆出的可观测性关注面；状态经 gateway_state(gst) 模块属性访问，
聚合引擎经 gateway_agg 模块属性调用（patch 可见）。路由层（/health /metrics
端点本体）在 gateway.py。
"""
import asyncio
import bisect
import json
import logging
import os
import time

import proc_common

import gateway_agg
import gateway_state as gst

log = logging.getLogger("kokoro.server")


def _snapshot_window(win) -> list:
    """deque → list 的突变容忍拷贝：/health、/metrics 已下放线程池，快照
    线程与事件循环线程并发读写同一 deque，迭代中被 append 会抛
    RuntimeError——空窗重试一次即可（下一轮心跳/抓取自然补齐）。"""
    for _ in range(2):
        try:
            return list(win)
        except RuntimeError:
            continue
    return []


def _quantile(sorted_vals: list, q: float) -> float | None:
    """线性插值分位数（滑动窗口 p50/p90/p99 用；空窗 None）。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _latency_bins() -> list:
    """区间计数列表（长度 = 档位数 + 1，最后一格为 > 总预算的尾巴）。

    测试会整体替换 gst._entry_stats，惰性重建避免 KeyError/短列表越界。"""
    bins = gst._entry_stats.get("latency_bins")
    if not isinstance(bins, list) or len(bins) != len(gst._LATENCY_BUCKETS) + 1:
        bins = gst._entry_stats["latency_bins"] = [0] * (len(gst._LATENCY_BUCKETS) + 1)
    return bins


def _worker_snapshot() -> dict:
    """本进程的实时状态快照（心跳落盘 + /health 聚合共用）。入口角色。"""
    wtag = os.getenv("WORKER_TAG") or f"pid{os.getpid()}"
    stats = dict(gst._entry_stats)
    bins = list(_latency_bins())
    stats["latency_bins"] = bins
    stats["latency_bounds"] = [float(b) for b in gst._LATENCY_BUCKETS]
    entries = _snapshot_window(gst._recent_latencies)
    if entries:
        # 以 sum/count 形态上报：/health 顶层 e2e 需跨网关加权合并
        stats["recent_count"] = len(entries)
        stats["recent_ms_sum"] = round(sum(e for e, _c in entries) * 1000, 1)
        stats["recent_chars_sum"] = round(sum(c for _e, c in entries), 1)
        # 滑动窗口分位数（秒）：/metrics 导出 p50/p90/p99 直读 Gauge
        durs = sorted(e for e, _c in entries)
        for q, key in ((0.50, "p50_s"), (0.90, "p90_s"), (0.99, "p99_s")):
            v = _quantile(durs, q)
            if v is not None:
                stats[key] = round(v, 3)
    failures = _snapshot_window(gst._recent_failures)
    if failures:
        stats["recent_failures"] = failures
    if gst._result_q is not None:
        stats["mailboxes"] = proc_common.mailbox_depths(
            {"gateway_queue": gst._result_q})
    return {"wtag": wtag, "pid": os.getpid(), "role": "gateway", "device": None,
            "ready": True,  # 入口无模型：uvicorn 起来即可服务
            "uptime_s": round(time.time() - gst._started_at, 1),
            "warm": {}, "stats": stats, "ts": round(time.time(), 1)}


def _write_own_health() -> None:
    """心跳落盘（原子替换，避免读到半截 JSON）。"""
    try:
        os.makedirs(gst._health_dir, exist_ok=True)
        tmp = os.path.join(gst._health_dir, f".tmp-{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_worker_snapshot(), f, ensure_ascii=False)
        os.replace(tmp, os.path.join(gst._health_dir, f"worker-{os.getpid()}.json"))
    except Exception as exc:  # noqa: BLE001  心跳失败只影响监控，不影响服务
        log.warning("heartbeat write failed: %s", exc)


def _health_dir_entries(prefixes=("worker-",), fresh_only: bool = True):
    """健康目录条目扫描 [(name, path, age_s)]：listdir + 前缀过滤 + 可选
    新鲜度过滤。/collect 聚合、/tts 的 asr 就绪探测、.tmp- 残片清理
    三个读者共用——listdir+mtime+json 是同步磁盘 IO，只写一处。

    worker-*.json 的删除权归 supervisor（reap_hung 依据 mtime/stuck_s
    判决）；网关只读，避免抢删导致 supervisor 看不到 stale 证据。

    TOCTOU 安全：getmtime 失败的条目静默跳过（写方 os.replace 的窗口，
    生产事故 20261110：.tmp-<pid> 存在窗口毫秒级，撞上即 FileNotFoundError）。"""
    try:
        names = sorted(os.listdir(gst._health_dir))   # 排序：/health 与指标输出确定性
    except OSError:
        return   # 目录瞬时不可用（人工清理/挂载抖动）：本轮空，5s 后自愈
    now = time.time()
    for name in names:
        if not name.startswith(tuple(prefixes)):
            continue
        path = os.path.join(gst._health_dir, name)
        try:
            age = now - os.path.getmtime(path)
        except OSError:
            continue
        if fresh_only and age > gst._HEALTH_STALE_AFTER:
            continue  # 超时未心跳：worker 已死（如被 SIGKILL），剔除
        yield name, path, age


def _sweep_orphan_shm() -> int:
    """回收无主音频 SHM 块（kokora_* 前缀且 mtime 超龄）。

    worker 被 SIGBUS 杀死时来不及走 unlink 兜底，半成品块永久占用 /dev/shm
    （20261109：64MB 写满 → 重生 worker 即死即泄漏，池无法恢复）。正常块的
    生命周期有界：单请求截止线 ≤WAIT_TIMEOUT_S(120s)（/tts 音频+时间戳
    共享同一预算）、聚合清扫 ≤240s——超龄线
    （gst.SHM_ORPHAN_MAX_AGE_S 公式有效值，默认 420s）之外还在的必是无主孤儿。非 POSIX/Windows 开发机目录不存在 → 0。"""
    try:
        names = os.listdir(gst._SHM_SWEEP_DIR)
    except OSError:
        return 0
    cutoff = time.time() - gst.SHM_ORPHAN_MAX_AGE_S
    n = 0
    for nm in names:
        if not nm.startswith("kokora_"):
            continue
        path = os.path.join(gst._SHM_SWEEP_DIR, nm)
        try:
            if os.path.getmtime(path) > cutoff:
                continue  # 仍在合法生命周期内
            os.unlink(path)
            n += 1
        except OSError:
            continue  # 已被并发回收 / Windows 句柄未释放
    if n:
        log.warning("swept %d orphan audio SHM blocks (age > %.0fs)",
                    n, gst.SHM_ORPHAN_MAX_AGE_S)
    return n


def _shm_stats() -> dict | None:
    """/dev/shm 使用情况（MB + 水位）；读不了返回 None。实现单一来源在
    proc_common.shm_usage（supervisor 启动自检共用）。"""
    return proc_common.shm_usage()


def _shm_used_pct() -> float:
    """/dev/shm 水位（0-100）；读不了（非 POSIX/Windows 开发机）返回 -1。"""
    return (_shm_stats() or {}).get("pct", -1.0)


def _cleanup_tmp_files() -> None:
    """只清 .tmp- 写一半的残片。worker-*.json 归 supervisor 所有：网关若按
    stale 线抢先删除，supervisor.reap_hung 就看不到 stale 证据（文件不存在
    ≠已死），SIGSTOP/心跳线程死亡但主循环仍活的 worker 永远不会被强杀
    （CPU E2E 实测复现：崩溃计数不增、僵尸进程长期占位）。"""
    for _name, path, age in _health_dir_entries(
            prefixes=(".tmp-",), fresh_only=False):
        if age > gst._HEALTH_STALE_AFTER:
            try:
                os.unlink(path)
            except OSError:
                pass


async def _heartbeat_loop() -> None:
    while True:
        try:  # 心跳线程绝不能死——死了会被 supervisor 判 hang 强杀
            _write_own_health()
            gateway_agg._sweep_leaks()
            _sweep_orphan_shm()
            pct = _shm_used_pct()
            if pct >= 80:  # SHM 高水位：再涨就 SIGBUS（长文本 /tts 的历史根因）
                _throttled_warn("shm", "/dev/shm 水位 %.0f%% —— 音频块写满即 SIGBUS，"
                                "加大 --shm-size 或降低 /tts 并发", pct)
            _cleanup_tmp_files()
        except Exception:  # noqa: BLE001
            log.exception("heartbeat/sweep error (ignored)")
        await asyncio.sleep(gst._HEALTHBEAT_INTERVAL)


def _throttled_warn(key: str, msg: str, *args) -> None:
    """/health 告警节流：同键 60s 最多一条。该端点会被 Docker HEALTHCHECK
    与外部监控秒级轮询——持续 degraded/积压不该每轮都刷一行日志。"""
    now = time.time()
    if now - gst._health_warn_ts.get(key, 0.0) < 60.0:
        return
    gst._health_warn_ts[key] = now
    log.warning(msg, *args)


# 角色→指标键映射（单一事实来源：/health workers 视图与 /metrics 共用，
# 新增角色/计数器只改这一处）
_ROLE_SPEC = {
    "gateway": {"done": "results", "sum": "latency_ms_sum",
                "err": ("timeouts", "rejections", "fast_reject")},
    "g2p": {"done": "requests", "sum": "g2p_ms",
            "err": ("errors", "ipc_dropped")},
    "tts": {"done": "jobs", "sum": "forward_ms",
            "err": ("forward_errors", "ipc_dropped"),
            "realtime_x": True},
    "asr": {"done": "jobs", "sum": "asr_ms",
            # ipc_dropped 不入列：asr 唯一的结果投递点是 chunk_ts 回投，
            # job_drops 已覆盖同一路径，再计会双计（err 虚高 2 倍）
            "err": ("job_drops", "job_errors")},
}


def _worker_row(w: dict, sup_crashes: dict) -> dict:
    """单 worker 指标行（_ROLE_SPEC 驱动）。/health 与 /metrics 共用，
    杜绝两处重复实现再次漂移（历史：collector 自算一遍导致三个指标族恒空）。"""
    role = w.get("role")
    entry = {"wtag": w.get("wtag"), "role": role,
             "device": w.get("device"), "ready": w.get("ready")}
    bad_warm = {k: v for k, v in (w.get("warm") or {}).items()
                if v not in ("ok", "off")}
    if bad_warm:
        entry["warm"] = bad_warm
    n_crash = (sup_crashes or {}).get(entry["wtag"])
    if n_crash:
        entry["crashes"] = n_crash
    spec = _ROLE_SPEC.get(role)
    if not spec:
        return entry
    s = w.get("stats") or {}
    done = s.get(spec["done"]) or 0
    entry["done"] = done
    if role == "asr":   # 批推可观测：生效档 + 安全补丁状态
        entry["batch_effective"] = s.get("batch_effective")
        entry["batch_patch_ok"] = s.get("batch_patch_ok")
    if role == "gateway":
        entry["requests"] = s.get("requests") or 0   # 本实例受理数
    if spec.get("sum") and done > 0 and s.get(spec["sum"]):
        entry["avg_ms"] = round(s[spec["sum"]] / done, 2)
    if spec.get("realtime_x"):
        fwd_ms = s.get("forward_ms") or 0
        if fwd_ms:
            entry["realtime_x"] = round(
                (s.get("audio_s") or 0) / (fwd_ms / 1000), 1)
    err = sum(s.get(k) or 0 for k in spec["err"])
    if err:
        entry["err"] = err
    return entry


def _record_latency(seconds: float, chars: int) -> None:
    """成功请求的统一延迟记账：区间桶（档位随 WAIT_TIMEOUT_S 推导）+ 总和 + 滑动窗口（/v1 与 /tts 同口径）。"""
    _latency_bins()[bisect.bisect_left(gst._LATENCY_BUCKETS, seconds)] += 1
    gst._entry_stats["latency_ms_sum"] += seconds * 1000
    gst._recent_latencies.append((seconds, chars))


def _collect() -> dict:
    """全量聚合数据源（/metrics 与精简 /health 共用，纯同步）。

    聚合内容：status/ready 判定（跨角色能力就绪）、队列 4 阶段、workers 全量
    （含 raw 快照）、e2e（跨网关合并）、SHM 水位、supervisor 崩溃计数。

    status=ok 需满足：全部角色就绪（ready）+ 无预热失败 + 无角色缺员
    （缺员判定来自 supervisor 心跳的 expected 声明，只影响 status）。
    """
    me = _worker_snapshot()
    workers, now = [], time.time()
    for _name, path, _age in _health_dir_entries():
        try:
            with open(path, encoding="utf-8") as f:
                workers.append(json.load(f))
        except (OSError, ValueError):
            continue
    if not workers:  # 刚启动、心跳尚未落盘的窗口期，至少展示自己
        workers = [me]

    # status 判定（缺员判定来自 supervisor 心跳的 expected 声明；诊断细节
    # 一律靠日志——预热失败/缺员/积压均有日志行，/health 只给结论）
    expected, missing = None, []
    for w in workers:
        if w.get("role") == "supervisor" and w.get("expected"):
            expected = w["expected"]
            break
    if expected:
        counts = {}
        for w in workers:
            counts[w.get("role")] = counts.get(w.get("role"), 0) + 1
        missing = [f"{r} {counts.get(r, 0)}/{n}"
                   for r, n in expected.items() if counts.get(r, 0) < n]
        for m in missing:
            _throttled_warn(f"missing:{m}", "missing role: %s (crash backoff or warming up)", m)

    # ── 就绪语义（串行启动后变更）：ready = 能力就绪 = 每个期望角色 ≥1 个
    # ready 工人即可接流量；扩容期其余 worker 热身中不算 not-ready。
    # status：ok=可服务且无异常；starting=某角色还没有任何 ready 工人；
    # degraded=可服务但存在预热失败 / 缺员进程（CPU 降级已移除）。
    expected_roles = set((expected or {}).keys())
    role_ready = {}
    for w in workers:
        if w.get("ready"):
            r = w.get("role")
            role_ready[r] = role_ready.get(r, 0) + 1
    if expected_roles:
        ready = all(role_ready.get(r, 0) >= 1 for r in expected_roles)
    else:
        # 无 supervisor 心跳 = 编排者不在（或刚启动）。**不得**用自身快照
        # （gateway 恒 ready=True）推断就绪——那会在整条流水线空转时谎报 ok
        # 给 K8s 接线（fail-open）。按"未就绪"处理，由 503 摘流量。
        ready = False
    any_fail = any((w.get("warm") or {}).get("g2p") == "fail"
                   or (w.get("warm") or {}).get("kokoro") == "fail"
                   or (w.get("warm") or {}).get("funasr") == "fail" for w in workers)
    # 近期服务端 5xx：Redis 不可达等"快速失败"不会产生超时/结果计数，
    # 只看 warm/missing 会让 status 永远 ok。60s 滑窗内出现过 5xx 即
    # degraded（200，不触发重启；wedge 判死由 supervisor 另管）。
    recent_5xx = any(
        now - float((w.get("stats") or {}).get("last_5xx_ts") or 0) < 60.0
        for w in workers if w.get("role") == "gateway")
    # degraded = 可服务但存在异常（预热失败/缺员/近期 5xx）。CPU 降级已移除：
    # NPU 持续失败走 fail-fast（warm=fail → 503 → 编排层重启容器）
    degraded = any_fail or bool(missing) or recent_5xx
    if not ready:
        status = "starting"
    elif degraded:
        status = "degraded"
    else:
        status = "ok"

    # ── 队列排队程度（supervisor 心跳的深度探针）──
    # 按"用户请求流经的顺序"排列：受理(g2p) → 前向认领(tts) → 对齐(asr)
    # → 终态回投(gateway)。每阶段 {depth, in, out, avg_wait_ms, bad}（多实例
    # depth/in/out/bad 求和、avg_wait 取最慢实例）；积压 ≥85% 只写日志
    # （每实例明细看 supervisor 心跳文件）。
    _STAGE_ORDER = ["g2p", "tts", "asr", "gateway"]
    _ZERO_Q = {"depth": 0, "in": 0, "out": 0, "avg_wait_ms": 0, "bad": 0}
    acc = {}
    sup = next((w for w in workers if w.get("role") == "supervisor"), None)
    depths = (sup or {}).get("mailboxes") or {}
    sup_crashes = ((sup or {}).get("stats") or {}).get("crashes") or {}
    # tts 队列真实容量是 MAX_INFLIGHT_SENTENCES（认领队列构造参数），
    # 不落 WORKER_MAILBOX_SIZE 默认——否则积压告警线 0.85×1024 永不可达
    caps = {"g2p_queue": gst._MAX_PENDING_REQUESTS,
            "tts_queue": proc_common.env_int("MAX_INFLIGHT_SENTENCES",
                                             minimum=1)}
    _STAGE_OF = {"g2p_queue": "g2p", "tts_queue": "tts",
                 "asr_queue": "asr"}
    for name, st in depths.items():
        if not isinstance(st, dict):
            st = {"depth": st}
        if _STAGE_OF.get(name):
            stage = _STAGE_OF[name]
        elif name.startswith("gateway_queues["):
            stage = "gateway"
        else:
            stage = None
        if stage:
            tgt = acc.setdefault(stage, dict(_ZERO_Q))
            tgt["depth"] += st.get("depth") or 0
            tgt["in"] += st.get("in") or 0
            tgt["out"] += st.get("out") or 0
            tgt["bad"] += st.get("bad") or 0
            tgt["avg_wait_ms"] = max(tgt["avg_wait_ms"], st.get("avg_wait_ms") or 0)
        depth = st.get("depth")
        if depth is None:
            continue
        cap = caps.get(name, gst._WORKER_MAILBOX_SIZE)
        # Redis 后端：depth = LLEN = 队列真实长度，85% 告警即真实水位
        # （无 enq-deq 记账漂移/幻影问题）；闸门/假死判死均不依赖该值
        if cap and depth >= 0.85 * cap:
            _throttled_warn(f"backlog:{name}",
                            "queue backlog: %s %d/%d (downstream not consuming fast enough)",
                            name, depth, cap)
    queues = {k: acc[k] for k in _STAGE_ORDER if k in acc}  # 固定按请求流经顺序输出

    # workers 精简列表：_ROLE_SPEC 驱动的单行指标（/health 与 /metrics 共用）
    shown = [_worker_row(w, sup_crashes) for w in workers
             if w.get("role") != "supervisor"]

    # ── 端到端服务级指标（跨全部网关实例聚合）：请求/成功总数、滑动平均
    # 处理时间与 text 字符长度（近 50 次/实例，加权合并）、最近 3 次失败
    # （按时间排序取最新 3 条）。全局口径，不随实例数翻倍。──
    gw_snaps = [w for w in workers if w.get("role") == "gateway"]

    def _sum(field: str) -> int:
        return sum((w.get("stats") or {}).get(field) or 0 for w in gw_snaps)

    ms_sum = sum((w.get("stats") or {}).get("recent_ms_sum") or 0 for w in gw_snaps)
    ms_cnt = sum((w.get("stats") or {}).get("recent_count") or 0 for w in gw_snaps)
    chars_sum = sum((w.get("stats") or {}).get("recent_chars_sum") or 0 for w in gw_snaps)
    failures = []
    for w in gw_snaps:
        failures.extend((w.get("stats") or {}).get("recent_failures") or [])
    failures.sort(key=lambda f: f.get("time") or "", reverse=True)  # 最新在前

    e2e = {
        "requests": _sum("requests"),          # 受理总数
        "successes": _sum("results"),          # 成功总数
        "timed_out": _sum("timeouts"),         # 超时累计
        "rejected": _sum("rejections"),        # 背压拒绝累计（队列满）
        "fast_reject": _sum("fast_reject"),    # 准入快速拒绝累计（排队积压预估超预算）
        "cancelled": _sum("cancelled"),        # 断连取消累计
        "failed": _sum("failed"),              # 终态失败累计（超时+合成失败+500；与 recent_failures 对账）
        "server_5xx": _sum("server_5xx"),      # 服务端 5xx 累计（Redis 不可达等快速失败）
        "avg_ms_recent": round(ms_sum / ms_cnt, 1) if ms_cnt else None,
        "avg_chars_recent": round(chars_sum / ms_cnt, 1) if ms_cnt else None,
        "recent_failures": failures[:3],       # 跨网关合并后的最近 3 次（最新在前）
    }

    # 跨网关延迟桶聚合（SO_REUSEPORT 下连续抓取命中不同进程——直方图必须
    # 用全实例求和，读本进程 gst._entry_stats 会让 rate/quantile 剧烈跳动）。
    # 心跳带 latency_bounds 校验；同容器内所有网关同版本同配置，边界必然
    # 一致——不一致（异常心跳/配置漂移）时跳过该实例的桶计数，绝不把
    # 错误粒度的 bins 直接相加污染直方图。
    nbins = len(gst._LATENCY_BUCKETS) + 1
    expect_bounds = [float(b) for b in gst._LATENCY_BUCKETS]
    lat = {"bins": [0] * nbins, "ms_sum": 0.0, "count": 0}
    for w in gw_snaps:
        st = w.get("stats") or {}
        bins = st.get("latency_bins")
        if (st.get("latency_bounds") == expect_bounds
                and isinstance(bins, list) and len(bins) == nbins
                and all(isinstance(v, (int, float)) for v in bins)):
            for i, v in enumerate(bins):
                lat["bins"][i] += v
        lat["ms_sum"] += st.get("latency_ms_sum") or 0
    lat["count"] = sum(lat["bins"])
    # 滑动窗口分位数聚合：分位数在数学上不可跨实例合并，两个入口经
    # SO_REUSEPORT 均衡分流、分布近似——取保守最大值（偏高告警优于偏低）
    for key in ("p50_s", "p90_s", "p99_s"):
        vals = [(w.get("stats") or {}).get(key) for w in gw_snaps]
        lat[key] = max((v for v in vals if v is not None), default=None)

    return {
        "status": status,   # ok=可服务 · starting=预热中/缺员 · degraded=可服务但异常
        "ready": ready,               # 能力就绪：每期望角色 ≥1 ready 即可接流量（空 workers → False）
        "version": gst._image_version,
        "shm": _shm_stats(),  # /dev/shm 使用情况（音频 SHM 块所在 tmpfs；写满=SIGBUS 根因，水位告警见日志）
        "e2e": e2e,           # 端到端服务级指标（全局口径，跨网关聚合）
        "queues": queues,
        "workers": shown,
        "workers_raw": workers,       # 原始快照（/metrics 的 per-worker gauge 源）
        "crashes": sup_crashes or {}, # 崩溃计数映射（/metrics 源）
        "latency": lat,               # 跨网关延迟桶聚合（/metrics 直方图源）
    }


def health_payload() -> dict:
    """精简结论卡（人类/告警可读；未就绪 503 / 就绪 200，readiness 语义）。

    完整数值指标走 /metrics（Prometheus）。"""
    d = _collect()
    return {
        "status": d["status"],
        "ready": d["ready"],
        "version": d["version"],
        "shm": d["shm"],
        "recent_failures": d["e2e"]["recent_failures"],
    }


# ── 健康聚合与指标数据面（HTTP 路由在 gateway.py：/health、/metrics）──


if gst.METRICS_ENABLED:
    from prometheus_client import CollectorRegistry
    from prometheus_client.core import (CounterMetricFamily,
                                        GaugeMetricFamily,
                                        HistogramMetricFamily)

    class _KokoroCollector:
        """Prometheus 自定义收集器：抓取时现算全量聚合（无陈旧 gauge）。"""

        def collect(self):
            data = _collect()

            # e2e 计数器（聚合绝对值；自定义 Collector 输出绝对值 Counter 合法）
            for name, doc, key in (
                    ("kokoro_e2e_requests", "受理请求总数", "requests"),
                    ("kokoro_e2e_successes", "成功总数", "successes"),
                    ("kokoro_e2e_timeouts", "超时累计", "timed_out"),
                    ("kokoro_e2e_rejected", "背压拒绝累计（队列满）", "rejected"),
                    ("kokoro_e2e_fast_reject",
                     "准入快速拒绝累计（排队积压预估超等待预算）", "fast_reject"),
                    ("kokoro_e2e_cancelled", "客户端断开取消累计", "cancelled"),
                    ("kokoro_e2e_failed",
                     "终态失败累计（超时+合成失败+服务端 500）", "failed"),
                    ("kokoro_e2e_server_5xx",
                     "服务端 5xx 累计（Redis 不可达等快速失败）", "server_5xx")):
                c = CounterMetricFamily(name, doc)
                c.add_metric([], data["e2e"][key])
                yield c

            # 队列 4 阶段（g2p → tts → asr → gateway）
            q_fams = {key: cls(name, doc, labels=["stage"]) for key, name, cls, doc in (
                    ("depth", "kokoro_queue_depth", GaugeMetricFamily, "队列当前堆积"),
                    ("in", "kokoro_queue_in_total", CounterMetricFamily, "队列累计入队"),
                    ("out", "kokoro_queue_out_total", CounterMetricFamily, "队列累计出队"),
                    ("avg_wait_ms", "kokoro_queue_avg_wait_ms", GaugeMetricFamily,
                     "近期平均排队毫秒"),
                    # 坏 member 丢弃（载荷损坏/外部误写）——非零即需排查
                    ("bad", "kokoro_queue_bad_total", CounterMetricFamily,
                     "坏 member 丢弃累计"))}
            for stage, q in data["queues"].items():
                for key, fam in q_fams.items():
                    fam.add_metric([stage], q[key])
            yield from q_fams.values()

            # 每 worker 指标族（声明表：短名 → (指标名, 类, help, labels)；
            # npu 内存明细一族一戳：allocated/reserved 来自 torch，context 是
            # CANN 上下文估算（tts 450 / asr 153），estimated = reserved+context，
            # planned = 0.98×budget+context）
            W = {short: cls(name, doc, labels=list(lb))
                 for short, name, cls, doc, lb in (
                    ("done", "kokoro_worker_done_total", CounterMetricFamily,
                     "完成工作单元数", ("worker", "role")),
                    ("err", "kokoro_worker_errors_total", CounterMetricFamily,
                     "错误累计", ("worker", "role")),
                    ("crashes", "kokoro_worker_crashes_total", CounterMetricFamily,
                     "累计崩溃次数", ("worker",)),
                    ("ready", "kokoro_worker_ready", GaugeMetricFamily,
                     "就绪状态 0/1", ("worker", "role")),
                    ("realtime_x", "kokoro_worker_realtime_x", GaugeMetricFamily,
                     "实时倍速（tts）", ("worker",)),
                    ("avg_ms", "kokoro_worker_avg_ms", GaugeMetricFamily,
                     "单位工作量平均毫秒", ("worker", "role")),
                    ("stuck", "kokoro_worker_stuck_seconds", GaugeMetricFamily,
                     "busy 卡住秒数", ("worker", "role")),
                    ("npu_reserved", "kokoro_worker_npu_mem_reserved_mb",
                     GaugeMetricFamily, "NPU 显存预留 MB（torch 缓存池）", ("worker",)),
                    ("npu_budget_pct", "kokoro_worker_npu_budget_pct",
                     GaugeMetricFamily, "NPU 预算占比（reserved/budget）", ("worker",)),
                    ("npu_allocated", "kokoro_worker_npu_mem_allocated_mb",
                     GaugeMetricFamily, "NPU 当前分配 MB", ("worker", "role")),
                    ("npu_estimated", "kokoro_worker_npu_mem_estimated_mb",
                     GaugeMetricFamily, "NPU 估算物理占用 MB（reserved+CANN ctx）",
                     ("worker", "role")),
                    ("npu_planned", "kokoro_worker_npu_mem_planned_mb",
                     GaugeMetricFamily, "NPU 容量核算 MB（0.98×budget+CANN ctx）",
                     ("worker", "role")),
                    ("npu_budget", "kokoro_worker_npu_mem_budget_mb",
                     GaugeMetricFamily, "NPU worker 预算 MB", ("worker", "role")),
                    ("npu_context", "kokoro_worker_npu_mem_context_mb",
                     GaugeMetricFamily, "CANN 上下文估算 MB", ("worker", "role")),
                    ("npu_card", "kokoro_worker_npu_mem_card_pct",
                     GaugeMetricFamily, "估算占用/整卡 %", ("worker", "role")),
                    ("npu_process", "kokoro_worker_npu_mem_process_mb",
                     GaugeMetricFamily, "npu-smi 实测进程内存 MB", ("worker", "role")),
                    ("npu_delta", "kokoro_worker_npu_mem_delta_mb",
                     GaugeMetricFamily, "实测 - planned MB（正=公式低估）",
                     ("worker", "role")),
                    ("npu_ctx_actual", "kokoro_worker_npu_mem_context_actual_mb",
                     GaugeMetricFamily, "实测 - torch reserved MB（实际非 torch 开销）",
                     ("worker", "role")),
                    ("npu_limit", "kokoro_worker_npu_mem_limit_mb",
                     GaugeMetricFamily, "fraction 允许的进程上限 MB（1.05×budget+context）",
                     ("worker", "role")),
                    ("cache_hits", "kokoro_g2p_cache_hits_total", CounterMetricFamily,
                     "g2p 缓存命中", ("worker",)),
                    ("cache_misses", "kokoro_g2p_cache_misses_total", CounterMetricFamily,
                     "g2p 缓存未命中", ("worker",)),
                    # 灰度期观察 fp16 流量占比（按批计；0 不输出）
                    ("fp16_batches", "kokoro_worker_fp16_batches_total",
                     CounterMetricFamily, "fp16 前向批数", ("worker",)),
                    # asr 批推可观测：生效档与安全补丁状态（0=未装→已自动降单条）
                    ("batch_effective", "kokoro_worker_batch_effective",
                     GaugeMetricFamily, "ASR 批推生效批宽", ("worker", "role")),
                    ("batch_patch", "kokoro_worker_batch_patch_ok",
                     GaugeMetricFamily, "ASR 批推安全补丁状态（1=已安装）",
                     ("worker", "role")))}

            # per-worker：直接消费 _collect() 的指标行（与 /health 同源同值，
            # 杜绝二次推导漂移）；npu/stuck/cache 补充自原始快照
            raw_by_tag = {w.get("wtag"): w for w in data["workers_raw"]}
            for row in data["workers"]:
                tag = row.get("wtag") or f"{row.get('role')}-?"
                role = row.get("role") or "?"
                if row.get("done"):
                    W["done"].add_metric([tag, role], row["done"])
                if row.get("err"):
                    W["err"].add_metric([tag, role], row["err"])
                if row.get("avg_ms") is not None:
                    W["avg_ms"].add_metric([tag, role], row["avg_ms"])
                W["ready"].add_metric([tag, role], 1 if row.get("ready") else 0)
                if row.get("realtime_x") is not None:
                    W["realtime_x"].add_metric([tag], row["realtime_x"])
                if row.get("batch_effective") is not None:
                    W["batch_effective"].add_metric([tag, role], row["batch_effective"])
                if row.get("batch_patch_ok") is not None:
                    W["batch_patch"].add_metric(
                        [tag, role], 1 if row["batch_patch_ok"] else 0)
                w_raw = raw_by_tag.get(tag) or {}
                st = w_raw.get("stats") or {}
                for key in ("cache_hits", "cache_misses"):
                    if st.get(key):
                        W[key].add_metric([tag], st[key])
                if role == "tts" and st.get("fp16_jobs"):
                    W["fp16_batches"].add_metric([tag], st["fp16_jobs"])
                # tts/asr 同一套内存明细（worker 快照缺哪项该指标自动缺省，
                # 不产生 0 值误导）
                nm = w_raw.get("npu_mem") or {}
                if nm.get("reserved_mb") is not None:
                    W["npu_reserved"].add_metric([tag], nm["reserved_mb"])
                if nm.get("budget_pct") is not None:
                    W["npu_budget_pct"].add_metric([tag], nm["budget_pct"])
                for key, short in (("allocated_mb", "npu_allocated"),
                                   ("estimated_mb", "npu_estimated"),
                                   ("planned_mb", "npu_planned"),
                                   ("budget_mb", "npu_budget"),
                                   ("context_mb", "npu_context"),
                                   ("card_pct", "npu_card"),
                                   ("process_mb", "npu_process"),
                                   ("delta_mb", "npu_delta"),
                                   ("context_actual_mb", "npu_ctx_actual"),
                                   ("limit_mb", "npu_limit")):
                    if nm.get(key) is not None:
                        W[short].add_metric([tag, role], nm[key])
                if w_raw.get("stuck_s"):
                    W["stuck"].add_metric([tag], w_raw["stuck_s"])
            # 崩溃计数来自 supervisor 快照映射（不随 worker 心跳上报）
            for wtag, n_crash in (data.get("crashes") or {}).items():
                if n_crash:
                    W["crashes"].add_metric([wtag], n_crash)
            yield from W.values()

            shm = data.get("shm")
            if shm:
                for key, doc in (("used_mb", "/dev/shm 已用 MB"),
                                 ("total_mb", "/dev/shm 总量 MB"),
                                 ("pct", "/dev/shm 水位百分比")):
                    g = GaugeMetricFamily(f"kokoro_shm_{key}", doc)
                    g.add_metric([], shm[key])
                    yield g
            g = GaugeMetricFamily("kokoro_degraded", "降级状态 0/1")
            g.add_metric([], 1 if data["status"] == "degraded" else 0)
            yield g
            bi = GaugeMetricFamily("kokoro_build_info", "构建信息（值恒 1）",
                                   labels=["version"])
            bi.add_metric([data["version"]], 1)
            yield bi

            # 请求时延直方（跨网关桶求和：SO_REUSEPORT 下抓取随机命中实例，
            # 读单进程计数会让 rate/quantile 剧烈跳动）。桶必须单调累积
            # （Prometheus le 语义）：histogram_quantile 对非累积桶会给出
            # 错误分位数；档位按 WAIT_TIMEOUT_S 推导（默认 1/2/…/60/120）。
            lat = data["latency"]
            cum, buckets = 0, []
            for edge, n in zip(gst._LATENCY_BUCKETS, lat["bins"]):
                cum += n
                buckets.append([gst._bucket_label(edge), cum])
            buckets.append(["+Inf", lat["count"]])
            h = HistogramMetricFamily(
                "kokoro_request_duration_seconds", "请求端到端耗时",
                buckets=buckets,
                sum_value=lat["ms_sum"] / 1000.0)
            yield h
            # 滑动窗口分位数直读 Gauge（近 50 次成功/网关，跨网关取最大）；
            # 直方图分桶再细也只能在桶内线性插值，直读值用于精确定位。
            for q in ("p50", "p90", "p99"):
                v = lat.get(f"{q}_s")
                if v is not None:
                    g = GaugeMetricFamily(
                        f"kokoro_request_duration_{q}_seconds",
                        f"请求端到端耗时 {q}（滑动窗口近 50 次/网关，跨网关取最大）")
                    g.add_metric([], v)
                    yield g

    _PROM_REGISTRY = {"r": None}

    def _prom_registry():
        """惰性构建注册表：导入期不执行 collect（原实现在 import 时注册并
        立即 collect，dependent 名字尚未定义 + 导入期磁盘 IO，生产事故源）。"""
        r = _PROM_REGISTRY["r"]
        if r is None:
            r = CollectorRegistry(auto_describe=True)
            r.register(_KokoroCollector())
            _PROM_REGISTRY["r"] = r
        return r

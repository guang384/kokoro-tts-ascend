"""Kokoro TTS 服务（昇腾 NPU 版）· 统一队列架构（入口角色）。

- 入口进程：HTTP 接入 → 整单入 g2p_queue → g2p 池切句/G2P → tts 池认领
  组批/前向 → synth_part 逐段直投本入口 → 【本进程按 req_id 聚合】
  → 纯 TTS 直接编码返回；/tts 的 chunk 前向后即派发对齐（tts → asr → chunk_ts，
    与剩余合成重叠执行），音频与时间戳两账本独立收齐。
- G2P 在 g2p 池、FunASR 在 asr 池——入口进程无模型，内存轻、启动快。
- 入口进程间以 SO_REUSEPORT 均匀分接连接；日志带 [gw-N] 标签。
- 消息契约见 protocol.py（单一事实来源）；环境变量见 README.md。
"""
import asyncio
import base64
import bisect
import collections
import io
import json
import logging
import os
import queue
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, List, Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# 离线属性强制（与 proc_common 同一约定）
os.environ["HF_HUB_OFFLINE"] = "1"

import pipeline
import proc_common
import protocol

# origin = 本网关在 gateway_queues 里的下标（supervisor 经 WORKER_IDX 注入）；
# 未注入（异常环境）按 0 兜底——绝不让 int("") 这类解析错误在首个请求时炸掉受理路径
_worker_idx = os.getenv("WORKER_IDX", "")
_worker_origin = int(_worker_idx) if _worker_idx.strip().lstrip("-").isdigit() else 0
_entry_wait_s = proc_common.env_float("WAIT_TIMEOUT_S", 120.0, minimum=1.0)  # 单请求总预算（秒；/tts 音频+时间戳共享，与 SHM 孤儿回收公式同一解析器，小数秒合法）


def _derive_latency_buckets(timeout_s: float) -> tuple:
    """请求耗时直方图档位：固定阶梯中小于总预算的档 + 总预算本身。

    成功请求耗时必然 ≤ 单请求总预算（_record_latency 只记成功路径），
    超过预算的桶永远为空，所以最后一档动态钉在 WAIT_TIMEOUT_S 上。
    默认 120s → 1/2/5/10/15/20/30/45/60/120；预算调小（如 60s）时
    自动裁掉 120，不会导出永远为空的桶。"""
    return tuple(b for b in _LATENCY_BUCKET_LADDER
                 if b < timeout_s) + (float(timeout_s),)


_LATENCY_BUCKET_LADDER = (1, 2, 5, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300)
_LATENCY_BUCKETS = _derive_latency_buckets(_entry_wait_s)


def _bucket_label(edge: float) -> str:
    """le 标签：整数档位不带小数点（120 而非 120.0），小数预算原样保留。"""
    return str(int(edge)) if float(edge).is_integer() else format(edge, "g")


FAST_REJECT_MARGIN = proc_common.env_float("FAST_REJECT_MARGIN", 1.15, minimum=1.0)  # 准入快速拒绝冗余系数：预估等待 > 预算×系数 才拒（EMA 是估计值，边界放行）
_MAX_PENDING_REQUESTS = proc_common.env_int("MAX_PENDING_REQUESTS", 200, minimum=1)  # /health 队列积压判定用
_WORKER_MAILBOX_SIZE = proc_common.env_int("WORKER_MAILBOX_SIZE", 1024, minimum=1)   # 同上（与 supervisor 同 env）


class _WorkerTagFilter(logging.Filter):
    def filter(self, record):
        record.wtag = os.getenv("WORKER_TAG") or f"pid{os.getpid()}"
        return True


logging.basicConfig(level=proc_common.log_level(),
                    format="[%(wtag)s] %(levelname)s:%(name)s:%(message)s")
# filter 必须挂在 handler 上：logger 级 filter 对子 logger 传播上来的记录不生效
for _h in logging.getLogger().handlers:
    _h.addFilter(_WorkerTagFilter())
log = logging.getLogger("kokoro.server")

_sample_rate = proc_common.env_int("KOKORO_SAMPLE_RATE", 24000, minimum=8000)
_speed_floor = proc_common.env_float("TTS_SPEED_FLOOR", 0.5, minimum=0.1)
_max_chars = proc_common.env_int("TTS_MAX_INPUT_CHARS", 2000, minimum=1)
_default_voice = os.getenv("KOKORO_VOICE", "zf_001")
_ffmpeg_timeout_s = proc_common.env_float("FFMPEG_TIMEOUT_S", 30.0, minimum=5.0)
_image_version = os.getenv("IMAGE_VERSION", "dev")  # 镜像版本号（构建期注入日期版本，本地直跑为 dev）
METRICS_ENABLED = proc_common.env_bool("METRICS_ENABLED", True)  # /metrics 开关（0 关闭返回 404）

# ── 心跳与 /health 聚合 ──
_health_dir = os.getenv("KOKORO_HEALTH_DIR", "/tmp/kokoro-health")
_HEALTHBEAT_INTERVAL = 5.0
# 心跳陈旧阈值：与 supervisor 判 hang 用同一 env（默认 30s = 6×心跳周期），
# 原硬编码 15s 会在心跳轻微抖动时把活 worker 从视图剔除 → readiness 抖动
_HEALTH_STALE_AFTER = proc_common.env_float("HEARTBEAT_STALE_S", 30.0, minimum=15.0)
_heartbeat_task: Optional[asyncio.Task] = None
_started_at = time.time()


# ── 入口运行时：队列句柄注入 + 结果泵 + 请求聚合 ──
_g2p_queue = None        # 原始请求队列（g2p 池拉取切句/G2P）
_ts_state = {}           # req_id → {"need", "items": {idx: (samples, ts)}, "created"}（TS 账本）
_CHUNK_GAP_S = 0.06      # chunk 间静音（_concat 插入）——TS 偏移必须同计（审计 F1：
                         # 时间戳与交付音频时间轴对齐，否则第 k 个 chunk 偏早 60ms×k）
_result_q = None         # 本入口的结果队列（compute 直投 synth_part/chunk_ts）
_main_loop = None        # 服务事件循环（结果泵回调）
_pending = {}            # ("req", req_id)/("ts", req_id) → Future
_request_meta = {}       # req_id → {"needs_align", "created"}（存续期 = _synth 在途期）
_agg_state = {}          # req_id → {"need", "idxs", "blocks", "texts", "created"}
# 重试批的重复 synth_part 块：ASR job 可能仍按名引用它（worker 在
# "已投 synth_part、未投 asr_job" 窗口死亡时，重试块是唯一副本）。
# 对齐请求在 TS 账本结束前延迟 unlink，由端点 finally / _sweep_leaks 收口。
_retired_blocks = {}     # req_id → [(block, created_ts), ...]
# _ts_state 见上（TS 账本）：chunk_ts 流水乱序到达，与 _agg_state 并行收齐
_entry_stats = {"requests": 0, "results": 0, "timeouts": 0,
                "failed": 0, "rejections": 0, "fast_reject": 0,
                "server_5xx": 0, "last_5xx_ts": 0.0,
                "swept": 0, "cancelled": 0,
                "latency_bins": [0] * (len(_LATENCY_BUCKETS) + 1),
                "latency_ms_sum": 0.0}
# 进程状态（/metrics 注册时 collect() 会被立即调用——状态定义必须在注册之前）
_recent_latencies = collections.deque(maxlen=50)  # 最近 50 次成功请求的 (端到端耗时秒, text 字符数)
_recent_failures = collections.deque(maxlen=3)    # 最近 3 次失败 [{time, status, reason}]



class _SynthError(Exception):
    """下游失败终态（带错误分级：invalid_input→400，compute→502）。"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


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

    测试会整体替换 _entry_stats，惰性重建避免 KeyError/短列表越界。"""
    bins = _entry_stats.get("latency_bins")
    if not isinstance(bins, list) or len(bins) != len(_LATENCY_BUCKETS) + 1:
        bins = _entry_stats["latency_bins"] = [0] * (len(_LATENCY_BUCKETS) + 1)
    return bins


def _worker_snapshot() -> dict:
    """本进程的实时状态快照（心跳落盘 + /health 聚合共用）。入口角色。"""
    wtag = os.getenv("WORKER_TAG") or f"pid{os.getpid()}"
    stats = dict(_entry_stats)
    bins = list(_latency_bins())
    stats["latency_bins"] = bins
    stats["latency_bounds"] = [float(b) for b in _LATENCY_BUCKETS]
    entries = _snapshot_window(_recent_latencies)
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
    failures = _snapshot_window(_recent_failures)
    if failures:
        stats["recent_failures"] = failures
    if _result_q is not None:
        stats["mailboxes"] = proc_common.mailbox_depths(
            {"gateway_queue": _result_q})
    return {"wtag": wtag, "pid": os.getpid(), "role": "gateway", "device": None,
            "ready": True,  # 入口无模型：uvicorn 起来即可服务
            "uptime_s": round(time.time() - _started_at, 1),
            "warm": {}, "stats": stats, "ts": round(time.time(), 1)}


def _write_own_health() -> None:
    """心跳落盘（原子替换，避免读到半截 JSON）。"""
    try:
        os.makedirs(_health_dir, exist_ok=True)
        tmp = os.path.join(_health_dir, f".tmp-{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_worker_snapshot(), f, ensure_ascii=False)
        os.replace(tmp, os.path.join(_health_dir, f"worker-{os.getpid()}.json"))
    except Exception as exc:  # noqa: BLE001  心跳失败只影响监控，不影响服务
        log.warning("heartbeat write failed: %s", exc)


def _sweep_leaks() -> None:
    """TTL 清扫：请求方已放弃的聚合状态/孤儿 Future——部分失败场景下
    （worker 崩溃丢块、asr 作业丢失）唯一的回收手段，防 _pending/
    _agg_state/_ts_state/SHM 块随时间单调增长。"""
    now = time.time()
    for key, fut in list(_pending.items()):
        if fut.done():
            _pending.pop(key, None)
    for rid, meta in list(_request_meta.items()):
        if now - meta["created"] > 3 * _entry_wait_s:
            _request_meta.pop(rid, None)
    for rid, st in list(_agg_state.items()):
        if now - st["created"] > 2 * _entry_wait_s:
            _agg_state.pop(rid, None)
            _unlink_blocks(st["blocks"])
            _entry_stats["swept"] += 1
    for rid, st in list(_ts_state.items()):
        if now - st["created"] > 2 * _entry_wait_s:
            _ts_state.pop(rid, None)  # 孤儿 TS 账本（fut_ts 已被端点 finally 摘除）
    for rid, items in list(_retired_blocks.items()):
        keep, old = [], []
        for block, ts in items:
            if now - ts > 2 * _entry_wait_s:
                old.append(block)
            else:
                keep.append((block, ts))
        if old:
            _unlink_blocks(old)
        if keep:
            _retired_blocks[rid] = keep
        else:
            _retired_blocks.pop(rid, None)


def _sweep_orphan_shm() -> int:
    """回收无主音频 SHM 块（kokora_* 前缀且 mtime 超龄）。

    worker 被 SIGBUS 杀死时来不及走 unlink 兜底，半成品块永久占用 /dev/shm
    （20261109：64MB 写满 → 重生 worker 即死即泄漏，池无法恢复）。正常块的
    生命周期有界：单请求截止线 ≤WAIT_TIMEOUT_S(120s)（/tts 音频+时间戳
    共享同一预算）、聚合清扫 ≤240s——超龄线
    （SHM_ORPHAN_MAX_AGE_S 公式有效值，默认 420s）之外还在的必是无主孤儿。非 POSIX/Windows 开发机目录不存在 → 0。"""
    try:
        names = os.listdir(_SHM_SWEEP_DIR)
    except OSError:
        return 0
    cutoff = time.time() - SHM_ORPHAN_MAX_AGE_S
    n = 0
    for nm in names:
        if not nm.startswith("kokora_"):
            continue
        path = os.path.join(_SHM_SWEEP_DIR, nm)
        try:
            if os.path.getmtime(path) > cutoff:
                continue  # 仍在合法生命周期内
            os.unlink(path)
            n += 1
        except OSError:
            continue  # 已被并发回收 / Windows 句柄未释放
    if n:
        log.warning("swept %d orphan audio SHM blocks (age > %.0fs)",
                    n, SHM_ORPHAN_MAX_AGE_S)
    return n


def _shm_stats() -> dict | None:
    """/dev/shm 使用情况（MB + 水位）；不可用（非 POSIX/Windows 开发机）返回 None。"""
    try:
        st = os.statvfs("/dev/shm")
        frsize = st.f_frsize
        total_mb = round(frsize * st.f_blocks / 2**20, 1)
        used_mb = round(frsize * (st.f_blocks - st.f_bavail) / 2**20, 1)
        if total_mb <= 0:
            return None
        return {"total_mb": total_mb, "used_mb": used_mb,
                "pct": round(used_mb / total_mb * 100, 1)}
    except (OSError, AttributeError):
        return None


def _shm_used_pct() -> float:
    """/dev/shm 水位（0-100）；读不了（非 POSIX/Windows 开发机）返回 -1。"""
    try:
        st = os.statvfs("/dev/shm")
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        if total <= 0:
            return -1.0
        return round((total - free) / total * 100, 1)
    except (OSError, AttributeError):
        return -1.0


async def _heartbeat_loop() -> None:
    while True:
        try:  # 心跳线程绝不能死——死了会被 supervisor 判 hang 强杀
            _write_own_health()
            _sweep_leaks()
            _sweep_orphan_shm()
            pct = _shm_used_pct()
            if pct >= 80:  # SHM 高水位：再涨就 SIGBUS（长文本 /tts 的历史根因）
                _throttled_warn("shm", "/dev/shm 水位 %.0f%% —— 音频块写满即 SIGBUS，"
                                "加大 --shm-size 或降低 /tts 并发", pct)
            now = time.time()
            try:
                hb_names = os.listdir(_health_dir)
            except OSError:
                hb_names = []   # 目录瞬时不可用（人工清理/挂载抖动）：本轮跳过，5s 后自愈
            for name in hb_names:  # 顺带清理已死 worker 的僵尸文件
                path = os.path.join(_health_dir, name)
                if not name.startswith(("worker-", ".tmp-")):
                    continue  # .tmp-：写一半崩溃的残片，同样回收
                try:
                    # 阈值对齐心跳 stale 线（30s）：supervisor 死后 /health
                    # 角色虚高窗口从 60s 减半；活 worker 心跳 5s 一刷不会误删
                    stale = now - os.path.getmtime(path) > _HEALTH_STALE_AFTER
                except OSError:
                    continue  # TOCTOU：listdir 与 getmtime 之间被写方 os.replace 走（生产事故 20261110：.tmp-<pid> 存在窗口毫秒级，撞上即 FileNotFoundError 刷 ERROR 栈）
                if stale:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
        except Exception:  # noqa: BLE001
            log.exception("heartbeat/sweep error (ignored)")
        await asyncio.sleep(_HEALTHBEAT_INTERVAL)


def _pump_once(result_q) -> bool:
    """单次取消息 + 递交；任何异常都不外抛（泵线程绝不能死）。

    get() 本身也可能抛：Redis 瞬断/超时、WRONGTYPE、坏 pickle（_recv 已
    兜坏 pickle，但连接类错误仍会到这里）。泵一死，所有后续请求 504，
    而 /health 可能还是绿的——这是最隐蔽的单点。返回 False 表示本轮
    取消息失败，调用方退避后重试。"""
    try:
        item = result_q.get()
    except Exception:  # noqa: BLE001
        log.exception("result pump dequeue failed (retry)")
        return False
    try:
        _main_loop.call_soon_threadsafe(_handle_result, item)
    except Exception:  # noqa: BLE001
        log.exception("result pump delivery failed (message dropped)")
    return True


def _result_pump(result_q) -> None:
    """独立线程：把结果队列的事件递交给服务事件循环。

    任何单条消息/递交异常都不允许杀死泵——泵一死，所有后续请求 504。
    loop 就绪判断在 get 之前：启动窗口期消息留在队列管道里（不消费不丢），
    loop 就绪后照常送达；若日后调整 _lifespan 顺序（泵先于 loop 赋值启动）
    也不会变成启动期静默丢结果。"""
    while True:
        if _main_loop is None or _main_loop.is_closed():
            time.sleep(0.05)
            continue
        if not _pump_once(result_q):
            time.sleep(0.05)


def _unlink_blocks(blocks: list) -> None:
    """SHM 块清理：attach → unlink（块由 tts 池创建，gateway 负责回收）。"""
    from multiprocessing import shared_memory

    for b in blocks or []:
        try:
            shm = shared_memory.SharedMemory(name=b["shm_name"])
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001  已被清理/名字不存在：幂等
            pass


def _retire_block(rid: str, block: dict) -> bool:
    """重复/孤儿 synth_part 块的延迟回收。

    /tts 的 ASR job 在 synth_part 之后派发，且只携带 shm_name——若 tts
    worker 在两者之间死亡，重试批的 synth_part 会被按重复段拦下，但该块
    仍是重试 ASR job 的唯一副本。此时立即 unlink 会让合法请求 502。
    仅当该请求的 TS Future 仍存在时挂起到 _retired_blocks，由端点 finally
    （TS 收齐/超时/断开）或 _sweep_leaks 统一回收；/v1 无 TS Future，
    沿用即时 unlink。"""
    if _pending.get(("ts", rid)) is not None:
        _retired_blocks.setdefault(rid, []).append((block, time.time()))
        return True
    _unlink_blocks([block])
    return False


# ── 结果消息处理（事件循环线程内执行） ──

def _agg_add(res: dict):
    """聚合一条 synth_part。返回 (st, done)；孤儿/未完成/重复返回 (None, False)。"""
    rid = res["req_id"]
    st = _agg_state.get(rid)
    if st is None:
        if rid not in _request_meta:
            # 请求已终止（超时/取消）：块即刻回收；但 /tts 的 TS 账本可能
            # 仍在等 ASR 读这个块（重试窗口）——由 _retire_block 判定。
            _retire_block(rid, res["block"])
            return None, False
        st = _agg_state[rid] = {"need": max(1, int(res["need"])), "idxs": set(),
                                "blocks": [], "texts": {}, "created": time.time(),
                                "t_g2p": 0.0, "t_q_g2p": 0.0,
                                # 请求级 TTS 排队跨度：min(首 chunk 入队) /
                                # max(末 chunk 出队)（seg 级 q_enq/q_claim）。
                                "t_q_tts_first": None, "t_q_tts_last": None,
                                # 请求级 TTS 墙钟跨度：min(首 chunk 认领) /
                                # max(末块前向完成 tts_done)。
                                "t_tts_claim_first": None,
                                "t_tts_done_last": None}
    # 幂等去重：worker 死亡后认领队列会把在途 chunk 精确重派（at-least-once），
    # 同一批 chunk 可能二次前向送达。重派批的 chunk 集与原批恒相同（同一批
    # 消息重投），全部 idx 已收过 → 重复，块即刻回收，绝不重复聚合（否则拼接
    # 出重复句）。部分重叠按重复处理——该不变量下不存在此形态，防御性从紧。
    all_segs = res["block"]["segs"]
    seg_idxs = {s["idx"] for s in all_segs}
    if seg_idxs and seg_idxs <= st["idxs"]:
        _retire_block(rid, res["block"])
        return None, False
    # 部分重叠（重派批重组：重派 chunk 与同请求新 chunk 并入一块）：
    # 只收新 segs——否则 _gather_audio 会把已收段读两遍（审计 F3，
    # 旧注释声称"部分重叠不存在"的不变量在单条重派下不成立）
    fresh = [s for s in all_segs if s["idx"] not in st["idxs"]]
    res["block"]["segs"] = fresh
    t = res.get("timings")
    if t and fresh:   # 只随实际入账的新 segs 计时（重复投递上面已拦，防御性兜底）
        # q_g2p/g2p 整单一次算、各 chunk 同值 → 取 max（重派批覆盖整块
        # 成员，max 口径对重复/部分重叠投递天然幂等）；tts_done 同理取 max。
        st["t_q_g2p"] = max(st["t_q_g2p"], float(t.get("q_g2p", 0.0)))
        st["t_g2p"] = max(st["t_g2p"], float(t.get("g2p", 0.0)))
        done = t.get("tts_done")
        if isinstance(done, (int, float)):
            st["t_tts_done_last"] = (done if st["t_tts_done_last"] is None
                                     else max(st["t_tts_done_last"], done))
    st["blocks"].append(res["block"])
    seg_enq, seg_claim = [], []
    for seg in fresh:
        st["idxs"].add(seg["idx"])
        st["texts"][seg["idx"]] = seg.get("text", "")
        if isinstance(seg.get("q_enq"), (int, float)):
            seg_enq.append(seg["q_enq"])
        if isinstance(seg.get("q_claim"), (int, float)):
            seg_claim.append(seg["q_claim"])
    if seg_claim:
        # tts 墙钟跨度起点：min(认领)。只由 fresh seg 推进（同 q_tts 跨度）。
        c_min = min(seg_claim)
        st["t_tts_claim_first"] = (c_min if st["t_tts_claim_first"] is None
                                   else min(st["t_tts_claim_first"], c_min))
    if seg_enq and seg_claim:
        # 请求级跨度：只由实际入账的 fresh seg 推进。重复/重试块里的已收
        # seg 不参与，避免重试的较晚 claim 把 q_tts 拉长。
        e, c_ = min(seg_enq), max(seg_claim)
        st["t_q_tts_first"] = (e if st["t_q_tts_first"] is None
                               else min(st["t_q_tts_first"], e))
        st["t_q_tts_last"] = (c_ if st["t_q_tts_last"] is None
                              else max(st["t_q_tts_last"], c_))
    if len(st["idxs"]) < st["need"]:
        return None, False
    _agg_state.pop(rid, None)
    return st, True


def _deliver(fut, st: dict, texts: list) -> bool:
    """把聚合完成的音频交付给等待方；请求已不在（超时/取消）则清块。"""
    if fut is None or fut.done():
        _unlink_blocks(st["blocks"])
        return False
    # 请求级墙钟跨度（同版本 worker 恒带戳；异常缺戳按 0 计，不炸交付）：
    # q_tts = 首个 chunk 入队 → 最后一个 chunk 被认领；
    # tts = 首个 chunk 认领 → 末块前向完成。
    q_tts = tts_ms = 0.0
    if (st["t_q_tts_first"] is not None
            and st["t_q_tts_last"] is not None):
        q_tts = max(0.0, (st["t_q_tts_last"]
                          - st["t_q_tts_first"]) * 1000)
    if (st["t_tts_claim_first"] is not None
            and st["t_tts_done_last"] is not None):
        tts_ms = max(0.0, (st["t_tts_done_last"]
                           - st["t_tts_claim_first"]) * 1000)
    fut.set_result({"blocks": st["blocks"], "texts": texts,
                    "fw": len(st["blocks"]),
                    "timings": {"q_g2p": st["t_q_g2p"], "g2p": st["t_g2p"],
                                "q_tts": q_tts, "tts": tts_ms,
                                "chunks": st["need"]}})
    return True


def _on_synth_part(res: dict) -> None:
    rid = res["req_id"]
    st, done = _agg_add(res)
    if not done:
        return
    texts = [st["texts"][i] for i in sorted(st["texts"])]
    fut = _pending.pop(("req", rid), None)
    _request_meta.pop(rid, None)  # 请求状态收口：后续重复消息按孤儿处理
    # 音频即到即交付：对齐（chunk_ts）与音频两账本独立收齐——chunk 级
    # 流水下 chunk_ts 可能先于音频聚合到达（乱序安全），/tts 的 fut_ts
    # 由 _on_chunk_ts 收齐后 resolve
    # 成功计数上移到端点（/v1 编码成功后、/tts 音频+TS 全链路成功后）——
    # 音频到达 ≠ 请求成功：/tts 后续还有对齐，编码也可能失败。
    _deliver(fut, st, texts)


def _on_synth_error(res: dict) -> None:
    rid = res["req_id"]
    st = _agg_state.pop(rid, None)
    if st is not None:
        _unlink_blocks(st["blocks"])
    fut = _pending.pop(("req", rid), None)
    fut_ts = _pending.get(("ts", rid))   # 不 pop：音频已交付时留给 TS 账本
    # 各 Future 独立判定：真失败（音频未交付）→ 两个 Future 都炸（各配
    # 独立异常实例，防 traceback 串线）；迟到 synth_error（批重派的第二
    # 次前向失败、音频已交付）→ 不碰 fut_ts，TS 在途由账本推进、请求
    # 应成功（审计 F2：f6266b2 曾在此把好的请求打成 502，方向修反）
    if fut is not None and not fut.done():
        fut.set_exception(_SynthError(res.get("kind", protocol.KIND_COMPUTE),
                                      res.get("error", "合成失败")))
        if fut_ts is not None and not fut_ts.done():
            _pending.pop(("ts", rid), None)
            fut_ts.set_exception(_SynthError(
                protocol.KIND_COMPUTE,
                f"对齐前合成失败: {str(res.get('error', ''))[:120]}"))



def _on_chunk_ts(res: dict) -> None:
    """TS 账本：乱序安全（ASR 与剩余 TTS 重叠执行，chunk_ts 可能先于
    音频聚合到达）；error 早退（整单对齐失败）；重复 idx 幂等（at-least-once
    重派的重复投递）。收齐 need 条 → 按 idx 排序拼接（偏移=前序采样数）
    → resolve fut_ts。"""
    rid = res["req_id"]
    st = _ts_state.get(rid)
    if st is None:
        if _pending.get(("ts", rid)) is None:
            return  # 请求已终结（超时/取消/失败）：孤儿丢弃
        st = _ts_state[rid] = {"need": max(1, int(res.get("need", 1))),
                               "items": {}, "created": time.time(),
                               # 请求级 ASR 排队跨度：min(首 job 入队) /
                               # max(末 job 出队)（job 级 _q_enq/_q_claim）。
                               "t_q_asr_first": None, "t_q_asr_last": None,
                               # 请求级 ASR 墙钟跨度：min(首 job 认领) /
                               # max(末桶完成 asr_done)。
                               # asr_buckets = 桶起始时刻去重集合 → asr_fw。
                               "t_asr_claim_first": None,
                               "t_asr_done_last": None,
                               "asr_buckets": set()}
    if res.get("error"):
        idx = int(res["idx"])
        if idx in st["items"]:
            return  # 该 chunk 已成功收账：重复 job 的偶发失败不翻案（审计 F10）
        _ts_state.pop(rid, None)
        fut_ts = _pending.pop(("ts", rid), None)
        if fut_ts is not None and not fut_ts.done():
            # 对齐服务故障 → 502（非通用 500；审计 F5）
            fut_ts.set_exception(_SynthError(
                protocol.KIND_COMPUTE,
                f"对齐服务失败: {str(res['error'])[:120]}"))
        return
    idx = int(res["idx"])
    if idx in st["items"]:
        return  # 重派重复投递：幂等
    t = res.get("timings")
    if t:   # 只维护请求级跨度戳（请求级耗时由跨度计算，不求和）
        e, c_ = t.get("q_asr_first"), t.get("q_asr_last")
        if isinstance(e, (int, float)):
            st["t_q_asr_first"] = (e if st["t_q_asr_first"] is None
                                   else min(st["t_q_asr_first"], e))
        if isinstance(c_, (int, float)):
            st["t_q_asr_last"] = (c_ if st["t_q_asr_last"] is None
                                  else max(st["t_q_asr_last"], c_))
            # asr 墙钟跨度起点 = min(认领)（q_asr_last 即认领时刻）
            st["t_asr_claim_first"] = (
                c_ if st["t_asr_claim_first"] is None
                else min(st["t_asr_claim_first"], c_))
        done = t.get("asr_done")
        if isinstance(done, (int, float)):
            st["t_asr_done_last"] = (done if st["t_asr_done_last"] is None
                                     else max(st["t_asr_done_last"], done))
        bkt = t.get("asr_bucket")
        if isinstance(bkt, (int, float)):
            st["asr_buckets"].add(bkt)
    st["items"][idx] = (int(res.get("samples", 0)),
                        res.get("timestamp") or [])
    if len(st["items"]) < st["need"]:
        return
    _ts_state.pop(rid, None)
    offset_ms = 0.0
    all_ts = []
    for i in sorted(st["items"]):
        samples, ts = st["items"][i]
        for s_, e_ in ts:
            all_ts.append([s_ + round(offset_ms), e_ + round(offset_ms)])
        # 偏移 = 前序采样 + 前序 gap（与 _concat 的插静音同轴——审计 F1）
        offset_ms += samples / _sample_rate * 1000 + _CHUNK_GAP_S * 1000
    fut_ts = _pending.pop(("ts", rid), None)
    if fut_ts is not None and not fut_ts.done():
        # 请求级墙钟跨度（同版本 worker 恒带戳；异常缺戳按 0 计）：
        # q_asr = 首 job 入队 → 末 job 出队；asr = 首 job 认领 → 末桶完成。
        q_asr = asr_ms = 0.0
        if (st["t_q_asr_first"] is not None
                and st["t_q_asr_last"] is not None):
            q_asr = max(0.0, (st["t_q_asr_last"]
                              - st["t_q_asr_first"]) * 1000)
        if (st["t_asr_claim_first"] is not None
                and st["t_asr_done_last"] is not None):
            asr_ms = max(0.0, (st["t_asr_done_last"]
                               - st["t_asr_claim_first"]) * 1000)
        fut_ts.set_result({"timestamp": all_ts,
                           "timings": {"asr": asr_ms,
                                       "q_asr": q_asr,
                                       "asr_fw": len(st["asr_buckets"])}})



def _handle_result(res: dict) -> None:
    try:
        t = res.get("type")
        if t == protocol.SYNTH_PART:
            _on_synth_part(res)
        elif t == protocol.SYNTH_ERROR:
            _on_synth_error(res)
        elif t == protocol.CHUNK_TS:
            _on_chunk_ts(res)
        else:
            log.warning("unknown result message type=%r (dropped)", t)
    except Exception:  # noqa: BLE001  单条消息异常不影响服务
        log.exception("result handling error (message dropped): %r", res)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动：登记事件循环 + 起心跳与结果泵；退出：摘心跳文件。
    （FastAPI lifespan 写法，替代已弃用的 on_event。）"""
    global _heartbeat_task, _main_loop
    _main_loop = asyncio.get_running_loop()
    _heartbeat_task = asyncio.get_running_loop().create_task(_heartbeat_loop())
    threading.Thread(target=_result_pump, args=(_result_q,), daemon=True).start()
    global _voices_cache
    _voices_cache = set(pipeline.list_voices())  # 预热：免首请求在事件循环扫盘（pipeline 已随模块导入）
    log.info("gateway ready (G2P in g2p pool, FunASR in asr pool, no model here)")
    yield
    if _heartbeat_task is not None:
        _heartbeat_task.cancel()
    try:  # 摘掉自己的心跳文件，聚合视图立即少一个 worker
        os.unlink(os.path.join(_health_dir, f"worker-{os.getpid()}.json"))
    except OSError:
        pass


app = FastAPI(
    title="Kokoro TTS (Ascend NPU)",
    version=_image_version,
    description="OpenAI-compatible TTS API for hexgrad/Kokoro-82M-v1.1-zh with dynamic batching",
    lifespan=_lifespan,
)

# 内网服务：允许任意源（tts-player 调试页面从 file:// 或独立静态服务器
# fetch 本服务时需要 CORS；无认证的纯内网 API，放开无额外暴露面）
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


_voices_cache: Optional[set] = None


def _check_voice(voice: str) -> None:
    global _voices_cache
    if _voices_cache is None:
        _voices_cache = set(pipeline.list_voices())
    if voice not in _voices_cache:
        raise HTTPException(400, f"未知音色: {voice}，可通过 GET /v1/voices 查询全部音色")


# /health 告警节流（必须定义在 _collect 之前：collector 注册/抓取会调用
# _collect，而 _collect 会触发缺员/积压告警）
_health_warn_ts = {}


def _throttled_warn(key: str, msg: str, *args) -> None:
    """/health 告警节流：同键 60s 最多一条。该端点会被 Docker HEALTHCHECK
    与外部监控秒级轮询——持续 degraded/积压不该每轮都刷一行日志。"""
    now = time.time()
    if now - _health_warn_ts.get(key, 0.0) < 60.0:
        return
    _health_warn_ts[key] = now
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
    _latency_bins()[bisect.bisect_left(_LATENCY_BUCKETS, seconds)] += 1
    _entry_stats["latency_ms_sum"] += seconds * 1000
    _recent_latencies.append((seconds, chars))


def _collect() -> dict:
    """全量聚合数据源（/metrics 与精简 /health 共用，纯同步）。

    聚合内容：status/ready 判定（跨角色能力就绪）、队列 4 阶段、workers 全量
    （含 raw 快照）、e2e（跨网关合并）、SHM 水位、supervisor 崩溃计数。

    status=ok 需满足：全部角色就绪（ready）+ 无预热失败 + 无角色缺员
    （缺员判定来自 supervisor 心跳的 expected 声明，只影响 status）。
    """
    me = _worker_snapshot()
    workers, now = [], time.time()
    try:
        for name in sorted(os.listdir(_health_dir)):
            if not name.startswith("worker-"):
                continue
            path = os.path.join(_health_dir, name)
            try:
                if now - os.path.getmtime(path) > _HEALTH_STALE_AFTER:
                    continue  # 超时未心跳：worker 已死（如被 SIGKILL），剔除
                with open(path, encoding="utf-8") as f:
                    workers.append(json.load(f))
            except (OSError, ValueError):
                continue
    except OSError:
        pass
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
    caps = {"g2p_queue": _MAX_PENDING_REQUESTS,
            "tts_queue": proc_common.env_int("MAX_INFLIGHT_SENTENCES", 512,
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
        cap = caps.get(name, _WORKER_MAILBOX_SIZE)
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
    # 用全实例求和，读本进程 _entry_stats 会让 rate/quantile 剧烈跳动）。
    # 心跳带 latency_bounds 校验；同容器内所有网关同版本同配置，边界必然
    # 一致——不一致（异常心跳/配置漂移）时跳过该实例的桶计数，绝不把
    # 错误粒度的 bins 直接相加污染直方图。
    nbins = len(_LATENCY_BUCKETS) + 1
    expect_bounds = [float(b) for b in _LATENCY_BUCKETS]
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
        "version": _image_version,
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


# ── 探针端点：/health（三探针共用，业界习惯）+ /metrics（指标）──
# 未就绪（能力未齐）→ 503：readiness 摘流量；预热期 liveness 由 startupProbe
# 宽限保护。degraded 是可服务态 → 200，不触发重启。

@app.get("/health")
async def health():
    """K8s 三探针共用（readiness/liveness/startup）：
    - ready=false（预热期/某角色零就绪）→ 503（必须配 startupProbe 宽限，
      否则预热期会被 liveness 反复重启）
    - ready=true → 200；degraded（预热失败/部分缺员但可服务）也 200。
    body 为精简结论卡，完整数值见 /metrics。聚合扫盘（心跳目录 listdir +
    逐文件 json.load）下放线程池——与 _asr_ready 自立的"事件循环不逐请求
    扫盘"规范一致。"""
    payload = await run_in_threadpool(health_payload)
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


if METRICS_ENABLED:
    from prometheus_client import CollectorRegistry
    from prometheus_client.core import (CounterMetricFamily,
                                        GaugeMetricFamily,
                                        HistogramMetricFamily)

    class _KokoroCollector:
        """Prometheus 自定义收集器：抓取时现算全量聚合（无陈旧 gauge）。"""

        def collect(self):
            data = _collect()

            # e2e 计数器（聚合绝对值；自定义 Collector 输出绝对值 Counter 合法）
            c = CounterMetricFamily("kokoro_e2e_requests", "受理请求总数")
            c.add_metric([], data["e2e"]["requests"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_successes", "成功总数")
            c.add_metric([], data["e2e"]["successes"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_timeouts", "超时累计")
            c.add_metric([], data["e2e"]["timed_out"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_rejected", "背压拒绝累计（队列满）")
            c.add_metric([], data["e2e"]["rejected"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_fast_reject",
                                    "准入快速拒绝累计（排队积压预估超等待预算）")
            c.add_metric([], data["e2e"]["fast_reject"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_cancelled", "客户端断开取消累计")
            c.add_metric([], data["e2e"]["cancelled"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_failed", "终态失败累计（超时+合成失败+服务端 500）")
            c.add_metric([], data["e2e"]["failed"])
            yield c
            c = CounterMetricFamily("kokoro_e2e_server_5xx",
                                    "服务端 5xx 累计（Redis 不可达等快速失败）")
            c.add_metric([], data["e2e"]["server_5xx"])
            yield c

            # 队列 4 阶段（g2p → tts → asr → gateway）
            depth = GaugeMetricFamily("kokoro_queue_depth", "队列当前堆积", labels=["stage"])
            in_t = CounterMetricFamily("kokoro_queue_in_total", "队列累计入队", labels=["stage"])
            out_t = CounterMetricFamily("kokoro_queue_out_total", "队列累计出队", labels=["stage"])
            wait = GaugeMetricFamily("kokoro_queue_avg_wait_ms", "近期平均排队毫秒", labels=["stage"])
            # 坏 member 丢弃（滚动升级混跑旧格式/载荷损坏）——非零即需排查
            bad_t = CounterMetricFamily("kokoro_queue_bad_total",
                                        "坏 member 丢弃累计", labels=["stage"])
            for stage, q in data["queues"].items():
                depth.add_metric([stage], q["depth"])
                in_t.add_metric([stage], q["in"])
                out_t.add_metric([stage], q["out"])
                wait.add_metric([stage], q["avg_wait_ms"])
                bad_t.add_metric([stage], q["bad"])
            yield depth
            yield in_t
            yield out_t
            yield wait
            yield bad_t

            # 每 worker（含 raw 快照里的 npu_mem/stuck_s/cache 统计）
            done = CounterMetricFamily("kokoro_worker_done_total", "完成工作单元数",
                                       labels=["worker", "role"])
            err = CounterMetricFamily("kokoro_worker_errors_total", "错误累计",
                                      labels=["worker", "role"])
            crashes = CounterMetricFamily("kokoro_worker_crashes_total", "累计崩溃次数",
                                          labels=["worker"])
            ready_g = GaugeMetricFamily("kokoro_worker_ready", "就绪状态 0/1",
                                        labels=["worker", "role"])
            rtx = GaugeMetricFamily("kokoro_worker_realtime_x", "实时倍速（tts）",
                                    labels=["worker"])
            wavg = GaugeMetricFamily("kokoro_worker_avg_ms", "单位工作量平均毫秒",
                                     labels=["worker", "role"])
            stuck = GaugeMetricFamily("kokoro_worker_stuck_seconds", "busy 卡住秒数",
                                      labels=["worker", "role"])
            npu_r = GaugeMetricFamily("kokoro_worker_npu_mem_reserved_mb",
                                      "NPU 显存预留 MB（torch 缓存池）",
                                      labels=["worker"])
            npu_p = GaugeMetricFamily("kokoro_worker_npu_budget_pct",
                                      "NPU 预算占比（reserved/budget）",
                                      labels=["worker"])
            # 每 worker 内存明细：allocated/reserved 来自 torch，
            # context 是 CANN 上下文估算（tts 450 / asr 153），
            # estimated = reserved+context，planned = 0.98×budget+context。
            npu_alloc = GaugeMetricFamily("kokoro_worker_npu_mem_allocated_mb",
                                          "NPU 当前分配 MB", labels=["worker", "role"])
            npu_est = GaugeMetricFamily("kokoro_worker_npu_mem_estimated_mb",
                                        "NPU 估算物理占用 MB（reserved+CANN ctx）",
                                        labels=["worker", "role"])
            npu_plan = GaugeMetricFamily("kokoro_worker_npu_mem_planned_mb",
                                         "NPU 容量核算 MB（0.98×budget+CANN ctx）",
                                         labels=["worker", "role"])
            npu_budget = GaugeMetricFamily("kokoro_worker_npu_mem_budget_mb",
                                           "NPU worker 预算 MB", labels=["worker", "role"])
            npu_ctx = GaugeMetricFamily("kokoro_worker_npu_mem_context_mb",
                                        "CANN 上下文估算 MB", labels=["worker", "role"])
            npu_card = GaugeMetricFamily("kokoro_worker_npu_mem_card_pct",
                                         "估算占用/整卡 %", labels=["worker", "role"])
            npu_proc = GaugeMetricFamily("kokoro_worker_npu_mem_process_mb",
                                         "npu-smi 实测进程内存 MB",
                                         labels=["worker", "role"])
            npu_delta = GaugeMetricFamily("kokoro_worker_npu_mem_delta_mb",
                                          "实测 - planned MB（正=公式低估）",
                                          labels=["worker", "role"])
            npu_ctx_actual = GaugeMetricFamily(
                "kokoro_worker_npu_mem_context_actual_mb",
                "实测 - torch reserved MB（实际非 torch 开销）",
                labels=["worker", "role"])
            npu_limit = GaugeMetricFamily(
                "kokoro_worker_npu_mem_limit_mb",
                "fraction 允许的进程上限 MB（1.05×budget+context）",
                labels=["worker", "role"])
            gh = CounterMetricFamily("kokoro_g2p_cache_hits_total", "g2p 缓存命中",
                                     labels=["worker"])
            gm = CounterMetricFamily("kokoro_g2p_cache_misses_total", "g2p 缓存未命中",
                                     labels=["worker"])
            # 灰度期观察 fp16 流量占比（按批计；0 不输出）
            fp16b = CounterMetricFamily("kokoro_worker_fp16_batches_total",
                                        "fp16 前向批数", labels=["worker"])
            # asr 批推可观测：生效档与安全补丁状态（0=未装→已自动降单条）
            batch_eff = GaugeMetricFamily("kokoro_worker_batch_effective",
                                          "ASR 批推生效批宽", labels=["worker", "role"])
            batch_patch = GaugeMetricFamily("kokoro_worker_batch_patch_ok",
                                            "ASR 批推安全补丁状态（1=已安装）",
                                            labels=["worker", "role"])

            # per-worker：直接消费 _collect() 的指标行（与 /health 同源同值，
            # 杜绝二次推导漂移）；npu/stuck/cache 补充自原始快照
            raw_by_tag = {w.get("wtag"): w for w in data["workers_raw"]}
            for row in data["workers"]:
                tag, role = row.get("wtag") or f"{row.get('role')}-?", row.get("role") or "?"
                if row.get("done"):
                    done.add_metric([tag, role], row["done"])
                if row.get("err"):
                    err.add_metric([tag, role], row["err"])
                if row.get("avg_ms") is not None:
                    wavg.add_metric([tag, role], row["avg_ms"])
                ready_g.add_metric([tag, role], 1 if row.get("ready") else 0)
                if row.get("realtime_x") is not None:
                    rtx.add_metric([tag], row["realtime_x"])
                if row.get("batch_effective") is not None:
                    batch_eff.add_metric([tag, role], row["batch_effective"])
                if row.get("batch_patch_ok") is not None:
                    batch_patch.add_metric(
                        [tag, role], 1 if row["batch_patch_ok"] else 0)
                w_raw = raw_by_tag.get(tag) or {}
                st = w_raw.get("stats") or {}
                for key, fam in (("cache_hits", gh), ("cache_misses", gm)):
                    if st.get(key):
                        fam.add_metric([tag], st[key])
                if role == "tts" and st.get("fp16_jobs"):
                    fp16b.add_metric([tag], st["fp16_jobs"])
                # tts/asr 同一套内存明细（旧 worker 只有 reserved/budget_pct
                # 时新指标自动缺省，不产生 0 值误导）
                nm = w_raw.get("npu_mem") or {}
                if nm.get("reserved_mb") is not None:
                    npu_r.add_metric([tag], nm["reserved_mb"])
                if nm.get("budget_pct") is not None:
                    npu_p.add_metric([tag], nm["budget_pct"])
                for key, fam in (("allocated_mb", npu_alloc),
                                 ("estimated_mb", npu_est),
                                 ("planned_mb", npu_plan),
                                 ("budget_mb", npu_budget),
                                 ("context_mb", npu_ctx),
                                 ("card_pct", npu_card),
                                 ("process_mb", npu_proc),
                                 ("delta_mb", npu_delta),
                                 ("context_actual_mb", npu_ctx_actual),
                                 ("limit_mb", npu_limit)):
                    if nm.get(key) is not None:
                        fam.add_metric([tag, role], nm[key])
                if w_raw.get("stuck_s"):
                    stuck.add_metric([tag], w_raw["stuck_s"])
            # 崩溃计数来自 supervisor 快照映射（不随 worker 心跳上报）
            for wtag, n_crash in (data.get("crashes") or {}).items():
                if n_crash:
                    crashes.add_metric([wtag], n_crash)
            yield done
            yield err
            yield crashes
            yield ready_g
            yield rtx
            yield wavg
            yield stuck
            yield npu_r
            yield npu_p
            yield npu_alloc
            yield npu_est
            yield npu_plan
            yield npu_budget
            yield npu_ctx
            yield npu_card
            yield npu_proc
            yield npu_delta
            yield npu_ctx_actual
            yield npu_limit
            yield gh
            yield gm
            yield fp16b
            yield batch_eff
            yield batch_patch

            shm = data.get("shm")
            if shm:
                g = GaugeMetricFamily("kokoro_shm_used_mb", "/dev/shm 已用 MB")
                g.add_metric([], shm["used_mb"])
                yield g
                g = GaugeMetricFamily("kokoro_shm_total_mb", "/dev/shm 总量 MB")
                g.add_metric([], shm["total_mb"])
                yield g
                g = GaugeMetricFamily("kokoro_shm_pct", "/dev/shm 水位百分比")
                g.add_metric([], shm["pct"])
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
            for edge, n in zip(_LATENCY_BUCKETS, lat["bins"]):
                cum += n
                buckets.append([_bucket_label(edge), cum])
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


@app.get("/metrics")
async def metrics():
    if not METRICS_ENABLED:
        raise HTTPException(404, "metrics disabled (METRICS_ENABLED=0)")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    body = await run_in_threadpool(
        lambda: generate_latest(_prom_registry()))  # collect+序列化下放线程池
    return Response(body, media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "kokoro-v1.1-zh", "object": "model", "owned_by": "hexgrad"}],
    }


@app.get("/v1/voices")
def voices():
    global _voices_cache
    if _voices_cache is None:
        _voices_cache = set(pipeline.list_voices())
    return {"voices": sorted(_voices_cache)}


class _ClientGone(Exception):
    """客户端已断开（等待期间被断连 watcher 取消）。"""


async def _synth(text: str, voice: str, speed: float, needs_align: bool,
                 req_id: str | None = None, release_shm: bool = True,
                 request: Request | None = None,
                 g2p_overrides: dict | None = None,
                 dtype: str = protocol.DTYPE_DEFAULT,
                 deadline: float | None = None):
    """合成入口：整单原子入 g2p_queue → g2p 池切句/G2P → tts 认领组批 →
    前向 → synth_part 逐段直投本入口 → 聚合收齐 need 句后返回。

    deadline：单一请求级截止线（perf_counter 绝对时刻）。本函数的音频
    等待只用剩余预算；None 按入口+WAIT_TIMEOUT_S 自算。/tts 由 handler
    传入入口时刻的 deadline，把余额留给对齐段——总时长硬上限 =
    WAIT_TIMEOUT_S（音频/对齐不再各 120s 串联）。

    返回 (audio, texts, fw, req_id, blocks_desc, timings, agg_s)；
    timings 为请求级阶段耗时（毫秒，{"q_g2p","g2p","q_tts","tts","chunks"}，
    由网关按 worker 捎带的跨度戳计算）；agg_s 为网关侧 SHM 读出+拼接耗时（秒）；
    audio 恒为拼接后的新数组（不回传 SHM 视图，杜绝悬垂）。
    release_shm=True（/v1）：聚合完成即收尾统一 unlink SHM；
    release_shm=False（/tts）：对齐阶段 asr 还要按名 attach 读块——本函数
    不清理，由 /tts 端点在全部 chunk_ts 收齐后回收；本函数自身的异常
    路径仍就地清理
    （请求终止时无人接手）。
    """
    # 语速钳制：帧数 = 预测时长 ÷ 语速，speed 无下限会把帧数（进而显存/时延
    # 上限）放大到不可控。下限默认 0.5，可用 TTS_SPEED_FLOOR 调整
    speed = max(_speed_floor, min(speed, 4.0))
    loop = asyncio.get_running_loop()
    req_id = req_id or uuid.uuid4().hex[:12]
    deadline = deadline if deadline is not None else (
        time.perf_counter() + _entry_wait_s)
    fut = loop.create_future()
    _pending[("req", req_id)] = fut
    _request_meta[req_id] = {"needs_align": bool(needs_align), "created": time.time()}

    # 断连取消：客户端断开 → 立即摘除本请求状态（后续 synth_part/chunk_ts
    # 走既有孤儿路径被丢弃）→ 写 CANCEL 标记（tts 认领后前向前过滤），
    # 已派发的批无法撤回但结果会被网关丢弃，不再浪费编码/对齐。
    poll_task = None
    if request is not None:
        async def _watch_disconnect() -> None:
            try:
                while not await request.is_disconnected():
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001  探测异常 ≠ 已断连：直接退出
                # watcher（不取消请求——落穿会把活连接误判 499），该请求由
                # 正常超时/完成路径收尾；watcher 死了只损失"提前取消"能力
                log.exception("disconnect watcher probe error (watcher exits, request continues)")
                return
            if fut.done():
                return
            _pending.pop(("req", req_id), None)
            _pending.pop(("ts", req_id), None)
            _request_meta.pop(req_id, None)
            abandoned = _agg_state.pop(req_id, None)   # 已聚合块即时回收
            if abandoned:
                _unlink_blocks(abandoned["blocks"])
            try:
                proc_common.cancel_add(req_id)   # g2p 推送前 / tts 前向前检查跳过
            except Exception as exc:  # noqa: BLE001  redis 抖动：标记尽力而为，
                # 不能让 watcher 任务死掉——否则 fut 永不 resolve，请求白等到截止线
                log.warning("req=%s cancel_add failed (disconnect path, ignored): %s",
                            req_id, exc)
            _entry_stats["cancelled"] = _entry_stats.get("cancelled", 0) + 1
            log.info("req=%s client disconnected, request cancelled", req_id)
            if not fut.done():
                fut.set_exception(_ClientGone("客户端已断开"))

        poll_task = loop.create_task(_watch_disconnect())

    from multiprocessing import shared_memory

    try:
        # 准入快速拒绝（背压闸门 1 前置）：请求路径各段 EMA 排队等待求和
        # （队列空的段计 0——EMA 仅认领时更新，空队视为陈旧值），超过
        # 预算×冗余系数 → 排到它必然超截止线，秒拒不受理（429+Retry-After，
        # 好过受理后骑满 120s 才 504）。asr 段仅 /tts（对齐）参与——/v1
        # 不付对齐成本。redis 瞬断 fail-open：放行，截止线+超时取消兜底。
        pressures = []
        try:
            g2p_st = _g2p_queue.stats()
            pressures.append(("g2p", int(g2p_st.get("depth") or 0),
                              float(g2p_st.get("avg_wait_ms") or 0)))
            pressures.append(("tts",) + proc_common.claim_queue_pressure(
                proc_common.QUEUE_INDEX["tts"], "tts"))
            if needs_align:
                pressures.append(("asr",) + proc_common.claim_queue_pressure(
                    proc_common.QUEUE_INDEX["asr"], "asr"))
        except Exception as exc:  # noqa: BLE001
            log.warning("fast reject probe failed (fail-open): %s", exc)
        est_ms = sum(w for _s, d, w in pressures if d > 0)
        threshold_ms = _entry_wait_s * 1000 * FAST_REJECT_MARGIN
        if est_ms > threshold_ms:
            _entry_stats["fast_reject"] += 1
            log.info(
                "fast reject req=%s est=%.2fs > threshold=%.2fs "
                "(budget=%.1fs×%.2f) %s",
                req_id, est_ms / 1000, threshold_ms / 1000,
                _entry_wait_s, FAST_REJECT_MARGIN,
                " ".join(f"{s}(d={d},w={w / 1000:.2f}s)"
                         for s, d, w in pressures))
            raise HTTPException(
                429, "服务繁忙（排队积压超过等待预算），请稍后重试",
                headers={"Retry-After": "1",
                         "X-Reject-Reason": "wait_budget"})
        # 背压闸门 1.5：/dev/shm 水位。音频块在聚合/对齐期间驻留 tmpfs，
        # 写满即 SIGBUS（tts worker rc=-7 的历史事故）。90% 时拒绝新请求，
        # 给在途请求留出回收余量；非 POSIX（Windows 开发机）返回 -1 不拦。
        shm_pct = _shm_used_pct()
        if shm_pct >= 90.0:
            _entry_stats["rejections"] += 1
            _throttled_warn("shm_reject",
                            "/dev/shm 水位 %.0f%% >= 90%%，拒绝新请求（SIGBUS 防护）",
                            shm_pct)
            raise HTTPException(
                429, "服务繁忙（共享内存水位高），请稍后重试",
                headers={"Retry-After": "1",
                         "X-Reject-Reason": "shm_pressure"})
        # 整单入队（文本 KB 级）；切句/G2P 由 g2p 池执行。
        # 背压闸门 1：g2p_queue 满 → 立即 429（不受理不排队，连文本都不处理）
        try:
            _g2p_queue.put_nowait(protocol.new_request(
                req_id, _worker_origin, text, voice, speed, needs_align,
                g2p_overrides=g2p_overrides, dtype=dtype))
        except queue.Full:
            _entry_stats["rejections"] += 1
            raise HTTPException(429, "服务繁忙，请稍后重试",
                                headers={"Retry-After": "1",
                                         "X-Reject-Reason": "queue_full"})
        _entry_stats["requests"] += 1
        # 音频段只用剩余预算（单一截止线），余额留给 /tts 的对齐段
        r = await asyncio.wait_for(
            fut, timeout=max(deadline - time.perf_counter(), 0.0))
    except asyncio.TimeoutError:
        _entry_stats["timeouts"] += 1
        # 超时即取消：与断连同路下发 CANCEL，在途 chunk 认领后前向前被
        # 丢弃——死请求不再浪费前向，过载恢复更快。redis 抖动时吞掉：
        # 取消是尽力而为，绝不能把 504 变成 500（也让 _record_failure 留着）。
        try:
            proc_common.cancel_add(req_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("req=%s cancel_add failed (timeout path, ignored): %s",
                        req_id, exc)
        _record_failure(504, "推理或对齐超时")
        raise HTTPException(504, "推理超时，请稍后重试")
    except _ClientGone:
        raise HTTPException(499, "客户端已断开")
    except _SynthError as exc:
        # 服务端留痕：否则坏输入只会 400 给客户端，服务端日志无线索
        log.info("req=%s synth failed kind=%s: %s", req_id, exc.kind, exc)
        _record_failure(400 if exc.kind == protocol.KIND_INVALID else 502,
                        str(exc))
        raise HTTPException(400 if exc.kind == protocol.KIND_INVALID else 502,
                            str(exc))
    finally:
        if poll_task is not None:
            poll_task.cancel()
        _pending.pop(("req", req_id), None)   # 任何路径都摘除：防 _pending 泄漏
        _request_meta.pop(req_id, None)
        # 中止路径（超时/取消/断连）已聚合的部分块立即回收：否则要等
        # _sweep_leaks 的 2×WAIT(240s) 才清，正是 SHM 写满事故的加压条件
        abandoned = _agg_state.pop(req_id, None)
        if abandoned:
            _unlink_blocks(abandoned["blocks"])

    fw, texts, blocks_desc = r["fw"], r["texts"], r["blocks"]
    t_agg = time.perf_counter()   # agg 计时起点 = 音频账本收齐（fut 交付）

    def _gather_audio() -> np.ndarray:
        """SHM 读出 + 拼接（线程池内执行）：长文本的 SHM 合计可达几十 MB，
        frombuffer 逐句拷贝 + concatenate 若在事件循环上跑会阻塞同网关全部
        在途请求——与编码下放线程池同一原则。"""
        shms_ = []
        try:
            # 逐块 append（不能用列表推导）：第 k 块 attach 失败时推导式
            # 中止、赋值不发生，前 k-1 个已打开句柄会整体逃出 finally 的
            # close——fd/mmap 泄漏（块名被并发回收/Windows keepalive 淘汰
            # 后 attach 落空是真实可触发场景）
            for b in blocks_desc:
                shms_.append(shared_memory.SharedMemory(name=b["shm_name"]))
            pieces = []
            for b, shm in zip(blocks_desc, shms_):
                for s in b["segs"]:
                    # SHM 线格式即交付格式（PCM16）：全程 int16 直通，不再做
                    # float32 往返（原实现每请求 3-4 次全量格式转换）。
                    # copy 成自有数组——视图绝不跨语句存活（BufferError 防护）
                    pieces.append((s["idx"],
                                   np.frombuffer(
                                       shm.buf[s["offset"]:
                                               s["offset"] + s["samples"] * 2],
                                       dtype="<i2").copy()))
            pieces.sort(key=lambda t: t[0])  # 按句序（跨块）
            return _concat([p for _i, p in pieces])
        except Exception:
            # 全量回收（含尚未 attach 的块）：attach 中途失败时，未打开的
            # 块名仍只在这里可见——只 unlink 已 attach 句柄会留下孤儿，
            # 要等 420s 清道夫。_unlink_blocks 幂等，finally 再 unlink 无害。
            _unlink_blocks(blocks_desc)
            raise
        finally:
            for shm in shms_:
                if release_shm:
                    try:
                        shm.unlink()   # 复用已打开句柄（免每块二次 attach）
                    except Exception:  # noqa: BLE001  已清理/名不存在：幂等
                        pass
                shm.close()

    audio = await run_in_threadpool(_gather_audio)
    agg_s = time.perf_counter() - t_agg
    return (audio, texts, fw, req_id, blocks_desc,
            r.get("timings") or {}, agg_s)


def _concat(parts: List[np.ndarray]) -> np.ndarray:
    """拼接句音频（PCM16），句间补 60ms 静音；恒定返回自有数组。

    单片快捷返回（单句请求常见）——入参已是 copy 的自有数组，无悬垂风险。"""
    if not parts:
        return np.zeros(0, dtype="<i2")
    if len(parts) == 1:
        return parts[0]
    gap = np.zeros(int(_sample_rate * _CHUNK_GAP_S), dtype="<i2")
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append(gap)
        out.append(p)
    return np.concatenate(out)


def _record_failure(status: int, reason: str) -> None:
    """记录一次请求级失败（时间 + 状态码 + 截断原因），/health 的
    recent_failures 展示最近 3 条（跨网关合并）。"""
    _entry_stats["failed"] += 1  # 终态失败计数（supervisor 卡死检测用）
    if status >= 500:            # 4xx 是客户端问题，不能把 /health 打成 degraded
        _entry_stats["server_5xx"] = _entry_stats.get("server_5xx", 0) + 1
        _entry_stats["last_5xx_ts"] = time.time()
    _recent_failures.append({"time": time.strftime("%H:%M:%S"),
                             "status": status, "reason": str(reason)[:80]})


# 无主音频块超龄回收线：必须大于块的合法最长持有期。块在 tts 前向时创建，
# /tts 最长合法生命周期现为单一截止线 ≤WAIT（音频+时间戳共享预算，不再
# 两段串行 2×WAIT）；公式 2×WAIT+60 保留为更保守的历史余量，不会误扫活块。
# 显式配置只能调大不能调小（低于公式值被抬高）。
SHM_ORPHAN_MAX_AGE_S = max(
    proc_common.env_float("SHM_ORPHAN_MAX_AGE_S", 300.0, minimum=30.0),
    2 * _entry_wait_s + 60.0)   # WAIT_TIMEOUT_S 只在模块头解析一次
_SHM_SWEEP_DIR = "/dev/shm"    # 音频 SHM 所在 tmpfs（测试可替换）


def _log_request_text(req_id: str, text: str) -> None:
    """请求文本日志策略（应用方要求）：INFO 打完整 TEXT（单行、换行替换为空格）。"""
    flat = text.replace(chr(10), " ").replace(chr(13), "")
    log.info("req=%s chars=%d text=%s", req_id, len(text), flat)


_MAX_BODY_BYTES = proc_common.env_int("MAX_BODY_BYTES", 4 * 2**20, minimum=1024)


class _BodySizeLimitMiddleware:
    """请求体上限 + Content-Type 归一（纯 ASGI）：Pydantic/JSON 解析前先看
    content-length——超大或恶意请求体先占内存再在事件循环上解析的路径
    直接 413 拒绝（内网无害，对外暴露时的基本防线）。分块编码无
    content-length 的由框架自然限流。

    Content-Type 归一：本服务端点只收 JSON 体，而 curl --data-raw 不带
    Content-Type（curl --data-raw 默认 application/x-www-form-urlencoded），
    FastAPI 对 JSON 模型体遇 urlencoded 会按表单解析 → 422。缺失或
    urlencoded 时把头部改写为 application/json（纯头部操作，零缓冲、
    不碰 receive 通道）。仅处理 POST；multipart 等其他类型不干预。

    必须是纯 ASGI：@app.middleware("http") 会套 BaseHTTPMiddleware，
    它包裹 receive 通道导致 request.is_disconnected() 恒为 False——
    断连取消链路（499/CANCEL/即时回收）整体失效（E2E 实测发现，
    生产一直在浪费断开客户端的推理算力）。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers")
            if scope.get("method") == "POST" and headers is not None:
                ct_idx = next((i for i, (k, _v) in enumerate(headers)
                               if k == b"content-type"), None)
                ct = (headers[ct_idx][1].decode("latin-1").lower()
                      if ct_idx is not None else "")
                if not ct or ct == "application/x-www-form-urlencoded":
                    new_ct = b"application/json"
                    if ct_idx is not None:
                        headers[ct_idx] = (b"content-type", new_ct)
                    else:
                        headers.append((b"content-type", new_ct))
            for k, v in scope.get("headers") or []:
                if k != b"content-length":
                    continue
                if v.isdigit() and int(v) > _MAX_BODY_BYTES:
                    await JSONResponse(status_code=413, content={
                        "detail": f"请求体过大（上限 {_MAX_BODY_BYTES // 2**20}MB）"
                    })(scope, receive, send)
                    return
                break
        await self.app(scope, receive, send)


app.add_middleware(_BodySizeLimitMiddleware)


class SpeechRequest(BaseModel):
    model: str = "kokoro-v1.1-zh"
    input: str = Field(..., min_length=1, description="待合成文本")
    voice: str | None = None
    response_format: Literal["wav", "pcm", "mp3", "flac", "opus", "aac"] = "wav"
    speed: float = Field(1.0, gt=0, le=4.0)
    g2p_overrides: dict[str, str | list[str]] | None = Field(
        None, description="请求级多音字覆盖 {词组: TONE3 拼音串/列表}，仅本请求生效")
    # Any 而非 str：非字符串输入（16/[]/{...}）也交给 protocol.normalize_dtype
    # 统一报 400 并列出可选值——Pydantic 的 str 类型会先拦成 422（只说"应为字符串"）
    dtype: Any = Field(
        None, description=f"计算精度，可选 {'/'.join(protocol.DTYPE_CHOICES)}"
                          f"（缺省 {protocol.DTYPE_DEFAULT}）")


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request = None):
    t_e2e = time.perf_counter()   # e2e 起点：进处理函数（校验/排队/合成/编码全程）
    text = req.input.strip()
    if not text:
        raise HTTPException(400, "input 不能为空")
    if len(text) > _max_chars:
        raise HTTPException(400, f"input 超长({len(text)} > {_max_chars} 字符)，请分段请求")
    try:
        overrides = protocol.normalize_overrides(req.g2p_overrides)
        dtype = protocol.normalize_dtype(req.dtype)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    voice = (req.voice or _default_voice).strip()

    t0 = time.perf_counter()
    try:
        _check_voice(voice)
        (audio, _texts, fw, rid, _blocks,
         timings, agg_s) = await _synth(
            text, voice, req.speed, needs_align=False,  # OpenAI 兼容接口不付对齐成本
            request=request, g2p_overrides=overrides, dtype=dtype,
            deadline=t_e2e + _entry_wait_s)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed voice=%s", voice)
        _record_failure(500, f"合成失败: {exc}")
        raise HTTPException(500, f"合成失败: {exc}")
    elapsed = time.perf_counter() - t0
    duration = len(audio) / _sample_rate
    _record_latency(elapsed, len(text))
    _log_request_text(rid, text)

    fmt = req.response_format
    t_enc = time.perf_counter()
    try:
        if fmt == "wav":
            mime = "audio/wav"
            body = await run_in_threadpool(_encode_wav, audio)
        elif fmt == "flac":
            mime = "audio/flac"
            body = await run_in_threadpool(_encode_flac, audio)
        elif fmt == "pcm":
            mime = "audio/pcm"
            # 一样下放线程池：满额请求 PCM 20-30MB，tobytes 一次拷贝 10-30ms
            body = await run_in_threadpool(_encode_pcm, audio)
        else:
            mime = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac"}[fmt]
            body = await run_in_threadpool(_encode_ffmpeg, audio, fmt)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    enc_s = time.perf_counter() - t_enc
    _entry_stats["results"] += 1   # 成功口径 = 完整响应体已就绪（非仅音频到达）
    # 单行汇总：只带网关独有字段（同 /tts done 口径）——批/句级明细由
    # tts worker 的 bid=/chunk= 行承担
    log.info("ep=/v1 req=%s voice=%s speed=%.2f audio_s=%.1f e2e_s=%.2f "
             "agg_ms=%.0f enc_ms=%.0f",
             rid, voice, req.speed, duration,
             time.perf_counter() - t_e2e, agg_s * 1000, enc_s * 1000)

    return Response(
        content=body,
        media_type=mime,
        headers={
            "X-Request-ID": rid,  # 与各 worker 日志的 req= 对齐，客户端报障可凭此定位
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Synth-Time": f"{elapsed:.3f}",
            "X-Batch-Size": str(fw),  # 该请求的前向批次数（>1 = 多批聚合而成）
        },
    )


class TTSRequest(BaseModel):
    """业务兼容接口（对齐业务报文契约：成功返回 audio_base64/code/duration/
    message/text/timestamp 6 字段，错误返回 {"error": msg}）。"""
    text: str | None = None
    speed: float = Field(1.0, gt=0, le=4.0)
    voice: str | None = None
    g2p_overrides: dict[str, str | list[str]] | None = Field(
        None, description="请求级多音字覆盖 {词组: TONE3 拼音串/列表}，仅本请求生效")
    # Any 而非 str：非字符串输入（16/[]/{...}）也交给 protocol.normalize_dtype
    # 统一报 400 并列出可选值——Pydantic 的 str 类型会先拦成 422（只说"应为字符串"）
    dtype: Any = Field(
        None, description=f"计算精度，可选 {'/'.join(protocol.DTYPE_CHOICES)}"
                          f"（缺省 {protocol.DTYPE_DEFAULT}）")


class BizError(Exception):
    """业务接口错误：响应体 {"error": msg}（对齐 Flask jsonify 版本）。"""

    def __init__(self, status_code: int, message: str, headers: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.headers = headers


@app.exception_handler(BizError)
async def biz_error_handler(request, exc: BizError):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.message}, headers=exc.headers)


_ASR_READY_TTL_S = 2.0  # 探测结果缓存时长：asr 池掉线最多延迟 2s 被感知
_asr_ready_cache = {"ts": 0.0, "val": False}


def _scan_asr_ready() -> bool:
    """健康目录里是否有就绪的 asr 工人（/tts 快速失败，避免对齐单永悬挂）。"""
    try:
        for name in os.listdir(_health_dir):
            if not name.startswith("worker-"):
                continue
            path = os.path.join(_health_dir, name)
            try:
                if time.time() - os.path.getmtime(path) > _HEALTH_STALE_AFTER:
                    continue
                with open(path, encoding="utf-8") as f:
                    w = json.load(f)
                if w.get("role") == "asr" and (w.get("warm") or {}).get("funasr") == "ok":
                    return True
            except (OSError, ValueError):
                continue
    except OSError:
        pass
    return False


def _asr_ready() -> bool:
    """带 TTL 缓存的 asr 就绪探测：扫描是 listdir+mtime+json.load 的同步
    磁盘 IO，且 /tts 每请求都要过这道闸——绝不能在事件循环上逐请求扫盘。"""
    now = time.time()
    c = _asr_ready_cache
    if now - c["ts"] >= _ASR_READY_TTL_S:
        c["val"] = _scan_asr_ready()
        c["ts"] = now
    return c["val"]


@app.post("/tts")
async def tts_business(req: TTSRequest, request: Request = None):
    """业务兼容接口：返回 {audio_base64, duration, text, timestamp}。

    错误响应体为 {"error": msg}（对齐 Flask 版本）：400 缺少/空 text、
    429 服务繁忙（Retry-After）、503 时间戳服务未就绪、504 超时。
    timestamp 为字级毫秒时间戳 [[起ms, 止ms], ...]，由 asr 池
    （FunASR speech_timestamp_prediction-v1-16k-offline）经 chunk 级流水产出。
    """
    t_e2e = time.perf_counter()   # e2e 起点：进处理函数（校验/排队/合成/对齐/编码全程）
    deadline = t_e2e + _entry_wait_s   # 单一请求级截止线：音频+对齐共享预算
    if req.text is None:
        raise BizError(400, "缺少 text 参数")
    text = req.text.strip()
    if not text:
        raise BizError(400, "text 不能为空")
    if len(text) > _max_chars:
        raise BizError(400, f"text 超长({len(text)} > {_max_chars} 字符)，请分段请求")
    try:
        overrides = protocol.normalize_overrides(req.g2p_overrides)
        dtype = protocol.normalize_dtype(req.dtype)
    except ValueError as exc:
        raise BizError(400, str(exc))
    voice = (req.voice or _default_voice).strip()

    if not _asr_ready():
        raise BizError(503, "时间戳服务暂不可用（asr 池未就绪），请稍后重试")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    req_id = uuid.uuid4().hex[:12]
    # 时间戳 Future 必须在 _synth 之前注册：chunk_ts 可能先于音频聚合到达
    # 立刻就能命中，不存在"先收到通知后登记"的时序窗口
    fut_ts = loop.create_future()
    _pending[("ts", req_id)] = fut_ts
    blocks_desc = None
    try:
        _check_voice(voice)
        (audio, _texts, fw, _rid, blocks_desc,
         synth_timings, agg_s) = await _synth(
            text, voice, req.speed, needs_align=True, req_id=req_id,
            release_shm=False,  # 对齐期间 asr 还要按名读块，这里不回收
            request=request, g2p_overrides=overrides, dtype=dtype,
            deadline=deadline)
        # 对齐段断连 watcher：音频段完成后 _synth 的 watcher 已随 finally
        # 取消，此后的对齐等待（长文本可达几十秒）原感知不到断开——现在
        # 立即 499 释放连接与受理额度（对齐/asr 已派发部分照旧算完，结果
        # 按孤儿丢弃）
        ts_poll = None
        if request is not None:
            async def _watch_ts_disconnect() -> None:
                try:
                    while not await request.is_disconnected():
                        await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001  探测异常≠断连：退出不误判
                    return
                if not fut_ts.done():
                    fut_ts.set_exception(_ClientGone("客户端已断开"))
            ts_poll = asyncio.get_running_loop().create_task(_watch_ts_disconnect())
        try:
            # 对齐段用剩余预算（单一截止线）：音频耗掉多少，对齐就只剩多少
            ts = await asyncio.wait_for(
                fut_ts, timeout=max(deadline - time.perf_counter(), 0.0))
        finally:
            if ts_poll is not None:
                ts_poll.cancel()
    except asyncio.TimeoutError:
        # 只可能是对齐段超时（_synth 内部的音频超时已转 HTTPException）：
        # 必须计入 timeouts/failed，否则 /health、/metrics、wedge 都看不到
        # "音频成功、对齐永远等不到"的故障。
        _entry_stats["timeouts"] += 1
        _record_failure(504, "推理或对齐超时")
        raise BizError(504, "推理或对齐超时，请稍后重试")
    except _ClientGone:
        _entry_stats["cancelled"] = _entry_stats.get("cancelled", 0) + 1
        raise BizError(499, "客户端已断开")
    except _SynthError as exc:
        # 对齐服务故障 → 502 / 输入问题 → 400（原实现落到 generic 500）
        status = 400 if exc.kind == protocol.KIND_INVALID else 502
        log.info("req=%s alignment failed kind=%s: %s", req_id, exc.kind, exc)
        _record_failure(status, str(exc))
        raise BizError(status, str(exc))
    except HTTPException as exc:  # 429/502/504 等，转业务格式
        raise BizError(exc.status_code, str(exc.detail), exc.headers)
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed voice=%s", voice)
        _record_failure(500, f"TTS 生成失败: {exc}")
        raise BizError(500, f"TTS 生成失败: {exc}")
    finally:
        _pending.pop(("ts", req_id), None)
        _ts_state.pop(req_id, None)   # TS 账本就地回收（审计 F6 对称性）
        retired = _retired_blocks.pop(req_id, None)
        if retired:
            _unlink_blocks([b for b, _t in retired])
        if blocks_desc is not None:
            _unlink_blocks(blocks_desc)  # 收尾统一回收（chunk_ts 收齐后块已无人读）

    ts_timings = ts.get("timings") or {}   # asr 阶段耗时随 chunk_ts 捎带（ms）
    tts_cost = time.perf_counter() - t0
    duration = len(audio) / _sample_rate

    t1 = time.perf_counter()
    ts_list = ts.get("timestamp") or []
    # 大 payload 序列化整体下放线程池：base64 + json.dumps 都不在事件循环上
    # 跑（长文本的 base64 可达几十 MB，事件循环内同步 dumps 会阻塞同网关
    # 全部在途请求），回预构字节绕开 JSONResponse 的同步 render
    payload = {
        "audio_base64": None,  # 占位，线程内填充
        "code": 0,
        "duration": round(duration, 3),
        "message": "success",
        "text": req.text,   # 契约字段（README/冒烟测试）
        "timestamp": ts_list,
    }
    # g2p 兼容别名已移除：对外 6 键契约不含此字段，多键对严格
    # 校验的消费方是兼容性风险
    body = await run_in_threadpool(
        lambda: json.dumps(
            {**payload, "audio_base64": base64.b64encode(
                _encode_wav(audio)).decode("utf-8")},
            ensure_ascii=False).encode("utf-8"))
    enc_cost = time.perf_counter() - t1
    _entry_stats["results"] += 1   # /tts 成功口径 = 音频+对齐+编码全部完成

    # 单行汇总（原为多行 print 含完整文本——绕过日志级别且打 PII 全文）。
    # 只带网关独有字段：e2e_s（受理→响应全程）、agg_ms/enc_ms（网关本地
    # 开销）、audio_s（聚合产物）、voice/speed（请求参数）。批/句级明细
    # （排队、前向、对齐）由 tts/asr worker 的 bid=/chunk= 行承担——
    # grep <req_id> 串全程，grep <bid> 看批次构成。
    _record_latency(tts_cost, len(req.text))  # 延迟直方图口径不变 = 合成+对齐全程
    log.info("ep=%s req=%s voice=%s speed=%.2f audio_s=%.1f e2e_s=%.2f "
             "agg_ms=%.0f enc_ms=%.0f",
             request.url.path if request is not None else "/tts",
             req_id, voice, req.speed, duration,
             time.perf_counter() - t_e2e, agg_s * 1000, enc_cost * 1000)
    _log_request_text(req_id, text)

    return Response(content=body, media_type="application/json",
                    headers={"X-Request-ID": req_id})


@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def _encode_wav(audio: np.ndarray) -> bytes:
    """PCM16 int16 直写（soundfile 对 int16 输入无损直写，不再经 float）。"""
    buf = io.BytesIO()
    sf.write(buf, audio, _sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _encode_flac(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, _sample_rate, subtype="PCM_16", format="FLAC")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    return audio.tobytes()   # 已是 PCM16 小端


_FFMPEG_ARGS = {
    "mp3": ("-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"),
    "opus": ("-c:a", "libopus", "-b:a", "96k", "-f", "ogg"),
    "aac": ("-c:a", "aac", "-b:a", "128k", "-f", "adts"),
}


def _encode_ffmpeg(audio: np.ndarray, fmt: str) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(_sample_rate), "-ac", "1", "-i", "pipe:0",
        *_FFMPEG_ARGS[fmt], "pipe:1",
    ]
    # 超时必设：挂死的 ffmpeg 会永久占住 run_in_threadpool 的线程槽，
    # 反复挂死即耗尽线程池、拖垮所有编码请求（run 超时后自动 kill 子进程）
    try:
        proc = subprocess.run(cmd, input=_encode_pcm(audio),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=_ffmpeg_timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg {fmt} 编码超时(>{_ffmpeg_timeout_s:.0f}s)，已终止")
    except FileNotFoundError:
        # 缺二进制：裸穿会是 WinError 2 / ENOENT 乱码——给运维可读的原因
        raise RuntimeError(f"ffmpeg 未安装，无法编码 {fmt}（镜像内已内置；裸机部署需安装 ffmpeg）")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {fmt} 编码失败: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout


def gateway_main(i: int, port: int, g2p_queue, result_q) -> None:
    """supervisor 经 mp.Process 启动网关进程：注入队列句柄 → 绑定 → 跑 uvicorn。

    Linux 生产：自建 socket 并设 SO_REUSEPORT——每个网关独立监听 socket，
    内核按四元组哈希均匀分发（多网关负载均衡的关键）；Windows 冒烟：
    无 SO_REUSEPORT，每网关绑定独立端口（port+1000+i）。"""
    global _result_q, _g2p_queue, _worker_origin
    _result_q = result_q
    _g2p_queue = g2p_queue
    _worker_origin = int(i)
    os.environ["WORKER_TAG"] = f"gw-{i + 1}"
    import socket
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 重启后 TIME_WAIT 不阻塞 bind
    if hasattr(socket, "SO_REUSEPORT"):  # Linux 生产：同端口多网关，内核哈希分发
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        bind_port = port
    else:  # Windows 冒烟：每网关独立端口（跨进程共享 accept 与 asyncio proactor 不兼容）
        bind_port = port + 1000 + i
    sock.bind(("0.0.0.0", bind_port))
    sock.listen(2048)
    config = uvicorn.Config(app, log_level=proc_common.log_level_name(),
                            timeout_keep_alive=proc_common.env_int(
                                "KEEP_ALIVE_TIMEOUT_S", 0, minimum=0),
                            # 优雅退出上限 4s < supervisor join(5)：在途响应
                            # 自己收干净，避免 5s 后被 SIGKILL 硬切
                            timeout_graceful_shutdown=4)
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    raise SystemExit("gateway 由 supervisor.py 编排启动（注入队列句柄 + REUSEPORT 监听），"
                     "不可直接运行。本地联调用 supervisor.py；接口冒烟用 tests/test_tts_api.py。")

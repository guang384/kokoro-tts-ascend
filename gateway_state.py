"""gateway 状态层：全部模块级可变登记表 + 可配常量（拆分的状态单一归属）。

agg/obs/enc 与 gateway 主模块一律以 `import gateway_state as gst` + `gst._X`
方式访问——测试通过 setattr 替换本模块的绑定（dict/float 不可 from-import
捕获），模块属性查找保证补丁对所有使用方可见。
"""
import asyncio
import collections
import os
import time
from typing import Optional

import proc_common


class _SynthError(Exception):
    """下游失败终态（带错误分级：invalid_input→400，compute→502）。"""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


class _ClientGone(Exception):
    """客户端已断开（等待期间被断连 watcher 取消）。"""

_worker_idx = os.getenv("WORKER_IDX", "")
_worker_origin = int(_worker_idx) if _worker_idx.strip().lstrip("-").isdigit() else 0
_entry_wait_s = proc_common.env_float("WAIT_TIMEOUT_S", minimum=1.0)  # 单请求总预算（秒；/tts 音频+时间戳共享，与 SHM 孤儿回收公式同一解析器，小数秒合法）


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


FAST_REJECT_MARGIN = proc_common.env_float("FAST_REJECT_MARGIN", minimum=1.0)  # 准入快速拒绝冗余系数：预估等待 > 预算×系数 才拒（EMA 是估计值，边界放行）
_MAX_PENDING_REQUESTS = proc_common.env_int("MAX_PENDING_REQUESTS", minimum=1)  # /health 队列积压判定用
_WORKER_MAILBOX_SIZE = proc_common.env_int("WORKER_MAILBOX_SIZE", minimum=1)   # 同上（与 supervisor 同 env）


# 日志走 proc_common.setup_logging（wtag 过滤器与格式单一来源，带 asctime；
# worker/gateway 同一款）。logger 名保留 kokoro.server（测试 caplog 锚定）。


_sample_rate = proc_common.env_int("KOKORO_SAMPLE_RATE", minimum=8000)
_speed_floor = proc_common.env_float("TTS_SPEED_FLOOR", minimum=0.1)
_max_chars = proc_common.env_int("TTS_MAX_INPUT_CHARS", minimum=1)
_default_voice = proc_common.env_str("KOKORO_VOICE")
_ffmpeg_timeout_s = proc_common.env_float("FFMPEG_TIMEOUT_S", minimum=5.0)
_image_version = os.getenv("IMAGE_VERSION", "dev")  # 镜像版本号（构建期注入日期版本，本地直跑为 dev）
METRICS_ENABLED = proc_common.env_bool("METRICS_ENABLED")  # /metrics 开关（0 关闭返回 404）

# ── 心跳与 /health 聚合 ──
_health_dir = proc_common.HEALTH_DIR   # 单一来源（proc_common 模块常量）
_HEALTHBEAT_INTERVAL = 5.0
# 心跳陈旧阈值：与 supervisor 判 hang 用同一 env（默认 30s = 6×心跳周期），
# 原硬编码 15s 会在心跳轻微抖动时把活 worker 从视图剔除 → readiness 抖动
_HEALTH_STALE_AFTER = proc_common.env_float("HEARTBEAT_STALE_S", minimum=15.0)
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


_voices_cache: Optional[set] = None


_health_warn_ts = {}


# 无主音频块超龄回收线：必须大于块的合法最长持有期。块在 tts 前向时创建，
# /tts 最长合法生命周期现为单一截止线 ≤WAIT（音频+时间戳共享预算，不再
# 两段串行 2×WAIT）；公式 2×WAIT+60 保留为更保守的历史余量，不会误扫活块。
# 显式配置只能调大不能调小（低于公式值被抬高）。
SHM_ORPHAN_MAX_AGE_S = max(
    proc_common.env_float("SHM_ORPHAN_MAX_AGE_S", 300.0, minimum=30.0),
    2 * _entry_wait_s + 60.0)   # WAIT_TIMEOUT_S 只在模块头解析一次
_SHM_SWEEP_DIR = "/dev/shm"    # 音频 SHM 所在 tmpfs（测试可替换）


_MAX_BODY_BYTES = proc_common.env_int("MAX_BODY_BYTES", minimum=1024)


_ASR_READY_TTL_S = 2.0  # 探测结果缓存时长：asr 池掉线最多延迟 2s 被感知
_asr_ready_cache = {"ts": 0.0, "val": False}

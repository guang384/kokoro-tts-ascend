"""进程公共件·配置层：可调参数注册表、env 健壮解析、日志初始化。

proc_common 拆分出的叶子模块（无项目内依赖）——HF_HUB_OFFLINE 强制与
KOKORO_HEALTH_DIR 单一来源也在此（全仓库 import proc_common 即传导生效）。
"""
import logging
import os

# 硬置 1（忽略 docker run -e 覆盖）——意外触网的调用毫秒级快速失败
# 而非挂起等超时；想放开只能改代码，误配置物理上不可能。
os.environ["HF_HUB_OFFLINE"] = "1"

HEALTH_DIR = os.getenv("KOKORO_HEALTH_DIR", "/tmp/kokoro-health")


TUNABLE_PARAMS = {
    # ── 进程池数 ──
    "GATEWAY_WORKERS": "2", "G2P_WORKERS": "2",
    # G2P_THREADS 默认动态 =G2P_WORKERS（g2p.py 显式传，此值仅供 dump 展示）
    "G2P_THREADS": "2",
    "TTS_WORKERS": "4",
    "ASR_WORKERS": "4",
    # ── 背压三级闸门 ──
    "MAX_PENDING_REQUESTS": "200", "MAX_INFLIGHT_SENTENCES": "512",
    "REDISPATCH_MAX": "1",
    "WORKER_MAILBOX_SIZE": "1024",
    # ── 内容形状（显存上界三乘数 + 请求限制）──
    "TTS_CHUNK_CHARS": "120", "TTS_SPEED_FLOOR": "0.5",
    "TTS_MAX_BATCH_SIZE": "8", "ASR_MAX_BATCH_SIZE": "8",
    "TTS_MAX_INPUT_CHARS": "2000",
    "TTS_BATCH_BUCKET_CHARS": "64",
    # ── 超时与对齐 ──
    "WAIT_TIMEOUT_S": "120", "FAST_REJECT_MARGIN": "1.15",
    "HEARTBEAT_STALE_S": "30",
    "KEEP_ALIVE_TIMEOUT_S": "0", "STUCK_KILL_S": "300",
    "WEDGE_RESTART_S": "900",
    "FFMPEG_TIMEOUT_S": "30", "SPAWN_READY_TIMEOUT_S": "360",
    "SHM_ORPHAN_MAX_AGE_S": "300", "METRICS_ENABLED": "1",
    "MAX_BODY_BYTES": "4194304",
    "IPC_REDIS_SOCKET": "/tmp/kokoro-redis.sock",
    "IPC_REDIS_URL": "",
    # ── 功能开关与设备 ──
    "DEVICE": "", "ASCEND_DEVICE_ID": "",
    "KOKORO_EN_G2P": "auto", "KOKORO_DISABLE_COMPLEX": "auto",
    "FUNASR_DEVICE": "auto", "FUNASR_INPUT": "auto",
    "G2P_CACHE_SIZE": "4096", "KOKORO_G2P_DICT": "",
    # ── 模型身份（KOKORO_）──
    "KOKORO_MODEL_ID": "hexgrad/Kokoro-82M-v1.1-zh",
    "KOKORO_SAMPLE_RATE": "24000",
    "KOKORO_VOICE": "zf_001",
    # ── 平台 ──
    "PORT": "5174", "LOG_LEVEL": "INFO",
}


def dump_config(log_fn) -> None:
    """启动时打印全部可调配参数的实际值（env 覆盖后）。"""
    log_fn("──── tunable params (name=value [default]) ────")
    for name, default in TUNABLE_PARAMS.items():
        val = os.getenv(name, default)
        if name == "IPC_REDIS_URL" and "://" in val and "@" in val:
            # redis://:password@host:port —— 密码不进日志
            scheme, rest = val.split("://", 1)
            val = f"{scheme}://***@{rest.rsplit('@', 1)[-1]}"
        mark = "" if val == default else "  <- overridden"
        log_fn(f"  {name}={val} [{default}]{mark}")
    log_fn("──── internal injection (do not configure) ────")
    for name in ("WORKER_TAG", "WORKER_IDX", "KOKORO_HEALTH_DIR", "IMAGE_VERSION",
                 "FUNASR_MODEL_DIR", "HF_HUB_OFFLINE"):
        log_fn(f"  {name}={os.getenv(name, '<unset>')}")
    log_fn("──── end of config dump ───")


# env 解析告警走独立 logger（调用多发生在 setup_logging 之前/之外）
_cfg_log = logging.getLogger("kokoro.config")
_qlog = logging.getLogger("kokoro.queue")


def log_level(default: str = "INFO") -> int:
    """LOG_LEVEL 健壮解析：非法值回退默认，绝不在 import/启动期抛异常。

    原实现把 env 原样喂给 logging.basicConfig(level=...)——一个拼写错误
    （如 "INF0"）会让每个进程 import 期直接 ValueError 崩溃重启循环。"""
    raw = (os.getenv("LOG_LEVEL") or default).strip().upper()
    level = getattr(logging, raw, None)
    if not isinstance(level, int):
        _cfg_log.warning("LOG_LEVEL=%r 非法，使用默认 %s", raw, default)
        return getattr(logging, default.upper(), logging.INFO)
    return level


def log_level_name(default: str = "info") -> str:
    """uvicorn 用的级别名（它要字符串）；非法值回退默认。"""
    raw = (os.getenv("LOG_LEVEL") or default).strip().lower()
    if raw == "warn":
        return "warning"
    if raw in ("critical", "error", "warning", "info", "debug", "trace"):
        return raw
    _cfg_log.warning("LOG_LEVEL=%r 非法（uvicorn），使用默认 %s", raw, default)
    return default


def env_int(name: str, default: int | None = None,
            minimum: int | None = None) -> int:
    """int(os.getenv(...)) 的健壮版：坏值告警 + 回退默认（生产可用性优先，
    而非裸 ValueError 崩溃重启循环）。空串视同未设置。

    default 省略时查 TUNABLE_PARAMS 注册表（默认值单一来源；未注册的
    env 属编程错误，KeyError 启动即炸——宁可响也不要静默猜）。"""
    if default is None:
        default = int(TUNABLE_PARAMS[name])
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        _cfg_log.warning("env %s=%r 非整数，使用默认值 %s", name, raw, default)
        return default
    if minimum is not None and val < minimum:
        _cfg_log.warning("env %s=%d 低于下限 %s，钳制为下限", name, val, minimum)
        return minimum
    return val


def env_float(name: str, default: float | None = None,
              minimum: float | None = None) -> float:
    """同 env_int 的浮点版（default 省略时查注册表）。"""
    if default is None:
        default = float(TUNABLE_PARAMS[name])
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        _cfg_log.warning("env %s=%r 非数值，使用默认值 %s", name, raw, default)
        return default
    if minimum is not None and val < minimum:
        _cfg_log.warning("env %s=%s 低于下限 %s，钳制为下限", name, val, minimum)
        return minimum
    return val


def _bool_of(raw: str) -> bool | None:
    """布尔字面量解析；不识别返回 None（由调用方决定告警/回退）。"""
    if raw in ("1", "true", "y", "yes", "on"):
        return True
    if raw in ("0", "false", "n", "no", "off"):
        return False
    return None


def env_bool(name: str, default: bool | None = None) -> bool:
    """布尔 env 解析：1/true/y/on=真，0/false/n/off=假（大小写不敏感）；
    坏值告警回退默认；未设置返回默认（省略时查注册表）。"""
    if default is None and name in TUNABLE_PARAMS:
        default = _bool_of(TUNABLE_PARAMS[name].lower())
        if default is None:
            raise KeyError(f"TUNABLE_PARAMS[{name}] 不是合法布尔字面量")
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    val = _bool_of(raw)
    if val is None:
        _cfg_log.warning("env %s=%r 非布尔（1/0/true/false），使用默认 %s",
                         name, raw, default)
        return default
    return val


def env_str(name: str, default: str | None = None) -> str:
    """字符串 env（strip 后取用；未设置/空串回退 default，省略时查注册表）。"""
    raw = (os.getenv(name) or "").strip()
    if raw:
        return raw
    if default is None:
        default = TUNABLE_PARAMS.get(name, "")
    return default


_wtag_filter = None  # 单例可变 tag 过滤器（幂等关键：重复调用不堆积）


def setup_logging(tag: str) -> logging.Logger:
    """带 [角色标签] 前缀的日志初始化（幂等，重复调用安全）。

    单例过滤器 + 可变 tag：重复调用只改 tag。历史实现每次调用都往 root
    handlers 追加一个新过滤器——g2p 在模块 import 与 main 各调
    一次，过滤器堆积且最终 wtag 取决于调用顺序。"""
    global _wtag_filter

    class _WTagFilter(logging.Filter):
        def filter(self, record):
            record.wtag = self.tag
            return True

    logging.basicConfig(level=log_level(),
                        format="[%(wtag)s] %(asctime)s "
                               "%(levelname)s:%(name)s:%(message)s")
    if _wtag_filter is None:
        _wtag_filter = _WTagFilter()
        for h in logging.getLogger().handlers:
            h.addFilter(_wtag_filter)
    _wtag_filter.tag = tag
    return logging.getLogger("kokoro." + tag)

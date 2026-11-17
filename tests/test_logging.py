"""日志体系改造回归：文本截断策略、/health 告警节流、X-Request-ID、
supervisor 正式日志、benchmark 随机文本。"""
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gateway
import gateway_obs
import gateway_state
import supervisor


@pytest.fixture(autouse=True)
def _reset_throttle(monkeypatch):
    monkeypatch.setattr(gateway_state, "_health_warn_ts", {})


# ── 请求文本日志策略（应用方要求：INFO 打完整 TEXT，单行）──

def test_request_text_full_at_info(caplog):
    text = "这是一段非常长的用户请求文本" * 10
    with caplog.at_level(logging.INFO, logger=gateway.log.name):
        gateway._log_request_text("rid-1", text)
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any(text in r.message for r in infos), "INFO 必须打完整文本"
    assert any("chars=%d" % len(text) in r.message for r in infos)


def test_request_text_newlines_flattened(caplog):
    text = "第一行。" + chr(10) + "第二行！" + chr(10) + "第三行"
    with caplog.at_level(logging.INFO, logger=gateway.log.name):
        gateway._log_request_text("rid-2", text)
    msg = caplog.records[-1].message
    assert chr(10) not in msg, "日志必须单行（换行替换）"
    assert "第一行。 第二行！ 第三行" in msg


# ── /health 告警节流（监控轮询不刷屏）──

def test_throttled_warn_once_per_key_per_window(caplog):
    with caplog.at_level(logging.WARNING, logger=gateway.log.name):
        for _ in range(5):   # 模拟监控 5 次连续抓取
            gateway_obs._throttled_warn("missing:tts 5/6", "missing role: %s", "tts 5/6")
    assert len([r for r in caplog.records if "missing role" in r.message]) == 1


def test_throttled_warn_independent_keys(caplog):
    with caplog.at_level(logging.WARNING, logger=gateway.log.name):
        gateway_obs._throttled_warn("missing:tts", "missing role: %s", "tts")
        gateway_obs._throttled_warn("backlog:asr_queue", "queue backlog: %s", "asr_queue")
    assert len(caplog.records) == 2


def test_throttled_warn_rearms_after_window(caplog, monkeypatch):
    monkeypatch.setattr(gateway.time, "time", lambda: 1000.0)
    gateway_obs._throttled_warn("k", "msg")                       # 首发（caplog 全程抓取）
    monkeypatch.setattr(gateway.time, "time", lambda: 1061.0)  # 窗口已过 → 重挂
    gateway_obs._throttled_warn("k", "msg")
    monkeypatch.setattr(gateway.time, "time", lambda: 1070.0)  # 窗口内 → 静默
    gateway_obs._throttled_warn("k", "msg")
    assert len([r for r in caplog.records if r.message == "msg"]) == 2


# ── supervisor 正式日志（print 全部退役）──

def test_supervisor_uses_logging_not_print():
    assert supervisor.log.name == "kokoro.supervisor"
    src = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert "print(" not in src, "supervisor 仍残留 print 输出"


def test_worker_module_import_does_not_steal_log_tag():
    """spawn 前 supervisor 父进程会 import g2p/gateway 取 target；这两个模块
    若在 import 时 setup_logging，会把父进程 tag 改成 [g2p]/[pidN]（实测
    supervisor 行变 [pid1]）。用子进程隔离验证 import 后 tag 仍为 supervisor。"""
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import proc_common, config;"
        "proc_common.setup_logging('supervisor');"
        "import g2p, gateway;"
        "print(config._wtag_filter.tag)"
    ) % str(root)
    out = subprocess.run([sys.executable, "-c", code], cwd=root,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip().splitlines()[-1] == "supervisor"


def test_gateway_main_initializes_gw_log_tag(monkeypatch):
    """gateway_main 在子进程入口按 gw-N 初始化日志 tag（模块 import 不再
    做）；否则 uvicorn 日志/请求行会带 [pidN]，与 README 的 [gw-1] 不符。"""
    import socket
    import uvicorn

    calls = []

    def fake_setup(tag):
        calls.append(tag)
        return logging.getLogger("kokoro.server")

    class _Sock:
        def setsockopt(self, *a):
            pass

        def bind(self, *a):
            pass

        def listen(self, *a):
            pass

    class _Server:
        def __init__(self, config):
            self.config = config

        async def serve(self, sockets=None):
            pass

    monkeypatch.setattr(gateway.proc_common, "setup_logging", fake_setup)
    monkeypatch.setattr(socket, "socket", lambda *a: _Sock())
    monkeypatch.setattr(uvicorn, "Server", _Server)
    monkeypatch.setattr(gateway.asyncio, "run", lambda coro: coro.close())
    old_env = os.environ.get("WORKER_TAG")
    old = (gateway.gst._worker_origin, gateway.gst._result_q,
           gateway.gst._g2p_queue)
    try:
        gateway.gateway_main(1, 5174, None, None)
    finally:
        if old_env is None:
            os.environ.pop("WORKER_TAG", None)
        else:
            os.environ["WORKER_TAG"] = old_env
        (gateway.gst._worker_origin, gateway.gst._result_q,
         gateway.gst._g2p_queue) = old
    assert calls == ["gw-2"]


# ── benchmark 随机文本（压测打穿 g2p 缓存的前提）──

@pytest.fixture()
def benchmark_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark

    return benchmark


def test_benchmark_random_texts_distinct(benchmark_mod):
    texts = benchmark_mod.random_texts(50)
    assert len(texts) == 50 and len(set(texts)) == 50   # 互不相同
    assert all(len(t) >= 2 for t in texts)


def test_benchmark_random_texts_reproducible(benchmark_mod):
    assert benchmark_mod.random_texts(20) == benchmark_mod.random_texts(20)


def test_benchmark_random_texts_mixed_lengths(benchmark_mod):
    """长短分层：短单（问候/重试形态）、多句长单（>TTS_CHUNK_CHARS=120，
    触发多批聚合）、超长单（≥400 字，触发对齐多组切分）都要出现。"""
    texts = benchmark_mod.random_texts(50)
    lens = [len(t) for t in texts]
    assert min(lens) <= 8                                   # 短层存在
    assert sum(1 for L in lens if L > 120) >= 10            # 长层：多句 chunk 路径
    assert max(lens) >= 400                                 # 超长层：最坏情况
    long_ones = [t for t in texts if len(t) > 120]
    assert any(c in long_ones[0] for c in "。！？；")       # 长单按句构造（对齐切组可切）


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_logging.py")

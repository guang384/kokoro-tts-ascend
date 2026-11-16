"""性能优化的行为回归：/tts 就绪探测缓存、g2p LRU 缓存、ffmpeg 超时。"""
import random
import subprocess as _sp

import pytest

import gateway
import gateway_state
import audio_enc
import g2p
from tests import rnd


# ── _asr_ready TTL 缓存（事件循环上不再逐请求扫盘）──

@pytest.fixture()
def scan_counter(monkeypatch):
    monkeypatch.setattr(gateway_state, "_asr_ready_cache", {"ts": 0.0, "val": False})
    calls = {"n": 0}

    def _scan():
        calls["n"] += 1
        return True

    monkeypatch.setattr(gateway, "_scan_asr_ready", _scan)
    return calls


def test_asr_ready_cached_within_ttl(scan_counter):
    assert gateway._asr_ready() is True
    assert gateway._asr_ready() is True   # TTL 内：命中缓存
    assert gateway._asr_ready() is True
    assert scan_counter["n"] == 1


def test_asr_ready_rescans_after_ttl(scan_counter, monkeypatch):
    monkeypatch.setattr(gateway_state, "_ASR_READY_TTL_S", 0.0)  # 每次都过期
    assert gateway._asr_ready() is True
    assert gateway._asr_ready() is True
    assert scan_counter["n"] == 2


# ── g2p LRU 缓存（make_phonemizer）──

class _CountingEngine:
    def __init__(self):
        self.calls = 0

    def phonemize(self, s):
        self.calls += 1
        return [f"ph:{s}"]


def test_phonemizer_caches_repeated_sentences():
    """随机句子集（含重复）送缓存 phonemize：底层引擎只跑去重后的句数。"""
    rng = random.Random(1120)
    engine = _CountingEngine()
    ph = g2p.make_phonemizer(engine, maxsize=128)
    sentences = [rnd.sentence(rng, 2, 8) for _ in range(24)]
    sentences += rng.sample(sentences, 8)  # 一半句子重复出现（生产常态）
    results = [ph(s) for s in sentences]
    assert results[0] == [f"ph:{sentences[0]}"]   # 返回 list（防外改缓存条目）
    assert engine.calls == len(set(sentences))
    assert ph.cache_info().hits >= 8


def test_phonemizer_cache_disabled_at_zero():
    rng = random.Random(2333)
    engine = _CountingEngine()
    ph = g2p.make_phonemizer(engine, maxsize=0)
    s = rnd.sentence(rng)
    ph(s)
    ph(s)
    assert engine.calls == 2   # maxsize=0 = 完全关闭


def test_phonemizer_result_not_aliased():
    """调用方改返回值不得污染缓存条目。"""
    engine = _CountingEngine()
    ph = g2p.make_phonemizer(engine, maxsize=8)
    r1 = ph("固定句")
    r1.append("污染")
    assert ph("固定句") == ["ph:固定句"]


# ── ffmpeg 编码超时（挂死不再永久占线程池槽）──

class _FakeCompleted(_sp.CompletedProcess):  # 便于构造成功返回
    def __init__(self):
        super().__init__(args=[], returncode=0, stdout=b"enc", stderr=b"")


def test_encode_ffmpeg_passes_timeout_and_returns_stdout(monkeypatch):
    seen = {}

    def _run(cmd, **kw):
        seen.update(kw)
        return _FakeCompleted()

    monkeypatch.setattr(gateway.subprocess, "run", _run)
    out = audio_enc._encode_ffmpeg(__import__("numpy").zeros(8, "float32"), "mp3")
    assert out == b"enc" and seen["timeout"] == gateway_state._ffmpeg_timeout_s


def test_encode_ffmpeg_timeout_raises_runtime(monkeypatch):
    def _run(cmd, **kw):
        raise _sp.TimeoutExpired(cmd="ffmpeg", timeout=kw["timeout"])

    monkeypatch.setattr(gateway.subprocess, "run", _run)
    with pytest.raises(RuntimeError, match="编码超时"):
        audio_enc._encode_ffmpeg(__import__("numpy").zeros(8, "float32"), "mp3")


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_perf.py")

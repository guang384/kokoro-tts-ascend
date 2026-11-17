"""NPU 路径单元验证（无卡）：预算/预占/估算/容量/设备选择。

CPU E2E 会跳过所有 `device.startswith("npu")` 分支（预算、预占、npu-smi
显存估算、capacity_check、纯 conv STFT），而这些正是真机 ARM/NPU 的核心
路径。本文件用 fake torch/torch.npu 把这些分支的数值与调用契约锁住。
"""
import sys
from types import SimpleNamespace

import pytest

import npu_ops


class FakeNpu:
    def __init__(self, total_bytes=64 * 2**30, fail=False):
        self.total_bytes = total_bytes
        self.fail = fail
        self.fractions = []
        self.prealloc = []
        self.allocated = 10 * 2**20
        self.reserved = 9500 * 2**20

    def get_device_properties(self, idx):
        if self.fail:
            raise RuntimeError("no npu")
        return SimpleNamespace(total_memory=self.total_bytes)

    def set_per_process_memory_fraction(self, frac, idx=0):
        self.fractions.append((frac, idx))

    def memory_allocated(self, idx=0):
        return self.allocated

    def memory_reserved(self, idx=0):
        return self.reserved


def fake_torch(npu=None):
    npu = npu or FakeNpu()

    def empty(size, dtype=None, device=None):
        npu.prealloc.append(int(size))
        return SimpleNamespace(size=int(size))

    torch = SimpleNamespace(npu=npu, uint8="uint8", empty=empty,
                            device=lambda spec: spec)
    return torch, npu


def test_tts_budget_anchor_values():
    """NPU 显存预算公式锚点（B=8/120 字/0.5 速 → 9.5GB；B=4 → 5.2GB）。
    CPU E2E 不会执行 _prime_memory，这里锁死真机预算数值。"""
    assert npu_ops._auto_budget_mb(8, 120, 0.5) == (9543, 126, 55.1)
    assert npu_ops._auto_budget_mb(4, 120, 0.5) == (5228, 126, 55.1)
    assert npu_ops._asr_auto_budget_mb(8, 30.0) == (2689, 500, 30.0)
    assert npu_ops._asr_auto_budget_mb(4, 30.0) == (1818, 500, 30.0)


def test_prime_memory_sets_fraction_and_prealloc(monkeypatch):
    torch, npu = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(npu_ops, "_last_budget", None)
    monkeypatch.setattr(npu_ops, "_budget_error", None)
    for k in ("TTS_MAX_BATCH_SIZE", "TTS_CHUNK_CHARS", "TTS_SPEED_FLOOR"):
        monkeypatch.delenv(k, raising=False)

    npu_ops._prime_memory("npu:0")
    budget_mb = 9543
    budget = budget_mb * 2**20
    assert npu.fractions == [(budget * 1.05 / npu.total_bytes, 0)]
    assert npu.prealloc == [int(budget * 0.98)]
    assert npu_ops._last_budget == (9543, 126, 55.1)
    assert npu_ops.budget_error() is None


def test_prime_memory_failure_is_observable(monkeypatch):
    torch, _npu = fake_torch(FakeNpu(fail=True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(npu_ops, "_last_budget", None)
    monkeypatch.setattr(npu_ops, "_budget_error", None)

    npu_ops._prime_memory("npu:0")     # 不抛：失败不阻塞服务
    assert npu_ops._last_budget is None
    assert "no npu" in (npu_ops.budget_error() or "")


def test_prime_asr_memory_sets_fraction_with_context_headroom(monkeypatch):
    torch, npu = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(npu_ops, "_last_asr_budget", None)
    monkeypatch.setattr(npu_ops, "_asr_budget_error", None)
    monkeypatch.delenv("ASR_MAX_BATCH_SIZE", raising=False)

    npu_ops.prime_asr_memory("npu:0")
    budget = 2689 * 2**20
    expected_frac = (budget * 0.98 + 256 * 2**20) / npu.total_bytes
    assert npu.fractions == [(expected_frac, 0)]
    assert npu.prealloc == [int(budget * 0.98)]
    assert npu_ops.asr_budget_meta() == (2689, 500, 30.0)
    assert npu_ops.asr_budget_error() is None


def test_prime_memory_skips_cpu(monkeypatch):
    called = {"n": 0}

    class Boom:
        def __getattr__(self, name):
            called["n"] += 1
            raise AssertionError("CPU 不应触达 torch.npu")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=Boom()))
    npu_ops._prime_memory("cpu")
    npu_ops.prime_asr_memory("cpu")
    assert called["n"] == 0


def test_npu_mem_estimate_reads_torch_counters(monkeypatch):
    torch, npu = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(npu_ops, "_DEVICE_TOTAL_MB", {})
    monkeypatch.setattr(npu_ops, "smi_process_mb", lambda dev: 10000.0)

    d = npu_ops.npu_mem_estimate("npu:0", budget_mb=9543, context_mb=450)
    assert d["allocated_mb"] == 10.0
    assert d["reserved_mb"] == 9500.0
    assert d["estimated_mb"] == 9950.0
    assert d["planned_mb"] == pytest.approx(0.98 * 9543 + 450, abs=0.2)
    assert d["process_mb"] == 10000.0
    assert d["delta_mb"] == pytest.approx(10000 - (0.98 * 9543 + 450), abs=0.2)
    assert d["budget_pct"] == pytest.approx(9500 / 9543 * 100, abs=0.2)
    # 估算只允许主线程调用；这里同时验证 CPU 返回 None
    assert npu_ops.npu_mem_estimate("cpu") is None


def test_capacity_check_formula_and_threshold(monkeypatch):
    monkeypatch.setattr(npu_ops, "_npu_total_mb", lambda: 65536.0)
    for k in ("TTS_MAX_BATCH_SIZE", "TTS_CHUNK_CHARS", "TTS_SPEED_FLOOR",
              "ASR_MAX_BATCH_SIZE"):
        monkeypatch.delenv(k, raising=False)

    tts_b = npu_ops._auto_budget_mb(8, 120, 0.5)[0]
    asr_b = npu_ops._asr_auto_budget_mb(8, 30.0)[0]
    expect = (4 * (1.05 * tts_b + npu_ops.TTS_CONTEXT_MB)
              + 4 * (1.05 * asr_b + npu_ops.ASR_CONTEXT_MB)) \
        * npu_ops.CAPACITY_SAFETY
    total, phys, over = npu_ops.capacity_check(4, 4)
    assert total == 65536.0
    assert phys == round(expect, 1)
    assert over is False
    total5, phys5, over5 = npu_ops.capacity_check(5, 4)
    assert over5 is True and phys5 > 0.95 * total5


def test_disable_complex_auto_follows_device(monkeypatch):
    monkeypatch.setenv("KOKORO_DISABLE_COMPLEX", "auto")
    assert npu_ops._resolve_disable_complex("npu:0") is True
    assert npu_ops._resolve_disable_complex("cpu") is False
    monkeypatch.setenv("KOKORO_DISABLE_COMPLEX", "1")
    assert npu_ops._resolve_disable_complex("cpu") is True
    monkeypatch.setenv("KOKORO_DISABLE_COMPLEX", "0")
    assert npu_ops._resolve_disable_complex("npu:0") is False


def test_resolve_device_explicit_and_fail_fast(monkeypatch):
    monkeypatch.setenv("DEVICE", "cpu")
    assert npu_ops._resolve_device() == "cpu"
    monkeypatch.setenv("DEVICE", "npu:0")
    assert npu_ops._resolve_device() == "npu:0"

    # auto：torch_npu 缺失 → CPU；存在但无卡 → 严格 fail-fast
    monkeypatch.delenv("DEVICE", raising=False)
    import importlib

    real_import = importlib.import_module

    def _missing(name, *a, **k):
        if name == "torch_npu":
            raise ImportError("missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _missing)
    assert npu_ops._resolve_device() == "cpu"

    def _present(name, *a, **k):
        if name == "torch_npu":
            return SimpleNamespace()
        return real_import(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", _present)
    monkeypatch.setattr(npu_ops, "_pick_npu_device", lambda: "")
    with pytest.raises(RuntimeError, match="没有任何可用 NPU"):
        npu_ops._resolve_device()

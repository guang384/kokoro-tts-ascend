"""NPU 设备探测：npu-smi 驱动层卡号解析 + 候选生成（vNPU 切分/池化场景）。

vNPU 故障模式（HAMi/cnos-vpool 等切分池化）：CANN Runtime 枚举缺失/按
物理槽位编号，torch.npu.device_count() 抛异常或返回 0（count=1 但真标识
是物理卡 7，甚至 count=0 但 npu-smi 看得见卡）——npu-smi/编排层注入的
env 走驱动接口仍能报真实卡号。候选生成必须把「显式 ASCEND_DEVICE_ID
（不丢）→ torch 枚举 → env 可见卡 → smi 卡号」并集，且枚举缺失时绝不
提前放弃（历史缺陷：count<=0 直接 return，驱动层兜底永远走不到）。"""
import sys
import types

import pipeline

_SMI_OUT = """\
| npu-smi 23.0.3                                     Version: 23.0.3                               |
| NPU       Name               Health          Bus-Id          Power(W)  Temp(C)  Memory-Usage(MB)  |
| Chip                                                 AICore(%)                                     |
| 7         910B2              OK              0000:42:00.0    97.1     56       xxx / xxx          |
| 0                                                           0       0       xxx / xxx             |
| Processes:                                                                                    |
| NPU  Chip   Process id   Process name                                                          |
| 7    0      292          python                                                                |
"""

_SMI_MULTI = """\
| NPU       Name               Health          Bus-Id          Power(W)     Memory-Usage(MB)       |
| 0         310P3              OK              0000:81:00.0    62.0         xxx / xxx              |
| 3         910B2              OK              0000:C2:00.0    97.1         xxx / xxx              |
| 7         Ascend910A         OK              0000:42:00.0    93.5         xxx / xxx              |
| Processes:  process id不可用                                                                 |
| 7    0      292          python                                                                |
"""


def _with_smi_output(monkeypatch, out=None, raise_exc=None):
    """把真实的 pipeline._smi_card_ids 驱动起来（monkeypatch subprocess）——
    不再用解析逻辑的复制品：实现漂移时复制品测试照样绿。"""
    import subprocess as sp

    def fake_run(cmd, **kw):
        assert cmd[:1] == ["npu-smi"]
        if raise_exc is not None:
            raise raise_exc
        return types.SimpleNamespace(stdout=out, returncode=0)

    monkeypatch.setattr(sp, "run", fake_run)


def test_smi_parses_chip_ids_and_skips_process_table(monkeypatch):
    """芯片行首列数字+次列型号 → 卡号；Chip 子行（首列无型号）与进程表
    （PID 首列）不误读；解析在进程表处截断。"""
    _with_smi_output(monkeypatch, _SMI_OUT)
    assert pipeline._smi_card_ids() == [7]


def test_smi_real_world_variants(monkeypatch):
    """多卡表 + 310P3 / Ascend910A 型号变体 + 进程表截断（中文"不可用"）。"""
    _with_smi_output(monkeypatch, _SMI_MULTI)
    assert pipeline._smi_card_ids() == [0, 3, 7]


def test_smi_missing_or_timeout_returns_empty(monkeypatch):
    """无二进制（FileNotFoundError）/超时（TimeoutExpired）→ [] 静默回退。"""
    import subprocess as sp

    _with_smi_output(monkeypatch, raise_exc=FileNotFoundError("npu-smi"))
    assert pipeline._smi_card_ids() == []
    _with_smi_output(monkeypatch, raise_exc=sp.TimeoutExpired(cmd="npu-smi", timeout=10))
    assert pipeline._smi_card_ids() == []


# ── 进程表解析：npu-smi 实测进程内存（process_mb）──

_SMI_PROC = """\
| npu-smi 23.0.3                                     Version: 23.0.3                               |
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 5       0                 | 967120        | python3                  | 9683                    |
| 5       0                 | 968566        | python3                  | 2196                    |
"""

_SMI_PROC_CONTAINER = """\
| NPU     Chip              | Process id    | Process id in container  | Process name             | Process memory(MB)      |
| 5       0                 | 967120        | 2196                     | python3                  | 9683                    |
"""

_SMI_PROC_MULTI = """\
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 5       0                 | 967120        | python3                  | 5000                    |
| 6       0                 | 967120        | python3                  | 4683                    |
"""

_SMI_PROC_NO_MEM = """\
| NPU     Chip              | Process id    | Process name             |
| 5       0                 | 967120        | python3                  |
"""

_SMI_PROC_SHORT = """\
Process id:967120 Process name:python3 Process memory(MB):9683
Process id:968566 Process name:python3 Process memory(MB):2196
"""


def test_smi_process_mem_parses_matching_pid():
    assert pipeline._parse_smi_process_mem(_SMI_PROC, {967120}) == 9683.0
    assert pipeline._parse_smi_process_mem(_SMI_PROC, {968566}) == 2196.0


def test_smi_process_mem_prefers_container_pid_column():
    """新版本带 Process id in container：用容器 PID 匹配，而不是宿主 PID。"""
    assert pipeline._parse_smi_process_mem(_SMI_PROC_CONTAINER, {2196}) == 9683.0
    assert pipeline._parse_smi_process_mem(_SMI_PROC_CONTAINER, {967120}) == 9683.0


def test_smi_process_mem_sums_multi_chip_rows():
    assert pipeline._parse_smi_process_mem(_SMI_PROC_MULTI, {967120}) == 9683.0


def test_smi_process_mem_no_memory_column_or_no_match_returns_none():
    assert pipeline._parse_smi_process_mem(_SMI_PROC_NO_MEM, {967120}) is None
    assert pipeline._parse_smi_process_mem(_SMI_PROC, {1}) is None
    assert pipeline._parse_smi_process_mem("", {967120}) is None


def test_smi_proc_mem_short_parser():
    assert pipeline._parse_smi_proc_mem_short(_SMI_PROC_SHORT, {967120}) == 9683.0
    assert pipeline._parse_smi_proc_mem_short(_SMI_PROC_SHORT, {968566}) == 2196.0
    assert pipeline._parse_smi_proc_mem_short(_SMI_PROC_SHORT, {1}) is None


def test_smi_process_mb_cache_and_fallback(monkeypatch):
    """完整表未命中 → `-t proc-mem` 兜底；命中后 30s 内不再 fork。"""
    calls = []

    def _run(cmd):
        calls.append(tuple(cmd))
        if tuple(cmd) == ("npu-smi", "info"):
            return _SMI_PROC_MULTI.replace("967120", "111111")  # 不含本 PID
        if "-t" in cmd and "proc-mem" in cmd:
            return "Process id:967120 Process name:python3 Process memory(MB):9683"
        return ""

    monkeypatch.setattr(pipeline, "_run_cmd",
                        lambda cmd, t, warn="": _run(cmd))
    monkeypatch.setattr(pipeline, "_pid_candidates", lambda: {967120})
    monkeypatch.setattr(pipeline, "_SMI_PROC_CACHE",
                        {"ts": 0.0, "pid": None, "dev": None, "mb": None})
    assert pipeline.smi_process_mb("npu:0") == 9683.0
    assert calls == [("npu-smi", "info"),
                     ("npu-smi", "info", "-t", "proc-mem", "-i", "0")]
    assert pipeline.smi_process_mb("npu:0") == 9683.0   # 缓存命中
    assert len(calls) == 2


def test_smi_process_mb_negative_cache(monkeypatch):
    """npu-smi 缺失/超时：返回 None；负缓存内不重复 fork。"""
    calls = []

    def _run(cmd):
        calls.append(tuple(cmd))
        return ""

    monkeypatch.setattr(pipeline, "_run_cmd",
                        lambda cmd, t, warn="": _run(cmd))
    monkeypatch.setattr(pipeline, "_pid_candidates", lambda: {967120})
    monkeypatch.setattr(pipeline, "_SMI_PROC_CACHE",
                        {"ts": 0.0, "pid": None, "dev": None, "mb": None})
    assert pipeline.smi_process_mb("npu:0") is None
    assert pipeline.smi_process_mb("npu:0") is None
    assert len(calls) == 2          # 首轮 info + proc-mem 兜底各一次


def _fake_torch(calls, ok_name="npu:7", count=1, count_raises=False):
    def _zeros(n, device=None):
        calls.append(device)
        if device != ok_name:
            raise RuntimeError("LazySetDevice failed (Device:0)")
        return object()

    def _count():
        if count_raises:
            raise RuntimeError("Driver: enum devices init failed")
        return count

    return types.SimpleNamespace(
        npu=types.SimpleNamespace(device_count=_count,
                                  synchronize=lambda d: None),
        device=lambda name: name,
        zeros=_zeros,
    )


def test_smi_cards_feed_probe_order(monkeypatch):
    """显式空 + torch 枚举全失败（count=1 只有 npu:0，probe 失败）→ smi
    卡号兜底命中 npu:7。探测真执行通过即为可用设备：枚举先试、smi 后补。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    assert pipeline._pick_npu_device() == "npu:7"
    assert calls == ["npu:0", "npu:7"]           # 枚举先试、smi 兜底


def test_multi_card_enumeration_distributes_by_pid(monkeypatch):
    """多卡容器（全部 probe 通过）：按 pid 取模打散——smi 物理卡号全集
    不得抢在枚举前把全部 worker 压在最低卡（发现逻辑曾有此缺陷）。"""
    import os as _os

    calls = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(calls, ok_name=None, count=2))  # None: 全失败
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")

    def _ok_zeros(n, device=None):
        calls.append(device)
        return object()

    torch_mod = sys.modules["torch"]
    torch_mod.zeros = _ok_zeros
    # smi 报出越界物理卡 7：若它排在枚举前，probe 顺序会先试 7
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    dev = pipeline._pick_npu_device()
    assert dev == f"npu:{_os.getpid() % 2}"      # pid 打散起点生效
    assert calls == [f"npu:{_os.getpid() % 2}"]  # 枚举首个候选即命中
    assert "npu:7" not in calls                  # smi 未抢在枚举前探测


def test_explicit_id_kept_even_if_out_of_range(monkeypatch):
    """ASCEND_DEVICE_ID 越界（vNPU 常态：count=1 但真标识是物理卡 7）
    不丢弃：显式值优先实探。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "7")
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [])
    assert pipeline._pick_npu_device() == "npu:7"
    assert calls[0] == "npu:7"


def test_smi_unavailable_falls_back_to_enumeration(monkeypatch):
    """npu-smi 缺失/解析失败：静默回退 torch 枚举（增强非依赖）。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls, ok_name="npu:0"))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [])
    assert pipeline._pick_npu_device() == "npu:0"


def test_all_probes_fail_returns_empty(monkeypatch):
    """全部候选失败 → ""（_resolve_device 走 fail-fast 或 CPU 逃生口）。"""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([], ok_name=None))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "7")
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    assert pipeline._pick_npu_device() == ""


# ── vNPU 切分（HAMi）：CANN 枚举缺失（count=0/异常）时不得提前放弃 ──

def _clear_env(monkeypatch):
    """清掉编排层可见卡 env（宿主机/CI 可能残留，污染候选顺序断言）。"""
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)


def test_count_zero_smi_fallback_still_probes(monkeypatch):
    """vNPU 切分核心故障（HAMi 实测）：CANN 枚举返回 0 但驱动层看得见卡——
    历史缺陷是 count<=0 直接 return，smi 兜底永远走不到。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls, count=0))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    _clear_env(monkeypatch)
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    assert pipeline._pick_npu_device() == "npu:7"
    assert calls == ["npu:7"]


def test_count_raises_smi_fallback_still_probes(monkeypatch):
    """device_count() 抛异常（CANN 枚举整体崩溃）同样不得提前放弃。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(calls, count_raises=True))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    _clear_env(monkeypatch)
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    assert pipeline._pick_npu_device() == "npu:7"


def test_count_zero_no_driver_candidates_fails_fast(monkeypatch):
    """count=0 且显式/env/smi 全无候选 → ""（fail-fast 语义不丢）。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(calls, count=0))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    _clear_env(monkeypatch)
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [])
    assert pipeline._pick_npu_device() == ""
    assert calls == []                            # 无候选则一次都不探


def test_hami_visible_env_ids_feed_candidates(monkeypatch):
    """HAMi 注入的 ASCEND_RT_VISIBLE_DEVICES 在枚举缺失时进候选实探——
    它是"分给本容器的精确集合"。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(calls, ok_name="npu:4", count=0))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,5")
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [])
    assert pipeline._pick_npu_device() == "npu:4"
    assert calls == ["npu:4"]


def test_explicit_id_first_at_count_zero(monkeypatch):
    """count=0 时显式 ASCEND_DEVICE_ID 仍排第一，不被 env/smi 超越。"""
    calls = []
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch(calls, ok_name="npu:3", count=0))
    monkeypatch.setenv("ASCEND_DEVICE_ID", "3")
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", "5")
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(pipeline, "_smi_card_ids", lambda: [7])
    assert pipeline._pick_npu_device() == "npu:3"
    assert calls == ["npu:3"]

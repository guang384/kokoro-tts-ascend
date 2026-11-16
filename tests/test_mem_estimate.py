"""每 worker NPU 内存估算（pipeline.estimate_worker_mem）纯函数契约。"""
import pipeline


def test_estimate_worker_mem_with_budget():
    d = pipeline.estimate_worker_mem(100.0, 9500.0, 65536.0, 9543.0, 450.0)
    assert d["allocated_mb"] == 100.0
    assert d["reserved_mb"] == 9500.0
    assert d["context_mb"] == 450.0
    assert d["estimated_mb"] == 9950.0
    assert d["planned_mb"] == 9802.1          # 0.98×9543 + 450（预占池）
    assert d["limit_mb"] == 10470.1           # 1.05×9543 + 450（进程上限）
    assert d["budget_pct"] == 99.5            # reserved / budget
    assert d["card_pct"] == 15.2              # estimated / 65536
    assert d["card_total_mb"] == 65536.0


def test_estimate_worker_mem_without_budget():
    """预算未成功设置（budget_error 路径）：只报实际估算，不伪造预算字段。"""
    d = pipeline.estimate_worker_mem(0.0, 0.0, 65536.0, None, 153.0)
    assert d["estimated_mb"] == 153.0
    assert "budget_mb" not in d and "planned_mb" not in d
    assert d["card_pct"] == 0.2


def test_estimate_worker_mem_zero_total_safe():
    """卡容量探测为 0（驱动异常）：card_pct 为 None，不除零。"""
    d = pipeline.estimate_worker_mem(0.0, 0.0, 0.0, 100.0, 153.0)
    assert d["card_pct"] is None
    assert d["budget_pct"] == 0.0


def test_estimate_worker_mem_with_actual_process_tts():
    """npu-smi 实测 tts：actual 9683 vs planned 9802.1 → delta -119.1。"""
    d = pipeline.estimate_worker_mem(100.0, 9500.0, 65536.0, 9543.0, 450.0,
                                     process_mb=9683.0)
    assert d["process_mb"] == 9683.0
    assert d["delta_mb"] == -119.1
    assert d["context_actual_mb"] == 183.0     # 9683 - 9500


def test_estimate_worker_mem_with_actual_process_asr():
    """npu-smi 实测 asr：ctx 450；actual 2196 vs planned 2231.6 → delta -35.6。"""
    d = pipeline.estimate_worker_mem(1600.0, 1780.0, 65536.0, 1818.0, 450.0,
                                     process_mb=2196.0)
    assert d["process_mb"] == 2196.0
    assert d["planned_mb"] == 2231.6
    assert d["limit_mb"] == 2358.9
    assert d["delta_mb"] == -35.6
    assert d["context_actual_mb"] == 416.0     # 2196 - 1780


def test_estimate_worker_mem_process_without_budget():
    """预算缺失：仍报 actual/context_actual，但不伪造 planned/delta。"""
    d = pipeline.estimate_worker_mem(0.0, 0.0, 65536.0, None, 450.0,
                                     process_mb=2196.0)
    assert d["process_mb"] == 2196.0
    assert d["context_actual_mb"] == 2196.0
    assert "planned_mb" not in d and "delta_mb" not in d


def test_estimate_worker_mem_without_process_omits_actual_fields():
    """npu-smi 不可用（进程表截断/缺二进制）：不产生 process/delta 键。"""
    d = pipeline.estimate_worker_mem(100.0, 9500.0, 65536.0, 9543.0, 450.0)
    assert "process_mb" not in d
    assert "delta_mb" not in d
    assert "context_actual_mb" not in d

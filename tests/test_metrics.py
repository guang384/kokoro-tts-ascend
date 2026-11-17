"""/metrics（Prometheus）契约：文本格式、指标族齐全、跨网关加权聚合、
label 转义、METRICS_ENABLED 开关。直接驱动 handler。"""
import asyncio
import bisect
import json
import os

import pytest

import gateway
import gateway_state
import gateway_obs
import proc_common


def _write_heartbeat(health_dir: str, wtag: str, snapshot: dict) -> None:
    path = os.path.join(health_dir, f"worker-{wtag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"wtag": wtag, **snapshot}, f, ensure_ascii=False)


@pytest.fixture()
def gw(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_state, "_health_dir", str(tmp_path))
    monkeypatch.setattr(gateway_state, "_result_q", None)
    # 固定档位，避免宿主 WAIT_TIMEOUT_S 污染文本断言
    monkeypatch.setattr(gateway_state, "_LATENCY_BUCKETS",
                        (1, 2, 5, 10, 15, 20, 30, 45, 60, 120))
    gateway_state._recent_latencies.clear()
    gateway_state._recent_failures.clear()
    return gateway


def _scrape() -> str:
    resp = asyncio.run(gateway.metrics())
    return resp.body.decode("utf-8") if isinstance(resp.body, bytes) else resp.body


def _full_fixture(gw):
    """一个网关 + tts + supervisor 心跳（覆盖主要指标族）。"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True,
        "expected": {"gateway": 1, "tts": 1},
        "stats": {"crashes": {"tts-0": 2}},
        "mailboxes": {"g2p_queue": {"depth": 3, "in": 100, "out": 97,
                                    "avg_wait_ms": 1.5}}})
    _write_heartbeat(gateway_state._health_dir, "gw-1", {
        "role": "gateway", "device": None, "ready": True, "warm": {},
        "stats": {"requests": 50, "results": 47, "timeouts": 1, "rejections": 2,
                  "fast_reject": 3,
                  "latency_bounds": [float(b) for b in gateway_state._LATENCY_BUCKETS],
                  "latency_bins": [40, 3, 5, 2, 1, 1, 1, 1, 1, 1, 1],
                  "latency_ms_sum": 9000.0, "recent_count": 10,
                  "p50_s": 0.8, "p90_s": 2.5, "p99_s": 4.9,
                  "recent_ms_sum": 20000.0, "recent_chars_sum": 500.0,
                  "recent_failures": []}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "stuck_s": 0.0,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 20, "forward_ms": 4000.0, "audio_s": 100.0,
                  "realtime_x": 25.0},
        "npu_mem": {"reserved_mb": 5340.0, "budget_pct": 102.1}})



def test_metrics_format_and_families(gw, monkeypatch):
    """Prometheus 文本格式：TYPE/HELP 行 + 核心指标族齐全 + 值正确。"""
    _full_fixture(gw)
    # SHM 在 Windows 开发机无 /dev/shm → patch 出可测值
    monkeypatch.setattr(gateway_obs, "_shm_stats",
                        lambda: {"total_mb": 2048.0, "used_mb": 88.0, "pct": 4.3})
    text = _scrape()
    for token in ("# TYPE kokoro_e2e_requests_total counter",
                  "# TYPE kokoro_e2e_fast_reject_total counter",
                  "# TYPE kokoro_queue_depth gauge",
                  "# TYPE kokoro_queue_bad_total counter",
                  "# TYPE kokoro_worker_done_total counter",
                  "# TYPE kokoro_worker_crashes_total counter",
                  "# TYPE kokoro_shm_pct gauge",
                  "# TYPE kokoro_degraded gauge",
                  "# TYPE kokoro_build_info gauge",
                  "# TYPE kokoro_request_duration_seconds histogram"):
        assert token in text, f"缺 {token}"
    # e2e 绝对值（单网关）
    assert 'kokoro_e2e_requests_total ' in text and " 50.0" in text
    assert "kokoro_e2e_fast_reject_total 3.0" in text
    # 直方图桶必须单调累积（Prometheus le 语义；曾为区间计数+错位标签，
    # histogram_quantile 会算错）。fixture bins=[40,3,5,2,1,1,1,1,1,1,1]
    assert 'kokoro_request_duration_seconds_bucket{le="1"} 40.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="2"} 43.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="5"} 48.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="10"} 50.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="15"} 51.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="20"} 52.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="30"} 53.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="45"} 54.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="60"} 55.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="120"} 56.0' in text
    assert 'kokoro_request_duration_seconds_bucket{le="+Inf"} 57.0' in text
    assert "kokoro_request_duration_seconds_count 57.0" in text
    assert "kokoro_request_duration_seconds_sum 9.0" in text
    # p50/p90/p99 直读 Gauge（fixture gw-1 心跳带 p50_s/p90_s/p99_s）
    assert 'kokoro_request_duration_p50_seconds 0.8' in text
    assert 'kokoro_request_duration_p90_seconds 2.5' in text
    assert 'kokoro_request_duration_p99_seconds 4.9' in text
    # 队列 label
    assert 'kokoro_queue_depth{stage="g2p"} 3.0' in text
    # 坏 member 计数（载荷损坏/外部误写信号；无则 0）
    assert 'kokoro_queue_bad_total{stage="g2p"} 0.0' in text
    # worker label + 崩溃计数（来自 supervisor crashes 映射）
    assert 'kokoro_worker_crashes_total{worker="tts-0"} 2.0' in text
    # realtime_x / npu 预算
    assert 'kokoro_worker_realtime_x{worker="tts-0"} 25.0' in text
    assert 'kokoro_worker_npu_budget_pct{worker="tts-0"} 102.1' in text
    # 旧式 heartbeat 无 process_mb：不得伪造 0 值指标
    assert "kokoro_worker_npu_mem_process_mb{" not in text



def test_stuck_metric_carries_worker_and_role_labels(gw):
    """stuck_seconds 声明了 {worker,role} 两个 label（README 契约）；
    旧实现只传 worker，scrape 时该样本缺 role（查询侧按 role 过滤会漏）。"""
    _full_fixture(gw)
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "stuck_s": 12.5,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 20, "forward_ms": 4000.0, "audio_s": 100.0},
        "npu_mem": {"reserved_mb": 5340.0}})
    text = _scrape()
    line = next(ln for ln in text.splitlines()
                if ln.startswith("kokoro_worker_stuck_seconds{"))
    assert 'worker="tts-0"' in line and 'role="tts"' in line
    assert line.endswith(" 12.5")


def test_worker_npu_mem_metrics_for_both_pools(gw):
    """tts/asr 心跳里的 npu_mem 都要变成 per-worker 指标：
    reserved/budget_pct 兼容旧族，allocated/estimated/planned/budget/
    context/card_pct 为新增明细。"""
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "ready": True, "warm": {}, "stats": {},
        "npu_mem": {"allocated_mb": 9000.0, "reserved_mb": 9500.0,
                    "estimated_mb": 9950.0, "context_mb": 450.0,
                    "budget_mb": 9543.0, "planned_mb": 9802.1,
                    "limit_mb": 10470.1,
                    "budget_pct": 99.5, "card_pct": 15.2,
                    "process_mb": 9683.0, "delta_mb": -119.1,
                    "context_actual_mb": 183.0,
                    "card_total_mb": 65536.0}})
    _write_heartbeat(gateway_state._health_dir, "asr-0", {
        "role": "asr", "ready": True, "warm": {}, "stats": {},
        "npu_mem": {"allocated_mb": 1700.0, "reserved_mb": 1780.0,
                    "estimated_mb": 2230.0, "context_mb": 450.0,
                    "budget_mb": 1818.0, "planned_mb": 2231.6,
                    "limit_mb": 2358.9,
                    "budget_pct": 97.9, "card_pct": 3.4,
                    "process_mb": 2196.0, "delta_mb": -35.6,
                    "context_actual_mb": 416.0,
                    "card_total_mb": 65536.0}})
    text = _scrape()

    def _has(metric, worker, role, value):
        return any(line.startswith(metric + "{")
                   and f'worker="{worker}"' in line
                   and f'role="{role}"' in line
                   and line.endswith(f" {value}")
                   for line in text.splitlines())

    assert _has("kokoro_worker_npu_mem_estimated_mb", "tts-0", "tts", "9950.0")
    assert _has("kokoro_worker_npu_mem_estimated_mb", "asr-0", "asr", "2230.0")
    assert _has("kokoro_worker_npu_mem_context_mb", "tts-0", "tts", "450.0")
    assert _has("kokoro_worker_npu_mem_context_mb", "asr-0", "asr", "450.0")
    assert _has("kokoro_worker_npu_mem_planned_mb", "asr-0", "asr", "2231.6")
    assert _has("kokoro_worker_npu_mem_limit_mb", "tts-0", "tts", "10470.1")
    assert _has("kokoro_worker_npu_mem_limit_mb", "asr-0", "asr", "2358.9")
    assert _has("kokoro_worker_npu_mem_card_pct", "tts-0", "tts", "15.2")
    assert _has("kokoro_worker_npu_mem_process_mb", "tts-0", "tts", "9683.0")
    assert _has("kokoro_worker_npu_mem_process_mb", "asr-0", "asr", "2196.0")
    assert _has("kokoro_worker_npu_mem_delta_mb", "tts-0", "tts", "-119.1")
    assert _has("kokoro_worker_npu_mem_delta_mb", "asr-0", "asr", "-35.6")
    assert _has("kokoro_worker_npu_mem_context_actual_mb",
                "tts-0", "tts", "183.0")
    assert _has("kokoro_worker_npu_mem_context_actual_mb",
                "asr-0", "asr", "416.0")
    assert ('kokoro_worker_npu_mem_reserved_mb{worker="asr-0"} 1780.0'
            in text)
    assert 'kokoro_worker_npu_budget_pct{worker="asr-0"} 97.9' in text


def test_metrics_aggregates_across_gateways(gw):
    """跨网关聚合：e2e 计数器求和（不随实例翻倍或漏计）。"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {"gateway": 2},
        "mailboxes": {}})
    for tag, req, ok in (("gw-1", 10, 9), ("gw-2", 30, 28)):
        _write_heartbeat(gateway_state._health_dir, tag, {
            "role": "gateway", "device": None, "ready": True, "warm": {},
            "stats": {"requests": req, "results": ok, "timeouts": 0,
                      "rejections": 0, "latency_ms_sum": 0.0}})
    text = _scrape()
    assert "kokoro_e2e_requests_total 40.0" in text
    assert "kokoro_e2e_successes_total 37.0" in text


def test_metrics_label_escaped(gw, monkeypatch):
    """label 值含特殊字符不破坏格式（prometheus_client 负责转义）。

    Windows 文件名不允许引号 → 用合成数据驱动 collector（不落盘）。"""
    real = gateway_obs._collect

    def fake():
        d = real()
        row = {"wtag": 'odd"tag', "role": "g2p", "ready": True, "done": 1}
        d["workers"] = [row]
        d["workers_raw"] = [{"wtag": 'odd"tag', "role": "g2p", "ready": True,
                             "stats": {"requests": 1}}]
        return d

    monkeypatch.setattr(gateway_obs, "_collect", fake)
    text = _scrape()
    # 引号被转义为 \" ，不产生格式错误的裸引号行
    assert 'worker="odd\\"tag"' in text


def test_metrics_disabled_returns_404(gw, monkeypatch):
    """METRICS_ENABLED=0 → 404（开关生效）。"""
    monkeypatch.setattr(gateway_state, "METRICS_ENABLED", False)
    with pytest.raises(Exception) as ei:
        asyncio.run(gateway.metrics())
    assert getattr(ei.value, "status_code", None) == 404


def test_env_bool_parsing(monkeypatch, caplog):
    monkeypatch.setenv("K_B", "true")
    assert proc_common.env_bool("K_B", False) is True
    monkeypatch.setenv("K_B", "OFF")
    assert proc_common.env_bool("K_B", True) is False
    monkeypatch.setenv("K_B", "maybe")
    import logging
    with caplog.at_level(logging.WARNING, logger="kokoro.config"):
        assert proc_common.env_bool("K_B", True) is True   # 坏值回退默认
    assert any("K_B" in r.message for r in caplog.records)


# ── 延迟桶：按 WAIT_TIMEOUT_S 推导 + 旧心跳兼容 ──

def test_latency_buckets_derive_from_timeout():
    """最后一档恒等于总预算；超出预算的固定档被裁掉，不导出空桶。"""
    assert gateway_state._derive_latency_buckets(120.0) == (
        1, 2, 5, 10, 15, 20, 30, 45, 60, 120.0)
    assert gateway_state._derive_latency_buckets(60.0) == (
        1, 2, 5, 10, 15, 20, 30, 45, 60.0)
    assert gateway_state._derive_latency_buckets(30.0) == (
        1, 2, 5, 10, 15, 20, 30.0)
    assert gateway_state._derive_latency_buckets(90.5) == (
        1, 2, 5, 10, 15, 20, 30, 45, 60, 90.5)
    assert gateway_state._derive_latency_buckets(300.0) == (
        1, 2, 5, 10, 15, 20, 30, 45, 60, 120, 180, 240, 300.0)
    assert gateway_state._derive_latency_buckets(1.0) == (1.0,)


def test_record_latency_bucket_edges(gw, monkeypatch):
    """le 语义为 ≤ 上界：边界值归入该桶；超出总预算落 +Inf 尾巴。"""
    monkeypatch.setattr(gateway_state, "_entry_stats", {
        "latency_bins": [0] * (len(gateway_state._LATENCY_BUCKETS) + 1),
        "latency_ms_sum": 0.0})
    vals = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0,
            60.0, 120.0, 120.5]
    for v in vals:
        gateway_obs._record_latency(v, 1)
    assert gateway_state._entry_stats["latency_bins"] == [
        2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert (gateway_state._entry_stats["latency_ms_sum"]
            == pytest.approx(sum(vals) * 1000))


def test_worker_snapshot_exports_bounds(gw, monkeypatch):
    """心跳带新 bins/bounds（档位边界随总预算动态导出，供聚合侧校验）。"""
    bins = [40, 3, 5, 2, 1, 1, 1, 1, 1, 1, 1]
    monkeypatch.setattr(gateway_state, "_entry_stats", {
        "latency_bins": list(bins), "latency_ms_sum": 1.0})
    stats = gateway_obs._worker_snapshot()["stats"]
    assert stats["latency_bounds"] == [float(b) for b in gateway_state._LATENCY_BUCKETS]
    assert stats["latency_bins"] == bins


def _cum_le(d, edge):
    bins = d["latency"]["bins"]
    idx = bisect.bisect_left(gateway_state._LATENCY_BUCKETS, edge) + 1
    return sum(bins[:idx])


def test_metrics_bounds_mismatch_skipped(gw):
    """心跳档位与本地不一致（异常心跳/配置漂移）→ 跳过该实例的桶计数，
    绝不把错误粒度的 bins 直接相加污染直方图。"""
    _write_heartbeat(gateway_state._health_dir, "gw-odd", {
        "role": "gateway", "ready": True, "warm": {},
        "stats": {"latency_bounds": [0.5, 1.0, 2.0, 5.0],
                  "latency_bins": [99, 99, 99, 99, 99],
                  "latency_ms_sum": 0.0}})
    d = gateway_obs._collect()
    assert d["latency"]["count"] == 0     # 跳过，而非相加 495


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_metrics.py")

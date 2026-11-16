"""网关终态/健康/结果泵回归：P1 审计项的离线锁定。

覆盖：
- /tts 对齐失败 _SynthError → 502（原 generic 500）
- /tts 对齐超时计入 timeouts/failed（原不可见）
- 重试块在 TS Future 存在时延迟 unlink；无 TS 时即时回收
- 结果泵 get()/投递异常不杀线程
- server_5xx/last_5xx_ts 只统计 5xx；近期 5xx 使 /health degraded
- /dev/shm ≥90% 时入队前 429 shm_pressure
"""
import asyncio
import json
import os
import time

import numpy as np
import pytest
from fastapi import HTTPException

import gateway
import gateway_state
import gateway_agg
import gateway_obs
import protocol
from gateway import BizError, TTSRequest, _SynthError, tts_business
from tests.fake_queue import FakeQueue


@pytest.fixture()
def gw(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_state, "_pending", {})
    monkeypatch.setattr(gateway_state, "_request_meta", {})
    monkeypatch.setattr(gateway_state, "_agg_state", {})
    monkeypatch.setattr(gateway_state, "_retired_blocks", {})
    monkeypatch.setattr(gateway_state, "_ts_state", {})
    monkeypatch.setattr(gateway_state, "_entry_wait_s", 0.2)
    monkeypatch.setattr(gateway_state, "_entry_stats", {
        "requests": 0, "results": 0, "timeouts": 0, "failed": 0,
        "rejections": 0, "fast_reject": 0, "server_5xx": 0,
        "last_5xx_ts": 0.0, "swept": 0, "cancelled": 0,
        "latency_ms_sum": 0.0})
    gateway_state._recent_failures.clear()
    monkeypatch.setattr(gateway_state, "_health_dir", str(tmp_path))
    monkeypatch.setattr(gateway, "_asr_ready", lambda: True)
    monkeypatch.setattr(gateway, "_check_voice", lambda voice: None)
    return gateway


def _fake_synth(ts_exc=None, ts_done=False):
    """返回 <audio, texts, fw, rid, blocks, timings, agg_s> 的 _synth 替身。

    ts_exc：给 tts_business 预先登记的 TS Future 注入异常；
    ts_done=False 且 ts_exc=None：不推进 Future（模拟对齐段超时）。"""
    async def fake(text, voice, speed, needs_align, req_id=None,
                   release_shm=True, request=None, g2p_overrides=None,
                   dtype=protocol.DTYPE_DEFAULT, deadline=None):
        fut_ts = gateway_state._pending.get(("ts", req_id))
        if ts_exc is not None and fut_ts is not None:
            fut_ts.set_exception(ts_exc)
        return (np.zeros(0, dtype="<i2"), [], 1, req_id, None,
                {"q_g2p": 0.0, "g2p": 0.0, "q_tts": 0.0, "tts": 0.0}, 0.0)
    return fake


def test_tts_alignment_error_maps_502(gw, monkeypatch):
    monkeypatch.setattr(gateway, "_synth", _fake_synth(
        _SynthError(protocol.KIND_COMPUTE, "ASR down")))
    with pytest.raises(BizError) as ei:
        asyncio.run(tts_business(TTSRequest(text="你好", speed=1.0)))
    assert ei.value.status_code == 502
    assert "ASR down" in ei.value.message
    assert gateway_state._entry_stats["server_5xx"] == 1
    assert gateway_state._entry_stats["failed"] == 1
    assert gateway_state._pending == {} and gateway_state._request_meta == {}


def test_tts_alignment_timeout_counted(gw, monkeypatch):
    monkeypatch.setattr(gateway, "_synth", _fake_synth(ts_exc=None))
    with pytest.raises(BizError) as ei:
        asyncio.run(tts_business(TTSRequest(text="你好", speed=1.0)))
    assert ei.value.status_code == 504
    assert gateway_state._entry_stats["timeouts"] == 1
    assert gateway_state._entry_stats["failed"] == 1
    assert gateway_state._recent_failures[-1]["status"] == 504
    assert gateway_state._pending == {} and gateway_state._request_meta == {}


def test_retire_block_defers_unlink_while_ts_pending(gw, monkeypatch):
    calls = []
    monkeypatch.setattr(gateway_agg, "_unlink_blocks", lambda blocks: calls.extend(blocks))
    block = {"shm_name": "x", "segs": []}
    loop = asyncio.new_event_loop()
    try:
        gateway_state._pending[("ts", "r")] = loop.create_future()
        assert gateway_agg._retire_block("r", block) is True
        assert gateway_state._retired_blocks["r"][0][0] is block
        assert calls == []
        # 无 TS Future（/v1 或请求已终态）：即时回收
        assert gateway_agg._retire_block("r2", block) is False
        assert calls == [block]
    finally:
        loop.close()


def test_sweep_leaks_cleans_aged_retired_blocks(gw, monkeypatch):
    calls = []
    monkeypatch.setattr(gateway_agg, "_unlink_blocks", lambda blocks: calls.extend(blocks))
    block = {"shm_name": "x", "segs": []}
    gateway_state._retired_blocks["old"] = [(block, time.time() - 999)]
    gateway_agg._sweep_leaks()
    assert calls == [block] and "old" not in gateway_state._retired_blocks


def test_result_pump_survives_get_exception(gw, monkeypatch):
    class _Loop:
        def __init__(self):
            self.items = []

        def is_closed(self):
            return False

        def call_soon_threadsafe(self, fn, item):
            self.items.append(item)

    class _Q:
        def __init__(self):
            self.n = 0

        def get(self):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("redis blip")
            return {"type": "unknown"}

    loop = _Loop()
    monkeypatch.setattr(gateway_state, "_main_loop", loop)
    q = _Q()
    assert gateway_agg._pump_once(q) is False       # get 异常被吞，泵可继续
    assert gateway_agg._pump_once(q) is True
    assert loop.items == [{"type": "unknown"}]


def test_record_failure_counts_only_5xx(gw):
    gw._record_failure(500, "redis down")
    assert gateway_state._entry_stats["server_5xx"] == 1
    assert gateway_state._entry_stats["last_5xx_ts"] > 0
    gw._record_failure(400, "bad input")
    assert gateway_state._entry_stats["server_5xx"] == 1   # 4xx 不污染健康
    assert gateway_state._entry_stats["failed"] == 2


def test_collect_degrades_on_recent_5xx(gw):
    now = time.time()
    with open(os.path.join(gateway_state._health_dir, "worker-sup.json"), "w",
              encoding="utf-8") as f:
        json.dump({"role": "supervisor", "ready": True,
                   "expected": {"gateway": 1},
                   "stats": {"crashes": {}}, "mailboxes": {}}, f)
    with open(os.path.join(gateway_state._health_dir, "worker-gw-1.json"), "w",
              encoding="utf-8") as f:
        json.dump({"wtag": "gw-1", "role": "gateway", "ready": True,
                   "warm": {}, "stats": {"requests": 1, "results": 0,
                                         "server_5xx": 1,
                                         "last_5xx_ts": now}}, f)
    d = gateway_obs._collect()
    assert d["ready"] is True
    assert d["status"] == "degraded"
    assert d["e2e"]["server_5xx"] == 1


def test_shm_pressure_rejects_before_enqueue(gw, monkeypatch):
    q = FakeQueue()
    gateway_state._g2p_queue = q
    monkeypatch.setattr(gateway_obs, "_shm_used_pct", lambda: 95.0)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(gw._synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 429
    assert ei.value.headers["X-Reject-Reason"] == "shm_pressure"
    assert q.qsize() == 0
    assert gateway_state._entry_stats["rejections"] == 1
    assert gateway_state._pending == {} and gateway_state._request_meta == {}


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_gateway_resilience.py")

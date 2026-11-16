"""背压与请求生命周期回归：受理闸门 429、超时 504、_pending/_request_meta
无泄漏、错误分级（compute→502 / invalid→400 / busy→429）。
全部驱动真实 gateway._synth 与 _handle_result（不起 HTTP 服务）。"""
import asyncio

import pytest
from fastapi import HTTPException

import gateway
import gateway_state
import gateway_agg
import protocol
from gateway import _synth
from tests.fake_queue import FakeQueue


@pytest.fixture()
def gw(monkeypatch):
    monkeypatch.setattr(gateway_state, "_pending", {})
    monkeypatch.setattr(gateway_state, "_request_meta", {})
    monkeypatch.setattr(gateway_state, "_agg_state", {})
    monkeypatch.setattr(gateway_state, "_entry_wait_s", 0.2)  # 超时测试不真等 120s
    monkeypatch.setattr(gateway_state, "_entry_stats",
                        {"requests": 0, "results": 0, "timeouts": 0,
                         "failed": 0, "rejections": 0, "fast_reject": 0,
                         "swept": 0, "cancelled": 0,
                         "latency_bins": [0] * (len(gateway_state._LATENCY_BUCKETS) + 1),
                         "latency_ms_sum": 0.0})
    return gateway


def _seed_pressure(name, avg_ms, depth=1):
    """fakeredis 预置认领队列压力（QUEUE_INDEX 固定序号 → kokoro:q:{i}:{name}）。"""
    import proc_common as pc

    c = pc.shared_redis()
    idx = pc.QUEUE_INDEX[name]
    if depth:
        c.zadd(f"kokoro:q:{idx}:{name}", {"dummy-member": 0})  # 成员内容无关：只为 ZCARD>0
    if avg_ms:
        c.set(f"kokoro:q:{idx}:{name}:avg", avg_ms)


def test_gate1_full_g2p_queue_rejects_429_without_leak(gw):
    """g2p_queue 满 → 立即 429；_pending/_request_meta 不泄漏（历史泄漏点）。"""
    q = FakeQueue(maxsize=1)
    q.put_nowait({"stale": True})
    gateway_state._g2p_queue = q
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 429
    assert ei.value.headers["Retry-After"] == "1"
    assert ei.value.headers["X-Reject-Reason"] == "queue_full"
    assert gateway_state._pending == {} and gateway_state._request_meta == {}
    assert gateway_state._entry_stats["rejections"] == 1


def test_timeout_504_without_leak(gw):
    """下游无终态 → 504；Future/聚合登记被清理，不留悬挂条目。"""
    gateway_state._g2p_queue = FakeQueue()  # 永无结果
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 504
    assert gateway_state._pending == {} and gateway_state._request_meta == {}
    assert gateway_state._entry_stats["timeouts"] == 1


def test_compute_error_maps_502(gw):
    """compute 前向失败（kind=compute）→ 502，而非历史的 400。"""
    gateway_state._g2p_queue = FakeQueue()

    async def _reply():
        await asyncio.sleep(0.02)
        rid = next(iter(gateway_state._pending))[1]
        gateway_agg._handle_result(protocol.synth_error(rid, "NPU OOM", protocol.KIND_COMPUTE))

    async def _call():
        task = asyncio.create_task(_reply())
        try:
            await _synth("你好", "zf_001", 1.0, needs_align=False)
        finally:
            await task

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call())
    assert ei.value.status_code == 502 and "NPU OOM" in ei.value.detail


def test_invalid_input_maps_400(gw):
    gateway_state._g2p_queue = FakeQueue()

    async def _reply():
        await asyncio.sleep(0.02)
        rid = next(iter(gateway_state._pending))[1]
        gateway_agg._handle_result(protocol.synth_error(rid, "文本不含可发音内容",
                                               protocol.KIND_INVALID))

    async def _call():
        task = asyncio.create_task(_reply())
        try:
            await _synth("。。。", "zf_001", 1.0, needs_align=False)
        finally:
            await task

    with pytest.raises(HTTPException) as ei:
        asyncio.run(_call())
    assert ei.value.status_code == 400


# ── 客户端断开取消（fix8）：断连 → 摘状态 + 逐级 CANCEL，不再浪费下游算力 ──

class _GoneRequest:
    """is_disconnected 恒 True 的请求替身。"""

    async def is_disconnected(self):
        return True


def test_client_disconnect_cancels_request(gw, fakeredis_client):
    """断连 → 499、状态零泄漏、取消标记进 Redis SET（g2p/worker 认领侧
    检查跳过）、stats 计数。"""
    gateway_state._g2p_queue = FakeQueue()

    async def scenario():
        return await _synth("你好世界", "zf_001", 1.0, needs_align=False,
                            request=_GoneRequest())

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(scenario())
    assert ei.value.status_code == 499
    assert gateway_state._pending == {} and gateway_state._request_meta == {}   # 状态已摘除
    assert len(fakeredis_client.zrange("kokoro:cancel", 0, -1)) == 1  # 断连标记入 ZSET
    assert gateway_state._entry_stats["cancelled"] == 1


def test_no_watch_without_request(gw):
    """不传 request（内部调用/旧用法）→ 无 watcher：下游无终态仍走 504，
    不产生 cancelled 计数。"""
    gateway_state._g2p_queue = FakeQueue()
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 504
    assert gateway_state._entry_stats.get("cancelled", 0) == 0


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_backpressure.py")


def test_late_synth_error_keeps_ts_future(gw):
    """审计 F2 回归：音频已交付时迟到 synth_error（批重派的第二次前向失败）
    不碰 fut_ts——TS 在途由账本推进、请求应成功（曾被 f6266b2 误炸成 502）。"""
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        fut = loop.create_future()
        fut.set_result("done")               # 音频段已完成（/tts 常态）
        fut_ts = loop.create_future()
        gateway_state._pending[("req", "r-late")] = fut
        gateway_state._pending[("ts", "r-late")] = fut_ts
        gateway_agg._on_synth_error({"req_id": "r-late", "kind": "compute",
                            "error": "前向失败"})
        assert ("ts", "r-late") in gateway_state._pending   # fut_ts 保留，等 TS 账本推进
        assert not fut_ts.done()                 # 不被迟到错误翻案
    finally:
        loop.close()
        gateway_state._pending.clear()

def test_body_size_limit_413(gw):
    """超大请求体在 Pydantic 解析前被 413 拒绝（content-length 预检），
    不占内存不在事件循环解析。"""
    from fastapi.testclient import TestClient

    client = TestClient(gw.app)
    # 中间件以请求头 content-length 预检——直接驱动中间件层（无需真发大 body）
    gateway_state._MAX_BODY_BYTES = 1024
    try:
        r = client.post("/tts", content=b"{" + b"y" * 4096 + b"}",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 413
    finally:
        gateway_state._MAX_BODY_BYTES = 4 * 2**20


def test_middleware_stack_pure_asgi(gw):
    """中间件栈不得含 BaseHTTPMiddleware：它会包裹 ASGI receive 通道，
    request.is_disconnected() 从此恒 False——断连取消链路（499/CANCEL/
    即时回收）整体静默失效（E2E 实测发现的生产事故）。请求体上限必须
    以纯 ASGI 中间件实现（_BodySizeLimitMiddleware）。"""
    from starlette.middleware.base import BaseHTTPMiddleware

    for mw in gw.app.user_middleware:
        assert not (isinstance(mw.cls, type)
                    and issubclass(mw.cls, BaseHTTPMiddleware)), \
            f"{mw.cls.__name__} 是 BaseHTTPMiddleware——会杀死断连检测"


def test_cancel_end_to_end_claim_filter(fakeredis_client, monkeypatch):
    """深度审计 F2 回归：断连标记（ZADD/ZSCORE）→ tts 认领侧过滤。
    曾 gateway 写 ZSET、compute 读 SISMEMBER → 首次断连后全池 WRONGTYPE。"""
    
    import proc_common
    import redis_transport

    # 1) gateway 断连路径写标记（真实模块函数）
    proc_common.cancel_add("req-gone")

    # 2) 认领侧读取（真实 TtsClaimQueue.cancelled_set → 批量 ZSCORE）
    monkeypatch.setattr(redis_transport, "_make_redis_client",
                        lambda: fakeredis_client)
    q = proc_common.TtsClaimQueue(16, index=46, name="cancel-e2e")
    assert q.cancelled_set(["req-gone", "req-alive"]) == {"req-gone"}   # WRONGTYPE 在此暴露


# ── 准入快速拒绝（fast_reject：路径求和预估 + 15% 冗余）──
# fixture _entry_wait_s=0.2 → budget=200ms、threshold=200×1.15=230ms

def test_fast_reject_429_when_tts_ema_exceeds_threshold(gw, caplog):
    """tts EMA(300ms) > threshold(230ms) 且 depth>0 → 秒拒 429：头齐全、
    不受理不登记、日志带完整估算过程（est/threshold/各段 d 与 w）。"""
    import logging

    gateway_state._g2p_queue = FakeQueue()
    _seed_pressure("tts", 300.0)
    with caplog.at_level(logging.INFO, logger="kokoro.server"):
        with pytest.raises(HTTPException) as ei:
            asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 429
    assert ei.value.headers["Retry-After"] == "1"
    assert ei.value.headers["X-Reject-Reason"] == "wait_budget"
    assert gateway_state._entry_stats["fast_reject"] == 1
    assert gateway_state._entry_stats["requests"] == 0
    assert gateway_state._pending == {} and gateway_state._request_meta == {}
    line = next(r.getMessage() for r in caplog.records
                if "fast reject" in r.getMessage())
    assert "est=0.30s" in line and "threshold=0.23s" in line
    assert "(budget=0.2s×1.15)" in line
    assert "tts(d=1,w=0.30s)" in line
    assert "asr(" not in line        # /v1 路径不查 asr 段


def test_fast_reject_margin_admits_borderline(gw):
    """冗余系数放行边界：est=210ms ∈ (budget 200, threshold 230] → 不拒，
    正常受理（无终态 → 504），fast_reject 不计数。"""
    gateway_state._g2p_queue = FakeQueue()
    _seed_pressure("tts", 210.0)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 504
    assert gateway_state._entry_stats["fast_reject"] == 0
    assert gateway_state._entry_stats["timeouts"] == 1


def test_fast_reject_sums_align_path_only(gw):
    """求和语义：tts=120ms + asr=120ms → /tts est=240ms > 230 拒；
    /v1 不查 asr → est=120ms 放行（走 504）。"""
    gateway_state._g2p_queue = FakeQueue()
    _seed_pressure("tts", 120.0)
    _seed_pressure("asr", 120.0)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=True))
    assert ei.value.status_code == 429
    assert gateway_state._entry_stats["fast_reject"] == 1

    with pytest.raises(HTTPException) as ei2:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei2.value.status_code == 504
    assert gateway_state._entry_stats["fast_reject"] == 1   # 未再增长


def test_fast_reject_ignores_stale_ema_when_queue_empty(gw):
    """EMA 陈旧值不采信：tts :avg 很高但 depth=0 → 计 0 → 放行（走 504）。"""
    gateway_state._g2p_queue = FakeQueue()
    _seed_pressure("tts", 9999.0, depth=0)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 504
    assert gateway_state._entry_stats["fast_reject"] == 0


def test_timeout_cancels_downstream_chunks(gw):
    """超时即取消：504 同时下发 CANCEL（与断连同路），在途 chunk 认领后
    前向前被丢弃，死请求不再浪费前向。"""
    import proc_common as pc

    gateway_state._g2p_queue = FakeQueue()  # 永无结果
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False,
                           req_id="rt1"))
    assert ei.value.status_code == 504
    assert pc.shared_redis().zscore("kokoro:cancel", "rt1") is not None
    assert gateway_state._entry_stats["timeouts"] == 1


def test_timeout_survives_cancel_add_failure(gw, monkeypatch):
    """Redis 抖动时取消标记只是尽力而为：504 不能被改写成 500。"""
    gateway_state._g2p_queue = FakeQueue()

    def _boom(_rid):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(gateway.proc_common, "cancel_add", _boom)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False))
    assert ei.value.status_code == 504
    assert gateway_state._entry_stats["timeouts"] == 1


def test_disconnect_survives_cancel_add_failure(gw, monkeypatch):
    """断连 watcher 内 cancel_add 抛错不能杀死任务：仍要 499 并 resolve fut。"""
    gateway_state._g2p_queue = FakeQueue()

    def _boom(_rid):
        raise RuntimeError("redis unavailable")

    monkeypatch.setattr(gateway.proc_common, "cancel_add", _boom)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(_synth("你好", "zf_001", 1.0, needs_align=False,
                           request=_GoneRequest()))
    assert ei.value.status_code == 499
    assert gateway_state._entry_stats["cancelled"] == 1


def test_curl_without_content_type_parses(gw, monkeypatch):
    """curl --data-raw 不带 Content-Type（默认 x-www-form-urlencoded）
    → 中间件归一为 application/json，JSON 体正常进业务层。历史行为：
    FastAPI 按表单解析 → 422。"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(gateway_state, "_asr_ready_cache",
                        {"ts": 0.0, "val": False})   # 强制就绪闸重扫 → 503
    client = TestClient(gw.app)
    r = client.post(
        "/tts",
        content='{"text":"你好，欢迎使用昇腾算力平台的语音合成服务。"}'.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 503, r.text       # 体已按 JSON 解析：走到就绪闸
    body = r.json()
    assert "error" in body and "时间戳服务" in body["error"]


def test_missing_content_type_also_normalized(gw, monkeypatch):
    """缺失 Content-Type 同样归一（部分 HTTP 客户端不发该头）。"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(gateway_state, "_asr_ready_cache",
                        {"ts": 0.0, "val": False})
    client = TestClient(gw.app)
    r = client.post("/tts", content='{"text":"你好"}'.encode("utf-8"))
    assert r.status_code == 503, r.text
    assert "error" in r.json()

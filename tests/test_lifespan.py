"""ASGI lifespan 启动/关闭回归。

历史事故（057ccb5）：gateway 拆分时丢失 @asynccontextmanager，FastAPI 在
生产启动即失败，而当时没有任何测试真正进入 lifespan——TestClient 不进入
上下文管理器时不会触发 lifespan。本文件锁死这一层：

  - _lifespan 必须是 async context manager（丢装饰器立即失败，而非等生产）；
  - 启动：事件循环登记、心跳任务、结果泵线程、音色预热；
  - 关闭：心跳任务被取消并落地，不留下 pending task。
"""
import asyncio
import threading

from fastapi.testclient import TestClient

import gateway
import gateway_agg
import gateway_obs
import gateway_state as gst
import pipeline


def test_lifespan_is_async_context_manager():
    """FastAPI 只接受 async context manager；丢装饰器是生产启动级故障。

    @asynccontextmanager 装饰后 _lifespan 是工厂函数而非 CM 本身，
    因此检查"调用结果"是否具备 __aenter__/__aexit__。"""
    cm = gateway._lifespan(gateway.app)
    assert hasattr(cm, "__aenter__") and hasattr(cm, "__aexit__")
    del cm  # 未进入的 async generator 无副作用，交给 GC


def test_asgi_lifespan_startup_and_shutdown(monkeypatch, tmp_path):
    """真实走一遍 ASGI lifespan（TestClient 上下文管理器），不加载模型。"""
    monkeypatch.setattr(gst, "_health_dir", str(tmp_path))
    monkeypatch.setattr(gst, "_main_loop", None)
    monkeypatch.setattr(gst, "_heartbeat_task", None)
    monkeypatch.setattr(gst, "_voices_cache", None)
    monkeypatch.setattr(gst, "_result_q", None)

    events = {"heartbeat": threading.Event(),
              "heartbeat_cancelled": threading.Event(),
              "pump": threading.Event()}

    async def fake_heartbeat():
        events["heartbeat"].set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            events["heartbeat_cancelled"].set()
            raise

    def fake_pump(_result_q):
        events["pump"].set()

    monkeypatch.setattr(gateway_obs, "_heartbeat_loop", fake_heartbeat)
    monkeypatch.setattr(gateway_agg, "_result_pump", fake_pump)
    monkeypatch.setattr(pipeline, "list_voices", lambda: ["zf_001", "zf_002"])

    with TestClient(gateway.app) as client:
        # 结果泵是独立线程；心跳任务先 create_task，请求一次让它跑到首 await
        assert events["pump"].wait(2.0), "结果泵线程未启动"
        assert client.get("/v1/models").status_code == 200
        assert events["heartbeat"].wait(2.0), "心跳任务未启动"
        assert gst._main_loop is not None and not gst._main_loop.is_closed()
        assert gst._voices_cache == {"zf_001", "zf_002"}
        assert client.get("/v1/voices").json() == {
            "voices": ["zf_001", "zf_002"]}

    # shutdown 必须等取消落地（只 cancel 不 await 会留 pending task）
    assert events["heartbeat_cancelled"].wait(2.0), "心跳任务未收到取消"
    assert gst._heartbeat_task is not None and gst._heartbeat_task.done()

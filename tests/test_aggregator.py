"""gateway 结果聚合/终态分发：synth_part 聚合、busy→429、错误分级、
align_done 修复路径、孤儿清理与 SHM 生命周期。全部用真实 SharedMemory
小对象驱动 gateway 的 _handle_result 分支（不起 HTTP 服务）。"""
import asyncio
import time
from multiprocessing import shared_memory

import numpy as np
import pytest
import audio_enc
import gateway_state

import protocol
from gateway import _SynthError
import gateway_agg
from gateway_agg import _agg_add, _on_synth_error, _on_synth_part
from tests.fake_queue import FakeQueue


@pytest.fixture()
def gw(monkeypatch):
    """隔离 gateway 全局状态（聚合/TS 账本均为模块级 dict）。"""
    import gateway

    monkeypatch.setattr(gateway_state, "_pending", {})
    monkeypatch.setattr(gateway_state, "_request_meta", {})
    monkeypatch.setattr(gateway_state, "_agg_state", {})
    monkeypatch.setattr(gateway_state, "_retired_blocks", {})
    monkeypatch.setattr(gateway_state, "_entry_stats",
                        {"requests": 0, "results": 0, "timeouts": 0,
                         "failed": 0, "rejections": 0, "swept": 0,
                         "cancelled": 0, "latency_ms_sum": 0.0})
    return gateway


def _loop():
    return asyncio.new_event_loop()


def _block(segs_spec):
    """建一个真实 SHM 块：segs_spec = [(idx, text), ...]，每段 8 采样。

    线格式必须是 PCM16（与 compute.ship_batch / gateway._gather_audio 一致）；
    旧夹具用 float32，测试永不经过 _gather_audio 时发现不了格式漂移。"""
    arrays = [np.arange(8, dtype=np.int16) + idx * 10 for idx, _t in segs_spec]
    total = sum(a.nbytes for a in arrays)
    shm = shared_memory.SharedMemory(create=True, size=total)
    segs, off = [], 0
    for (idx, text), a in zip(segs_spec, arrays):
        shm.buf[off:off + a.nbytes] = a.tobytes()
        segs.append({"idx": idx, "offset": off, "samples": a.shape[0],
                     "text": text})
        off += a.nbytes
    return {"shm_name": shm.name, "segs": segs}, shm, arrays


def _part(rid, need, block):
    return protocol.synth_part(rid, need, block)


def _shm_drop(shm) -> None:
    """测试侧 SHM 清理（幂等 unlink）：Linux 下块名可能已被 gateway 的
    终态清理移除——Windows 双 unlink 无害，Linux 再 unlink 会
    FileNotFoundError（arm64 容器内验证时暴露的可移植性问题）。"""
    shm.close()
    try:
        shm.unlink()
    except FileNotFoundError:
        pass


# ── 纯 TTS（needs_align=false）──

def test_synth_part_aggregates_across_batches(gw):
    """多批请求：两个 synth_part（各 3 句）→ 收齐 need=6 才交付（历史根因缺陷）。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r1"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b1, shm1, a1 = _block([(0, "一"), (1, "二"), (2, "三")])
    b2, shm2, a2 = _block([(3, "四"), (4, "五"), (5, "六")])
    try:
        _on_synth_part(_part(rid, 6, b1))
        assert not fut.done()          # 未收齐不交付
        assert rid in gateway_state._agg_state
        _on_synth_part(_part(rid, 6, b2))
        assert fut.done() and fut.exception() is None
        r = fut.result()
        assert len(r["blocks"]) == 2 and r["fw"] == 2
        assert r["texts"] == ["一", "二", "三", "四", "五", "六"]
        assert rid not in gateway_state._agg_state and rid not in gateway_state._request_meta
    finally:
        shm1.close(); shm1.unlink(); shm2.close(); shm2.unlink()


def test_synth_part_single_batch_delivers_audio(gw):
    loop = _loop()
    fut = loop.create_future()
    rid = "r2"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, arrays = _block([(0, "你好")])
    try:
        _on_synth_part(_part(rid, 1, b))
        r = fut.result()
        assert r["fw"] == 1
        # 从块描述符重建视图应得到写入的 PCM16 字节（协议一致性）
        shm2 = shared_memory.SharedMemory(name=b["shm_name"])
        view = np.frombuffer(shm2.buf[0:arrays[0].nbytes], dtype="<i2")
        assert np.array_equal(view, arrays[0])
        del view  # 释放导出指针再 close（mmap 语义）
        shm2.close()
    finally:
        _shm_drop(shm)


def test_orphan_part_cleaned_without_state(gw):
    """请求已终止（超时/取消）后迟到的 part：不建聚合状态、不崩溃。"""
    block, shm, _a = _block([(0, "一")])
    try:
        _on_synth_part(_part("ghost", 2, block))
        assert gateway_state._agg_state == {}
    finally:
        _shm_drop(shm)


def test_concat_failure_not_masked_by_buffererror(gw, monkeypatch):
    """聚合完成后的拼装阶段抛错：必须透传根因——异常分支要先释放 SHM
    视图再走清理，否则 finally 的 close() 因导出视图抛 BufferError 顶掉
    真正的故障原因（历史：MemoryError/坏描述符全被报成 BufferError）。"""
    import gateway as gateway_mod

    gateway_state._g2p_queue = FakeQueue()

    def _boom(_parts):
        raise ValueError("concat boom")

    monkeypatch.setattr(audio_enc, "_concat", _boom)
    b, shm, _a = _block([(0, "一")])
    try:
        async def scenario():
            async def reply():
                await asyncio.sleep(0.02)  # 等 _synth 注册 Future 并入队
                rid = next(iter(gateway_state._pending))[1]
                gateway_agg._handle_result(_part(rid, 1, b))

            return await asyncio.gather(
                gateway_mod._synth("文本", "zf_001", 1.0, needs_align=False),
                reply())

        with pytest.raises(ValueError, match="concat boom"):
            asyncio.run(scenario())
    finally:
        shm.close()
        try:
            shm.unlink()
        except Exception:  # noqa: BLE001  gateway 已 unlink 时幂等
            pass
    assert gateway_state._agg_state == {}


# ── 失败终态与 busy ──

def test_synth_error_kind_mapping(gw):
    """错误分级：invalid_input/compute 两种 kind 都原样带给端点映射。"""
    loop = _loop()
    fut = loop.create_future()
    gateway_state._pending[("req", "r7")] = fut
    _on_synth_error(protocol.synth_error("r7", "不含可发音内容",
                                         protocol.KIND_INVALID))
    assert isinstance(fut.exception(), _SynthError)
    assert fut.exception().kind == protocol.KIND_INVALID

    fut2 = loop.create_future()
    gateway_state._pending[("req", "r8")] = fut2
    _on_synth_error(protocol.synth_error("r8", "前向失败", protocol.KIND_COMPUTE))
    assert fut2.exception().kind == protocol.KIND_COMPUTE


def test_duplicate_terminal_message_is_orphan_safe(gw):
    """重复终态（历史 C4/C6 的衍生）：第二遍不崩溃、不重复交付。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r10"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一")])
    try:
        _on_synth_part(_part(rid, 1, b))
        _on_synth_part(_part(rid, 1, b))  # 迟到的重复：孤儿路径，幂等
        assert fut.result()["fw"] == 1
        assert gateway_state._agg_state == {}
    finally:
        _shm_drop(shm)


def test_shm_view_release_before_close():
    """回归：SHM 视图（np.frombuffer）全部释放后 close 不抛 BufferError。"""
    shm = shared_memory.SharedMemory(create=True, size=64)
    pieces = [np.frombuffer(shm.buf[0:32], dtype=np.float32)]
    del pieces  # 视图消亡 → 导出指针归零
    _shm_drop(shm)


def test_agg_add_clamps_bad_need(gw):
    """need 下限保护：need=0/缺失按 ≥1 处理；给 1 段即完成（不悬挂）。"""
    gateway_state._request_meta["r11"] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一")])
    try:
        st, done = _agg_add({"req_id": "r11", "need": 0, "block": b})
        assert done and st["need"] == 1  # 钳到 1：不会因 0 需求产生不可完成的请求
        assert gateway_state._agg_state == {}
    finally:
        _shm_drop(shm)



def test_agg_add_dedupes_redispatched_part(gw):
    """重派批幂等：同一 chunk 集二次送达（tts worker 死亡后认领队列精确重派）
    → 重复 part 丢弃并回收其块，绝不重复聚合（否则拼接出重复句）。"""
    gateway_state._request_meta["r12"] = {"needs_align": False, "created": time.time()}
    b1, shm1, _a = _block([(0, "一")])
    b2, shm2, _a2 = _block([(0, "一")])   # 重派批重投同一 chunk（idx 相同、块名不同）
    try:
        _st, done = _agg_add({"req_id": "r12", "need": 2, "block": b1})
        assert not done  # 未收齐：_agg_add 对未完成恒返回 (None, False)
        assert gateway_state._agg_state["r12"]["idxs"] == {0}
        # 重派批二次送达同一 chunk（块名不同的重复投递）
        dup = _agg_add({"req_id": "r12", "need": 2, "block": b2})
        assert dup == (None, False), "重复 part 必须被丢弃"
        assert gateway_state._agg_state["r12"]["blocks"] == [b1], "账本不得混入重复块"
        assert gateway_state._agg_state["r12"]["idxs"] == {0}
        # 后续新 chunk 正常完成聚合
        b3, shm3, _a3 = _block([(1, "二")])
        try:
            st, done = _agg_add({"req_id": "r12", "need": 2, "block": b3})
            assert done and st["blocks"] == [b1, b3]
            assert st["texts"][0] == "一" and st["texts"][1] == "二"
        finally:
            _shm_drop(shm3)
    finally:
        _shm_drop(shm1)
        _shm_drop(shm2)


# ── 阶段耗时聚合（/tts 日志分解）──

def test_synth_part_timings_max_and_span(gw):
    """q_g2p/g2p 整单同值 → 跨批取 max（不随批数翻倍）；
    q_tts/tts 为请求级墙钟跨度（seg 级戳推算，非逐 chunk 求和）。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r13"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b1, shm1, _a = _block([(0, "一")])
    b2, shm2, _a2 = _block([(1, "二")])
    b1["segs"][0].update(q_enq=100.0, q_claim=101.0)
    b2["segs"][0].update(q_enq=100.5, q_claim=103.0)
    try:
        _on_synth_part(protocol.synth_part(rid, 2, b1, timings={
            "q_g2p": 5.0, "g2p": 100.0, "tts_done": 102.0}))
        _on_synth_part(protocol.synth_part(rid, 2, b2, timings={
            "q_g2p": 5.0, "g2p": 100.0, "tts_done": 104.0}))
        r = fut.result()
        assert r["timings"] == {"q_g2p": 5.0, "g2p": 100.0,
                                "q_tts": 3000.0,   # 103.0-100.0（入队→认领）
                                "tts": 3000.0,     # 104.0-101.0（首认领→末前向完成）
                                "chunks": 2}
    finally:
        _shm_drop(shm1)
        _shm_drop(shm2)


def test_tts_span_first_claim_to_last_forward_done(gw):
    """tts 请求级墙钟跨度 = 首个 chunk 认领 → 末块前向完成；
    chunks = need 随 timings 带回。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r-tts-span"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一"), (1, "二")])
    b["segs"][0].update(q_claim=200.0)
    b["segs"][1].update(q_claim=200.5)
    try:
        _on_synth_part(protocol.synth_part(rid, 2, b, timings={
            "tts_done": 202.0}))
        r = fut.result()
        assert r["timings"]["tts"] == 2000.0   # 202.0-200.0
        assert r["timings"]["chunks"] == 2
        assert r["fw"] == 1
    finally:
        _shm_drop(shm)


def test_tts_missing_done_stamp_zero(gw):
    """异常消息缺 tts_done 戳（有认领）：tts 跨度按 0 计，交付不炸。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r-tts-fb"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一"), (1, "二")])
    b["segs"][0].update(q_claim=200.0)
    b["segs"][1].update(q_claim=201.0)
    try:
        _on_synth_part(protocol.synth_part(rid, 2, b))
        r = fut.result()
        assert r["timings"]["tts"] == 0.0
    finally:
        _shm_drop(shm)


def test_q_tts_span_first_enqueue_to_last_claim(gw):
    """q_tts 新口径：同请求所有 chunk 从首个入队到最后一个被认领的墙钟跨度，
    不是逐 chunk 等待求和。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r-span"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一"), (1, "二")])
    b["segs"][0].update(q_enq=100.0, q_claim=101.0)
    b["segs"][1].update(q_enq=100.5, q_claim=101.5)
    try:
        _on_synth_part(protocol.synth_part(rid, 2, b))
        assert fut.result()["timings"]["q_tts"] == 1500.0   # 101.5-100.0
    finally:
        _shm_drop(shm)


def test_q_tts_span_ignores_duplicate_seg_claim(gw):
    """部分重叠重试：只有 fresh seg 推进跨度，重复 seg 的晚 claim 不拉长。"""
    gateway_state._request_meta["r-span2"] = {"needs_align": False, "created": time.time()}
    b1, shm1, _a = _block([(0, "一")])
    b1["segs"][0].update(q_enq=100.0, q_claim=101.0)
    b2, shm2, _a2 = _block([(0, "一"), (1, "二")])
    b2["segs"][0].update(q_enq=100.0, q_claim=105.0)   # 重复 seg，晚 claim
    b2["segs"][1].update(q_enq=102.0, q_claim=103.0)   # fresh seg
    try:
        _agg_add({"req_id": "r-span2", "need": 3, "block": b1})
        _agg_add({"req_id": "r-span2", "need": 3, "block": b2})
        st = gateway_state._agg_state["r-span2"]
        assert st["t_q_tts_first"] == 100.0
        assert st["t_q_tts_last"] == 103.0            # 105.0 被忽略
    finally:
        _shm_drop(shm1)
        _shm_drop(shm2)


def test_synth_part_timings_absent_zero(gw):
    """消息无 timings → 聚合不炸，耗时按 0 计。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r14"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b, shm, _a = _block([(0, "一")])
    try:
        _on_synth_part(_part(rid, 1, b))
        assert fut.result()["timings"] == {"q_g2p": 0.0, "g2p": 0.0,
                                           "q_tts": 0.0, "tts": 0.0,
                                           "chunks": 1}
    finally:
        _shm_drop(shm)


def test_synth_part_duplicate_delivery_no_double_count(gw):
    """重派重复投递（与块幂等同一拦截线）：首个 part 即交付，重复 part
    按孤儿回收，其晚认领戳不影响已交付的跨度。"""
    loop = _loop()
    fut = loop.create_future()
    rid = "r15"
    gateway_state._pending[("req", rid)] = fut
    gateway_state._request_meta[rid] = {"needs_align": False, "created": time.time()}
    b1, shm1, _a = _block([(0, "一")])
    b2, shm2, _a2 = _block([(0, "一")])
    b1["segs"][0].update(q_enq=100.0, q_claim=101.0)
    b2["segs"][0].update(q_enq=100.0, q_claim=105.0)   # 重复 seg 的晚 claim
    try:
        _on_synth_part(protocol.synth_part(rid, 1, b1, timings={
            "q_g2p": 5.0, "g2p": 100.0, "tts_done": 102.0}))
        _on_synth_part(protocol.synth_part(rid, 1, b2, timings={
            "q_g2p": 5.0, "g2p": 100.0, "tts_done": 106.0}))
        r = fut.result()
        assert r["timings"]["q_tts"] == 1000.0    # 101.0-100.0，105 不拉长
        assert r["timings"]["tts"] == 1000.0      # 102.0-101.0
    finally:
        _shm_drop(shm1)
        _shm_drop(shm2)


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_aggregator.py")

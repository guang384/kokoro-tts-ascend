"""TS 账本（gateway._on_chunk_ts）：chunk 级对齐流水的乱序收齐语义。

替代 aligner 的"整单收齐拼接"：ASR 与剩余 TTS 重叠执行，chunk_ts 可能
先于音频聚合到达、顺序不保证——账本按 idx 归位、need 收齐拼接。覆盖：
乱序/幂等/错误早退/孤儿丢弃/双投收齐/偏移拼接/TTL 清扫。"""
import asyncio
import time

import pytest

import gateway
import protocol


@pytest.fixture()
def gw(monkeypatch):
    monkeypatch.setattr(gateway, "_pending", {})
    monkeypatch.setattr(gateway, "_request_meta", {})
    monkeypatch.setattr(gateway, "_agg_state", {})
    monkeypatch.setattr(gateway, "_ts_state", {})
    return gateway


def _fut(gw, rid):
    loop = asyncio.new_event_loop()
    fut = loop.create_future()
    gw._pending[("ts", rid)] = fut
    gw._request_meta[rid] = {"needs_align": True, "created": time.time()}
    return loop, fut


def test_out_of_order_chunks_stitch(gw):
    """乱序到达：chunk_ts(1) 先于 chunk_ts(0) → 收齐拼接，
    偏移 = 前序 chunk 采样数换算 ms（chunk 级流水的常态时序）。"""
    loop, fut = _fut(gw, "r1")
    gw._on_chunk_ts(protocol.chunk_ts("r1", 1, 2, 24000, [[100, 150]]))
    assert not fut.done()                      # 未收齐
    gw._on_chunk_ts(protocol.chunk_ts("r1", 0, 2, 48000, [[0, 50]]))
    assert fut.done()
    ts = fut.result()["timestamp"]
    # chunk0（48000 采样 = 2000ms）在前；chunk1 偏移 +2000ms
    assert ts == [[0, 50], [2160, 2210]]   # 偏移=2000ms 采样+60ms gap（与 _concat 同轴）
    assert "r1" not in gw._ts_state            # 账本收口
    loop.close()


def test_duplicate_idx_idempotent(gw):
    """at-least-once 重派双投：重复 idx 忽略，不破坏收齐。"""
    loop, fut = _fut(gw, "r2")
    gw._on_chunk_ts(protocol.chunk_ts("r2", 0, 2, 800, [[0, 50]]))
    gw._on_chunk_ts(protocol.chunk_ts("r2", 0, 2, 800, [[0, 50]]))  # 重复
    gw._on_chunk_ts(protocol.chunk_ts("r2", 1, 2, 800, [[60, 110]]))
    assert fut.done()
    ts = fut.result()["timestamp"]
    assert ts == [[0, 50], [153, 203]]         # 偏移 = round(33.33+60) = 93ms（采样+gap）
    loop.close()


def test_error_early_exit(gw):
    """任一 chunk 对齐失败 → 整单早退（fut_ts set_exception），账本清空。"""
    loop, fut = _fut(gw, "r3")
    gw._on_chunk_ts(protocol.chunk_ts("r3", 0, 3, 800, [], error="模型炸了"))
    assert fut.done()
    assert "模型炸了" in str(fut.exception())
    assert "r3" not in gw._ts_state
    loop.close()


def test_orphan_chunk_ts_dropped(gw):
    """请求已终结（无 fut_ts）：孤儿 chunk_ts 静默丢弃，不建账本。"""
    gw._on_chunk_ts(protocol.chunk_ts("ghost", 0, 1, 800, [[0, 50]]))
    assert gw._ts_state == {}                   # 不建账本
    assert gw._pending == {}


def test_duplicate_error_after_success_is_orphan_safe(gw):
    """成功收齐后迟到的重复/错误 chunk_ts：账本已清、fut_ts 已摘 → 孤儿丢弃。"""
    loop, fut = _fut(gw, "r4")
    gw._on_chunk_ts(protocol.chunk_ts("r4", 0, 1, 800, [[0, 50]]))
    assert fut.done()
    gw._pending.pop(("ts", "r4"), None)         # 端点 finally 摘除
    gw._on_chunk_ts(protocol.chunk_ts("r4", 0, 1, 800, [], error="late"))
    assert gw._ts_state == {}                   # 幂等孤儿丢弃，无状态泄漏
    loop.close()


def test_need_from_first_chunk_clamps(gw):
    """need 由首条 chunk_ts 确定（clamp ≥1）：坏消息 need=0 → 按 1 收齐。"""
    loop, fut = _fut(gw, "r5")
    gw._on_chunk_ts(protocol.chunk_ts("r5", 0, 0, 800, [[0, 50]]))
    assert fut.done()                            # need 0 → clamp 1，单条收齐
    loop.close()


def test_sweep_leaks_cleans_orphan_ts_state(gw, monkeypatch):
    """TTL 清扫：孤儿 _ts_state（fut_ts 已被端点摘除但账本残留）超龄回收。"""
    gw._ts_state["r6"] = {"need": 2, "items": {},
                          "created": time.time() - 999}
    monkeypatch.setattr(gateway, "_entry_wait_s", 120)
    gw._sweep_leaks()
    assert "r6" not in gw._ts_state


def test_partial_overlap_adds_only_fresh_segs(gw):
    """审计 F3 回归：重派批重组（重派 chunk 与新 chunk 并入一块）只收新
    segs——否则 _gather_audio 把已收段读两遍（音频重复拼接）。"""
    gw._request_meta["r7"] = {"needs_align": True, "created": time.time()}
    gw._on_synth_part(protocol.synth_part("r7", 3, {
        "shm_name": "x", "segs": [{"idx": 0, "offset": 0, "samples": 8,
                                   "text": "a"}]}))
    st = gw._agg_state["r7"]
    assert [s_["idx"] for b in st["blocks"] for s_ in b["segs"]] == [0]
    # 部分重叠块：seg0（已收）+ seg1（新）——need=3 保持账本存活到断言
    gw._on_synth_part(protocol.synth_part("r7", 3, {
        "shm_name": "y", "segs": [{"idx": 0, "offset": 0, "samples": 8,
                                   "text": "a"},
                                  {"idx": 1, "offset": 8, "samples": 8,
                                   "text": "b"}]}))
    st = gw._agg_state.get("r7")
    segs = [s_["idx"] for b in st["blocks"] for s_ in b["segs"]]
    assert segs == [0, 1]                        # seg0 不重复入账
    assert len(st["blocks"]) == 2                # 块引用保留，段已过滤


def test_error_after_success_not_flipped(gw):
    """审计 F10 回归：已收成功的 chunk 被重复 job 的偶发失败不翻案。"""
    loop, fut = _fut(gw, "r8")
    gw._on_chunk_ts(protocol.chunk_ts("r8", 0, 2, 800, [[0, 50]]))
    gw._on_chunk_ts(protocol.chunk_ts("r8", 0, 2, 800, [],
                                      error="重复 job 偶发失败"))
    assert not fut.done()                        # 不炸（该 chunk 已成功）
    gw._on_chunk_ts(protocol.chunk_ts("r8", 1, 2, 800, [[60, 110]]))
    assert fut.done()                            # 正常收齐
    ts = fut.result()["timestamp"]
    assert ts == [[0, 50], [153, 203]]
    loop.close()


# ── 阶段耗时随行（/tts 日志分解的 asr 段）──

def test_chunk_ts_timings_sum_and_idempotent(gw):
    """asr 计算/队列等待逐 chunk 累加（缺戳时的兜底口径）；
    重复 idx（at-least-once 双投）不双计。"""
    loop, fut = _fut(gw, "r9")
    gw._on_chunk_ts(protocol.chunk_ts("r9", 0, 2, 800, [[0, 50]],
                                      timings={"q_asr": 1.0, "asr": 200.0}))
    gw._on_chunk_ts(protocol.chunk_ts("r9", 0, 2, 800, [[0, 50]],
                                      timings={"q_asr": 1.0, "asr": 200.0}))
    gw._on_chunk_ts(protocol.chunk_ts("r9", 1, 2, 800, [[60, 110]],
                                      timings={"q_asr": 2.0, "asr": 300.0}))
    assert fut.done()
    assert fut.result()["timings"] == {"asr": 500.0, "q_asr": 3.0, "asr_fw": 0}
    loop.close()


def test_asr_span_first_claim_to_last_bucket_done(gw):
    """asr 新口径：首 job 认领 → 末桶推理完成的请求级墙钟跨度（非逐 chunk
    份额求和）；asr_fw = 桶起始时刻去重计数（多桶 = ASR 分桶/跨批）。"""
    loop, fut = _fut(gw, "r-asr-span")
    gw._on_chunk_ts(protocol.chunk_ts(
        "r-asr-span", 0, 3, 800, [[0, 50]],
        timings={"q_asr": 1.0, "asr": 999.0,
                 "q_asr_first": 100.0, "q_asr_last": 100.5,
                 "asr_done": 102.0, "asr_bucket": 400.0}))
    gw._on_chunk_ts(protocol.chunk_ts(
        "r-asr-span", 1, 3, 800, [[60, 110]],
        timings={"q_asr": 1.0, "asr": 999.0,
                 "q_asr_first": 100.0, "q_asr_last": 101.0,
                 "asr_done": 102.0, "asr_bucket": 400.0}))   # 同桶
    gw._on_chunk_ts(protocol.chunk_ts(
        "r-asr-span", 2, 3, 800, [[120, 170]],
        timings={"q_asr": 1.0, "asr": 999.0,
                 "q_asr_first": 100.5, "q_asr_last": 103.0,
                 "asr_done": 104.5, "asr_bucket": 405.0}))   # 第二桶
    r = fut.result()["timings"]
    assert r["asr"] == 4000.0    # 104.5-100.5 跨度（非 999×3 求和）
    assert r["asr_fw"] == 2      # 去重桶数
    assert r["q_asr"] == 3000.0  # 103.0-100.0 排队跨度
    loop.close()


def test_q_asr_span_first_enqueue_to_last_claim(gw):
    """q_asr 新口径：请求所有 chunk 从首个 ASR job 入队到最后一个被认领的
    墙钟跨度，不是逐 chunk 等待求和。缺 asr_done 戳时 asr 回退算力总和。"""
    loop, fut = _fut(gw, "r-span")
    gw._on_chunk_ts(protocol.chunk_ts(
        "r-span", 0, 2, 800, [[0, 50]],
        timings={"q_asr": 1.0, "asr": 100.0,
                 "q_asr_first": 100.0, "q_asr_last": 101.0}))
    gw._on_chunk_ts(protocol.chunk_ts(
        "r-span", 1, 2, 800, [[60, 110]],
        timings={"q_asr": 2.0, "asr": 200.0,
                 "q_asr_first": 100.2, "q_asr_last": 103.0}))
    assert fut.done()
    assert fut.result()["timings"]["q_asr"] == 3000.0    # 103.0-100.0
    assert fut.result()["timings"]["asr"] == 300.0       # 缺 done 戳 → 求和兜底
    loop.close()


def test_chunk_ts_timings_legacy_absent_zero(gw):
    """旧消息无 timings → 账本不炸，耗时按 0 计。"""
    loop, fut = _fut(gw, "r10")
    gw._on_chunk_ts(protocol.chunk_ts("r10", 0, 1, 800, [[0, 50]]))
    assert fut.result()["timings"] == {"asr": 0.0, "q_asr": 0.0, "asr_fw": 0}
    loop.close()

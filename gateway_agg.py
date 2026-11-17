"""gateway 聚合引擎：结果泵 + synth_part/chunk_ts 账本 + SHM 块生命周期。

从 gateway 拆出的事件循环侧纯逻辑；全部可变状态经 gateway_state(gst) 模块
属性访问（测试 patch gst 的绑定对这里可见）。_handle_result 由 _pump_once
分发；_sweep_leaks 由 gateway_obs 的心跳循环周期调用。
"""
import logging
import time

import protocol

import gateway_state as gst
from gateway_state import _SynthError

log = logging.getLogger("kokoro.server")


def _sweep_leaks() -> None:
    """TTL 清扫：请求方已放弃的聚合状态/孤儿 Future——部分失败场景下
    （worker 崩溃丢块、asr 作业丢失）唯一的回收手段，防 gst._pending/
    gst._agg_state/gst._ts_state/SHM 块随时间单调增长。"""
    now = time.time()
    for key, fut in list(gst._pending.items()):
        if fut.done():
            gst._pending.pop(key, None)
    for rid, meta in list(gst._request_meta.items()):
        if now - meta["created"] > 3 * gst._entry_wait_s:
            gst._request_meta.pop(rid, None)
    for rid, st in list(gst._agg_state.items()):
        if now - st["created"] > 2 * gst._entry_wait_s:
            gst._agg_state.pop(rid, None)
            _unlink_blocks(st["blocks"])
            gst._entry_stats["swept"] += 1
    for rid, st in list(gst._ts_state.items()):
        if now - st["created"] > 2 * gst._entry_wait_s:
            gst._ts_state.pop(rid, None)  # 孤儿 TS 账本（fut_ts 已被端点 finally 摘除）
    for rid, items in list(gst._retired_blocks.items()):
        keep, old = [], []
        for block, ts in items:
            if now - ts > 2 * gst._entry_wait_s:
                old.append(block)
            else:
                keep.append((block, ts))
        if old:
            _unlink_blocks(old)
        if keep:
            gst._retired_blocks[rid] = keep
        else:
            gst._retired_blocks.pop(rid, None)


def _pump_once(result_q) -> bool:
    """单次取消息 + 递交；任何异常都不外抛（泵线程绝不能死）。

    get() 本身也可能抛：Redis 瞬断/超时、WRONGTYPE、坏 pickle（_recv 已
    兜坏 pickle，但连接类错误仍会到这里）。泵一死，所有后续请求 504，
    而 /health 可能还是绿的——这是最隐蔽的单点。返回 False 表示本轮
    取消息失败，调用方退避后重试。"""
    try:
        item = result_q.get()
    except Exception:  # noqa: BLE001
        log.exception("result pump dequeue failed (retry)")
        return False
    try:
        gst._main_loop.call_soon_threadsafe(_handle_result, item)
    except Exception:  # noqa: BLE001
        log.exception("result pump delivery failed (message dropped)")
    return True


def _result_pump(result_q) -> None:
    """独立线程：把结果队列的事件递交给服务事件循环。

    任何单条消息/递交异常都不允许杀死泵——泵一死，所有后续请求 504。
    loop 就绪判断在 get 之前：启动窗口期消息留在队列管道里（不消费不丢），
    loop 就绪后照常送达；若日后调整 _lifespan 顺序（泵先于 loop 赋值启动）
    也不会变成启动期静默丢结果。"""
    while True:
        if gst._main_loop is None or gst._main_loop.is_closed():
            time.sleep(0.05)
            continue
        if not _pump_once(result_q):
            time.sleep(0.05)


def _unlink_blocks(blocks: list) -> None:
    """SHM 块清理：attach → unlink（块由 tts 池创建，gateway 负责回收）。"""
    from multiprocessing import shared_memory

    for b in blocks or []:
        try:
            shm = shared_memory.SharedMemory(name=b["shm_name"])
            shm.close()
            shm.unlink()
        except Exception:  # noqa: BLE001  已被清理/名字不存在：幂等
            pass


def _retire_block(rid: str, block: dict) -> bool:
    """重复/孤儿 synth_part 块的延迟回收。

    /tts 的 ASR job 在 synth_part 之后派发，且只携带 shm_name——若 tts
    worker 在两者之间死亡，重试批的 synth_part 会被按重复段拦下，但该块
    仍是重试 ASR job 的唯一副本。此时立即 unlink 会让合法请求 502。
    仅当该请求的 TS Future 仍存在时挂起到 gst._retired_blocks，由端点 finally
    （TS 收齐/超时/断开）或 _sweep_leaks 统一回收；/v1 无 TS Future，
    沿用即时 unlink。"""
    if gst._pending.get(("ts", rid)) is not None:
        gst._retired_blocks.setdefault(rid, []).append((block, time.time()))
        return True
    _unlink_blocks([block])
    return False


# ── 结果消息处理（事件循环线程内执行） ──

def _agg_add(res: dict):
    """聚合一条 synth_part。返回 (st, done)；孤儿/未完成/重复返回 (None, False)。"""
    rid = res["req_id"]
    st = gst._agg_state.get(rid)
    if st is None:
        if rid not in gst._request_meta:
            # 请求已终止（超时/取消）：块即刻回收；但 /tts 的 TS 账本可能
            # 仍在等 ASR 读这个块（重试窗口）——由 _retire_block 判定。
            _retire_block(rid, res["block"])
            return None, False
        st = gst._agg_state[rid] = {"need": max(1, int(res["need"])), "idxs": set(),
                                "blocks": [], "texts": {}, "created": time.time(),
                                "t_g2p": 0.0, "t_q_g2p": 0.0,
                                # 请求级 TTS 排队跨度：min(首 chunk 入队) /
                                # max(末 chunk 出队)（seg 级 q_enq/q_claim）。
                                "t_q_tts_first": None, "t_q_tts_last": None,
                                # 请求级 TTS 墙钟跨度：min(首 chunk 认领) /
                                # max(末块前向完成 tts_done)。
                                "t_tts_claim_first": None,
                                "t_tts_done_last": None}
    # 幂等去重：worker 死亡后认领队列会把在途 chunk 精确重派（at-least-once），
    # 同一批 chunk 可能二次前向送达。重派批的 chunk 集与原批恒相同（同一批
    # 消息重投），全部 idx 已收过 → 重复，块即刻回收，绝不重复聚合（否则拼接
    # 出重复句）。部分重叠按重复处理——该不变量下不存在此形态，防御性从紧。
    all_segs = res["block"]["segs"]
    seg_idxs = {s["idx"] for s in all_segs}
    if seg_idxs and seg_idxs <= st["idxs"]:
        _retire_block(rid, res["block"])
        return None, False
    # 部分重叠（重派批重组：重派 chunk 与同请求新 chunk 并入一块）：
    # 只收新 segs——否则 _gather_audio 会把已收段读两遍（审计 F3，
    # 旧注释声称"部分重叠不存在"的不变量在单条重派下不成立）
    fresh = [s for s in all_segs if s["idx"] not in st["idxs"]]
    res["block"]["segs"] = fresh
    t = res.get("timings")
    if t and fresh:   # 只随实际入账的新 segs 计时（重复投递上面已拦，防御性兜底）
        # q_g2p/g2p 整单一次算、各 chunk 同值 → 取 max（重派批覆盖整块
        # 成员，max 口径对重复/部分重叠投递天然幂等）；tts_done 同理取 max。
        st["t_q_g2p"] = max(st["t_q_g2p"], float(t.get("q_g2p", 0.0)))
        st["t_g2p"] = max(st["t_g2p"], float(t.get("g2p", 0.0)))
        done = t.get("tts_done")
        if isinstance(done, (int, float)):
            st["t_tts_done_last"] = (done if st["t_tts_done_last"] is None
                                     else max(st["t_tts_done_last"], done))
    st["blocks"].append(res["block"])
    seg_enq, seg_claim = [], []
    for seg in fresh:
        st["idxs"].add(seg["idx"])
        st["texts"][seg["idx"]] = seg.get("text", "")
        if isinstance(seg.get("q_enq"), (int, float)):
            seg_enq.append(seg["q_enq"])
        if isinstance(seg.get("q_claim"), (int, float)):
            seg_claim.append(seg["q_claim"])
    if seg_claim:
        # tts 墙钟跨度起点：min(认领)。只由 fresh seg 推进（同 q_tts 跨度）。
        c_min = min(seg_claim)
        st["t_tts_claim_first"] = (c_min if st["t_tts_claim_first"] is None
                                   else min(st["t_tts_claim_first"], c_min))
    if seg_enq and seg_claim:
        # 请求级跨度：只由实际入账的 fresh seg 推进。重复/重试块里的已收
        # seg 不参与，避免重试的较晚 claim 把 q_tts 拉长。
        e, c_ = min(seg_enq), max(seg_claim)
        st["t_q_tts_first"] = (e if st["t_q_tts_first"] is None
                               else min(st["t_q_tts_first"], e))
        st["t_q_tts_last"] = (c_ if st["t_q_tts_last"] is None
                              else max(st["t_q_tts_last"], c_))
    if len(st["idxs"]) < st["need"]:
        return None, False
    gst._agg_state.pop(rid, None)
    return st, True


def _deliver(fut, st: dict, texts: list) -> bool:
    """把聚合完成的音频交付给等待方；请求已不在（超时/取消）则清块。"""
    if fut is None or fut.done():
        _unlink_blocks(st["blocks"])
        return False
    # 请求级墙钟跨度（同版本 worker 恒带戳；异常缺戳按 0 计，不炸交付）：
    # q_tts = 首个 chunk 入队 → 最后一个 chunk 被认领；
    # tts = 首个 chunk 认领 → 末块前向完成。
    q_tts = tts_ms = 0.0
    if (st["t_q_tts_first"] is not None
            and st["t_q_tts_last"] is not None):
        q_tts = max(0.0, (st["t_q_tts_last"]
                          - st["t_q_tts_first"]) * 1000)
    if (st["t_tts_claim_first"] is not None
            and st["t_tts_done_last"] is not None):
        tts_ms = max(0.0, (st["t_tts_done_last"]
                           - st["t_tts_claim_first"]) * 1000)
    fut.set_result({"blocks": st["blocks"], "texts": texts,
                    "fw": len(st["blocks"]),
                    "timings": {"q_g2p": st["t_q_g2p"], "g2p": st["t_g2p"],
                                "q_tts": q_tts, "tts": tts_ms,
                                "chunks": st["need"]}})
    return True


def _on_synth_part(res: dict) -> None:
    rid = res["req_id"]
    st, done = _agg_add(res)
    if not done:
        return
    texts = [st["texts"][i] for i in sorted(st["texts"])]
    fut = gst._pending.pop(("req", rid), None)
    gst._request_meta.pop(rid, None)  # 请求状态收口：后续重复消息按孤儿处理
    # 音频即到即交付：对齐（chunk_ts）与音频两账本独立收齐——chunk 级
    # 流水下 chunk_ts 可能先于音频聚合到达（乱序安全），/tts 的 fut_ts
    # 由 _on_chunk_ts 收齐后 resolve
    # 成功计数上移到端点（/v1 编码成功后、/tts 音频+TS 全链路成功后）——
    # 音频到达 ≠ 请求成功：/tts 后续还有对齐，编码也可能失败。
    _deliver(fut, st, texts)


def _on_synth_error(res: dict) -> None:
    rid = res["req_id"]
    st = gst._agg_state.pop(rid, None)
    if st is not None:
        _unlink_blocks(st["blocks"])
    fut = gst._pending.pop(("req", rid), None)
    fut_ts = gst._pending.get(("ts", rid))   # 不 pop：音频已交付时留给 TS 账本
    # 各 Future 独立判定：真失败（音频未交付）→ 两个 Future 都炸（各配
    # 独立异常实例，防 traceback 串线）；迟到 synth_error（批重派的第二
    # 次前向失败、音频已交付）→ 不碰 fut_ts，TS 在途由账本推进、请求
    # 应成功（审计 F2：f6266b2 曾在此把好的请求打成 502，方向修反）
    if fut is not None and not fut.done():
        fut.set_exception(_SynthError(res.get("kind", protocol.KIND_COMPUTE),
                                      res.get("error", "合成失败")))
        if fut_ts is not None and not fut_ts.done():
            gst._pending.pop(("ts", rid), None)
            fut_ts.set_exception(_SynthError(
                protocol.KIND_COMPUTE,
                f"对齐前合成失败: {str(res.get('error', ''))[:120]}"))



def _on_chunk_ts(res: dict) -> None:
    """TS 账本：乱序安全（ASR 与剩余 TTS 重叠执行，chunk_ts 可能先于
    音频聚合到达）；error 早退（整单对齐失败）；重复 idx 幂等（at-least-once
    重派的重复投递）。收齐 need 条 → 按 idx 排序拼接（偏移=前序采样数）
    → resolve fut_ts。"""
    rid = res["req_id"]
    st = gst._ts_state.get(rid)
    if st is None:
        if gst._pending.get(("ts", rid)) is None:
            return  # 请求已终结（超时/取消/失败）：孤儿丢弃
        st = gst._ts_state[rid] = {"need": max(1, int(res.get("need", 1))),
                               "items": {}, "created": time.time(),
                               # 请求级 ASR 排队跨度：min(首 job 入队) /
                               # max(末 job 出队)（job 级 _q_enq/_q_claim）。
                               "t_q_asr_first": None, "t_q_asr_last": None,
                               # 请求级 ASR 墙钟跨度：min(首 job 认领) /
                               # max(末桶完成 asr_done)。
                               # asr_buckets = 桶起始时刻去重集合 → asr_fw。
                               "t_asr_claim_first": None,
                               "t_asr_done_last": None,
                               "asr_buckets": set()}
    if res.get("error"):
        idx = int(res["idx"])
        if idx in st["items"]:
            return  # 该 chunk 已成功收账：重复 job 的偶发失败不翻案（审计 F10）
        gst._ts_state.pop(rid, None)
        fut_ts = gst._pending.pop(("ts", rid), None)
        if fut_ts is not None and not fut_ts.done():
            # 对齐服务故障 → 502（非通用 500；审计 F5）
            fut_ts.set_exception(_SynthError(
                protocol.KIND_COMPUTE,
                f"对齐服务失败: {str(res['error'])[:120]}"))
        return
    idx = int(res["idx"])
    if idx in st["items"]:
        return  # 重派重复投递：幂等
    t = res.get("timings")
    if t:   # 只维护请求级跨度戳（请求级耗时由跨度计算，不求和）
        e, c_ = t.get("q_asr_first"), t.get("q_asr_last")
        if isinstance(e, (int, float)):
            st["t_q_asr_first"] = (e if st["t_q_asr_first"] is None
                                   else min(st["t_q_asr_first"], e))
        if isinstance(c_, (int, float)):
            st["t_q_asr_last"] = (c_ if st["t_q_asr_last"] is None
                                  else max(st["t_q_asr_last"], c_))
            # asr 墙钟跨度起点 = min(认领)（q_asr_last 即认领时刻）
            st["t_asr_claim_first"] = (
                c_ if st["t_asr_claim_first"] is None
                else min(st["t_asr_claim_first"], c_))
        done = t.get("asr_done")
        if isinstance(done, (int, float)):
            st["t_asr_done_last"] = (done if st["t_asr_done_last"] is None
                                     else max(st["t_asr_done_last"], done))
        bkt = t.get("asr_bucket")
        if isinstance(bkt, (int, float)):
            st["asr_buckets"].add(bkt)
    st["items"][idx] = (int(res.get("samples", 0)),
                        res.get("timestamp") or [])
    if len(st["items"]) < st["need"]:
        return
    gst._ts_state.pop(rid, None)
    offset_ms = 0.0
    all_ts = []
    for i in sorted(st["items"]):
        samples, ts = st["items"][i]
        for s_, e_ in ts:
            all_ts.append([s_ + round(offset_ms), e_ + round(offset_ms)])
        # 偏移 = 前序采样 + 前序 gap（与 _concat 的插静音同轴——审计 F1）
        offset_ms += samples / gst._sample_rate * 1000 + gst._CHUNK_GAP_S * 1000
    fut_ts = gst._pending.pop(("ts", rid), None)
    if fut_ts is not None and not fut_ts.done():
        # 请求级墙钟跨度（同版本 worker 恒带戳；异常缺戳按 0 计）：
        # q_asr = 首 job 入队 → 末 job 出队；asr = 首 job 认领 → 末桶完成。
        q_asr = asr_ms = 0.0
        if (st["t_q_asr_first"] is not None
                and st["t_q_asr_last"] is not None):
            q_asr = max(0.0, (st["t_q_asr_last"]
                              - st["t_q_asr_first"]) * 1000)
        if (st["t_asr_claim_first"] is not None
                and st["t_asr_done_last"] is not None):
            asr_ms = max(0.0, (st["t_asr_done_last"]
                               - st["t_asr_claim_first"]) * 1000)
        fut_ts.set_result({"timestamp": all_ts,
                           "timings": {"asr": asr_ms,
                                       "q_asr": q_asr,
                                       "asr_fw": len(st["asr_buckets"])}})



def _handle_result(res: dict) -> None:
    try:
        t = res.get("type")
        if t == protocol.SYNTH_PART:
            _on_synth_part(res)
        elif t == protocol.SYNTH_ERROR:
            _on_synth_error(res)
        elif t == protocol.CHUNK_TS:
            _on_chunk_ts(res)
        else:
            log.warning("unknown result message type=%r (dropped)", t)
    except Exception:  # noqa: BLE001  单条消息异常不影响服务
        log.exception("result handling error (message dropped): %r", res)

"""请求级计算精度（dtype）：校验 / 协议 / 认领凑批隔离 / 端到端透传。

dtype=fp16 走 autocast（权重保 fp32），凑批类键 = (音素长度类, dtype)——
两种精度不混批（autocast 是整批上下文）。KOKORO_DTYPE 进程级开关已退役。
"""
import asyncio
import struct

import pytest

import protocol


# ── 校验：非法值必须列出可选值 ──
@pytest.mark.parametrize("bad", ["fp8", "FP32X", "half", "float16", 16, ["fp16"]])
def test_normalize_dtype_rejects_with_choices(bad):
    with pytest.raises(ValueError) as ei:
        protocol.normalize_dtype(bad)
    msg = str(ei.value)
    assert "fp32/fp16" in msg          # 可选值在错误说明里
    assert "缺省 fp32" in msg


@pytest.mark.parametrize("ok,expect", [
    (None, "fp32"), ("", "fp32"), ("fp32", "fp32"),
    ("fp16", "fp16"), (" FP16 ", "fp16"),   # 大小写/空白容错
])
def test_normalize_dtype_accepts(ok, expect):
    assert protocol.normalize_dtype(ok) == expect


def test_new_request_and_chunk_carry_dtype():
    msg = protocol.new_request("r1", 0, "你好", "zf_001", 1.0, True)
    assert msg["dtype"] == "fp32"                       # 缺省显式化
    msg2 = protocol.new_request("r2", 0, "你好", "zf_001", 1.0, True, dtype="fp16")
    assert msg2["dtype"] == "fp16"
    c = protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 0, 1, "你好", dtype="fp16")
    assert c["dtype"] == "fp16"
    assert protocol.chunk("r1", 1, "ə", "zf_001", 1.0, 0, 1, "你好")["dtype"] == "fp32"


# ── 认领凑批：dtype 不混批（member 头部第 3 字节 + Lua 类键）──
def _chunk(rid="r", idx=0, n=30, dtype="fp32"):
    return {"req_id": rid, "idx": idx, "ps": "ə" * n, "voice": "zf_001",
            "speed": 1.0, "origin": 0, "need": 1, "text": "句", "dtype": dtype}


def test_claim_dtype_isolation(fakeredis_client, monkeypatch):
    """fp32/fp16 同长度类混在队列：认领只取与 head 同 dtype 的——两种精度
    不混批（autocast 整批上下文；混入即错精度执行）。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=52, name="dtype-iso")
    q.put(_chunk("a0", dtype="fp32"))
    q.put(_chunk("a1", dtype="fp32"))
    q.put(_chunk("b0", dtype="fp16"))
    q.put(_chunk("b1", dtype="fp16"))
    claimed = q.claim(4, "tts-0-1")
    dts = [c["dtype"] for _m, c in claimed]
    assert set(dts) == {"fp32"}                       # head=fp32 → 只取 fp32
    assert len(claimed) == 2
    q.ack([m for m, _c in claimed], "tts-0-1")
    claimed2 = q.claim(4, "tts-0-1")
    assert [c["dtype"] for _m, c in claimed2] == ["fp16", "fp16"]  # 下一轮 fp16
    q.ack([m for m, _c in claimed2], "tts-0-1")


def test_member_header_dtype_byte(fakeredis_client, monkeypatch):
    """member 头部第 3 字节编码精度（Lua 类键读它）；序号偏移随之 3:23。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=53, name="hdr")
    m, _payload, _seq = q._encode(_chunk("x", dtype="fp16"))
    assert struct.unpack(">H", m[:2])[0] == 30        # 长度头不变
    assert m[2] == 1                                  # fp16 → 1
    m32, _p, _s = q._encode(_chunk("y", dtype="fp32"))
    assert m32[2] == 0
    assert len(m) == pc._CLAIM_HEADER + len(_payload)  # 23B 头


def test_requeue_preserves_fifo_with_dtype(fakeredis_client, monkeypatch):
    """死亡重派：序号偏移 3:23（+1B 精度）——FIFO 序保持（同长度类跨 dtype
    亦按到达序回队列）。"""
    import pickle

    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=54, name="dtype-fifo")
    q.put(_chunk("ra", dtype="fp32"))
    q.put(_chunk("rb", dtype="fp16"))
    claimed = q.claim(4, "tts-0-9")                   # 只认领 head 的 fp32 → ra
    assert [c["req_id"] for _m, c in claimed] == ["ra"]
    # 模拟死亡：账本残留，requeue_worker 精确回队列
    n = q.requeue_worker("tts-0-9")
    assert n == 1
    scores = fakeredis_client.zrange(q._key, 0, -1, withscores=True)
    order = [pickle.loads(m[pc._CLAIM_HEADER:])["req_id"] for m, _s in scores]
    assert order == ["ra", "rb"]                      # ra 回到 rb 之前（FIFO）


# ── 端到端透传（dryrun 网关替身，G2P 层只锁协议）──
@pytest.mark.parametrize("bad", ["fp8", 16, 1.5, True, ["fp16"], {"a": 1}])
def test_gateway_dtype_validation_and_passthrough(bad):
    """坏 dtype（含非字符串类型，模型字段为 Any）→ 400 且列可选值。"""
    from gateway import BizError, TTSRequest, tts_business

    req = TTSRequest(text="你好", dtype=bad)
    with pytest.raises(BizError) as ei:
        asyncio.run(tts_business(req))
    assert ei.value.status_code == 400
    assert "fp32/fp16" in ei.value.message
    assert "缺省 fp32" in ei.value.message


def test_synth_batch_dtype_signature():
    """synth_batch 接受 dtype 参数；fp16 走 autocast（device_type 取实际
    设备——CPU 上同样生效，仅作本地验证；真机 NPU 由 CANN 选型）。"""
    from pipeline import TTSCompute
    sig = TTSCompute.synth_batch.__code__.co_varnames[:3]
    assert "dtype" in sig
    import inspect
    src = inspect.getsource(TTSCompute.synth_batch)
    assert "torch.autocast" in src and 'dtype == "fp16"' in src


# ── 毒 member 全路径：claim / sweep 都不可被坏载荷带崩 ──
def test_claim_short_member_does_not_break_lua(fakeredis_client, monkeypatch):
    """1 字节 member（外部误写）：旧 Lua 的 string.byte(m,2)=nil 会算术
    报错 → 每次 claim 都失败 → tts 池永久停摆。兜 0 后由 Python 侧丢弃。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=57, name="short")
    fakeredis_client.zadd(q._key, {b"\x00": 1})
    assert q.claim(4, "tts-0-1") == []       # 不抛 = worker 活着
    assert q.stats()["bad"] == 1


def test_claim_drops_non_dict_payload(fakeredis_client, monkeypatch):
    """合法 pickle 但非 dict（载荷被替换）：与解码失败同款丢弃。
    否则 chunk.pop 抛 AttributeError，member 卡在 infl 直至 TTL，
    每次重派再崩一次（受影响请求全部 504）。"""
    import pickle

    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=55, name="nondict")
    payload = pickle.dumps(["not", "a", "dict"])
    member = struct.pack(">H", 30) + b"\x00" + b"0" * 20 + payload
    fakeredis_client.zadd(q._key, {member: 1})
    assert q.claim(4, "tts-0-1") == []
    assert q.stats()["bad"] == 1
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-1") == 0


def test_sweep_drops_bad_inflight_without_aborting_others(
        fakeredis_client, monkeypatch):
    """infl 账本里的旧格式/损坏 member：丢弃计数，不得中断同账本其余
    chunk 的精确重派（滚动升级混跑的收尾路径）。"""
    import time

    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=56, name="sweepbad")
    q.put(_chunk("ok", dtype="fp32"))
    assert len(q.claim(4, "tts-0-9")) == 1
    bad = b"\x00\x1e\x00" + b"not-a-seq"     # 序号区非数字：int() 必抛
    fakeredis_client.zadd(f"{q._key}:infl:tts-0-9", {bad: 1})
    out = q.sweep(alive_pids=set(), now=time.time(), ttl=600.0,
                  max_redispatch=3)
    assert out["redispatched"] == 1          # 正常 chunk 照常重派
    assert q.stats()["bad"] == 1
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-9") == 0


# ── 离线（fakeredis+lua）覆盖近期队列改动：22B 兼容 / ack / 毒块 / 后置簿记 ──

class _FlakyRedis:
    """委托代理：按需让 get/eval 抛错，验证 claim 后置簿记/ack 重试。"""

    def __init__(self, inner, fail_get=False, fail_eval_times=0):
        self._inner = inner
        self._fail_get = fail_get
        self._fail_eval_times = fail_eval_times

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get(self, *a, **k):
        if self._fail_get:
            raise RuntimeError("injected get failure")
        return self._inner.get(*a, **k)

    def eval(self, *a, **k):
        if self._fail_eval_times > 0:
            self._fail_eval_times -= 1
            raise RuntimeError("injected eval failure")
        return self._inner.eval(*a, **k)


def _legacy_member(rid="legacy", ps=30, seq=1234):
    """旧版 member：2B 音素数 + 20B 序号 + pickle（无精度字节）。"""
    import pickle

    chunk = _chunk(rid, 0, n=ps)
    chunk.pop("dtype")
    return (struct.pack(">H", ps) + f"{seq:020d}".encode()
            + pickle.dumps(chunk))


def test_claim_accepts_legacy_22b_member_offline(fakeredis_client, monkeypatch):
    """22B 单向兼容：新代码消费旧 member 时显式补 fp32，而不是丢弃。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=60, name="legacy")
    fakeredis_client.zadd(q._key, {_legacy_member(): 1234})
    claimed = q.claim(4, "tts-0-9")
    assert len(claimed) == 1
    assert claimed[0][1]["req_id"] == "legacy"
    assert claimed[0][1]["dtype"] == "fp32"
    assert q.stats()["bad"] == 0
    assert q.ack([claimed[0][0]], "tts-0-9") == 1


def test_requeue_worker_accepts_legacy_inflight_offline(fakeredis_client, monkeypatch):
    """旧 worker 认领的 22B member 留在 infl：新代码按旧头解出 seq 回队列。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=61, name="legacy-infl")
    legacy = _legacy_member("legacy-infl", seq=4321)
    fakeredis_client.zadd(f"{q._key}:infl:tts-0-7", {legacy: 1})
    assert q.requeue_worker("tts-0-7") == 1
    assert q.depth() == 1 and q.stats()["bad"] == 0
    claimed = q.claim(4, "tts-0-9")
    assert len(claimed) == 1 and claimed[0][1]["req_id"] == "legacy-infl"


def test_ack_clears_redispatch_counter_offline(fakeredis_client, monkeypatch):
    """ack 成功 = ZREM 成功才 HDEL :ratt；否则慢 worker 晚 ack 会误清。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=62, name="ack")
    q.put(_chunk("r-ack"))
    m, _c = q.claim(4, "tts-0-321")[0]
    fakeredis_client.hincrby(q._key_ratt, m, 1)
    assert q.ack([m], "tts-0-321") == 1
    assert fakeredis_client.hlen(q._key_ratt) == 0
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-321") == 0


def test_redispatch_max_zero_drops_on_death_path(fakeredis_client, monkeypatch):
    """REDISPATCH_MAX=0 必须与 sweep 一致：死亡主路径也不重派（毒块防护）。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=63, name="redisp0")
    q.put(_chunk("poison"))
    assert len(q.claim(4, "tts-0-1")) == 1
    assert q.requeue_worker("tts-0-1", max_redispatch=0) == 0
    assert q.depth() == 0 and q.stats()["bad"] == 1


def test_claim_mixed_good_bad_cleans_bad_only(fakeredis_client, monkeypatch):
    """好坏混合认领：好 chunk 照常返回，坏 member 从 infl/:ratt 清掉。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=64, name="mixed")
    q.put(_chunk("ok"))
    bad = b"\x00\x1e\x00" + b"not-a-seq"   # 长度头 30 + dtype 0 + 非法序号
    fakeredis_client.zadd(q._key, {bad: 1})
    claimed = q.claim(4, "tts-0-9")
    assert [c["req_id"] for _m, c in claimed] == ["ok"]
    assert q.stats()["bad"] == 1
    # 好 chunk 仍在 infl（等 ack）；坏 member 已清
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-9") == 1
    assert q.ack([claimed[0][0]], "tts-0-9") == 1
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-9") == 0


def test_claim_returns_members_when_bookkeeping_fails(fakeredis_client, monkeypatch):
    """claim 的 Lua 已成功 → 后置 EMA/计数失败不能把已认领批抛掉。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=65, name="claim-besteffort")
    q.put(_chunk("ok"))
    monkeypatch.setattr(
        q, "_redis", lambda: _FlakyRedis(fakeredis_client, fail_get=True))
    claimed = q.claim(4, "tts-0-1")
    assert len(claimed) == 1 and claimed[0][1]["req_id"] == "ok"


def test_ack_retries_once_after_transient_eval_failure(fakeredis_client, monkeypatch):
    """ack 第一次 EVAL 失败 → 重试一次（幂等），不把 infl 留给 TTL。"""
    import proc_common as pc

    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(64, index=66, name="ack-retry")
    q.put(_chunk("r-retry"))
    m, _c = q.claim(4, "tts-0-1")[0]
    proxy = _FlakyRedis(fakeredis_client, fail_eval_times=1)
    monkeypatch.setattr(q, "_redis", lambda: proxy)
    assert q.ack([m], "tts-0-1") == 1
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-1") == 0


def test_ship_batch_routes_chunk_dtype():
    """ship_batch 把 chunk 的 dtype 传给 synth_batch（最后一跳接线：
    认领类键已隔离，这里锁住执行侧不会静默踩 fp32）。"""
    from multiprocessing import shared_memory

    import numpy as np

    import compute
    import proc_common
    from tests.fake_queue import FakeQueue

    seen = {}

    class _C:
        device = "cpu"

        def synth_batch(self, jobs, dtype="fp32"):
            seen["dtype"] = dtype
            return [np.zeros(4, dtype="<i2") for _ in jobs]

    gq = FakeQueue()
    log = proc_common.setup_logging("test-dtype")
    chunk = protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 0, 1, "你好",
                           dtype="fp16")
    compute.ship_batch(_C(), {"batch_id": "b", "chunks": [chunk]}, [gq],
                       FakeQueue(), log, np, shared_memory)
    assert seen["dtype"] == "fp16"
    assert gq.get_nowait()["type"] == protocol.SYNTH_PART

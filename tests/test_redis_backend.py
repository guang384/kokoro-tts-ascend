"""Redis 传输后端单元语义（fakeredis 单进程全锁）：
put/get 往返、depth=LLEN 真实等待数、容量门 Full、INCR 计数、EMA、
_q_ts 用后即除、poll 循环的唤醒即时性。"""
import queue
import threading
import time

import pytest

import proc_common as pc


@pytest.fixture()
def q(fakeredis_client, monkeypatch):
    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.InstrumentedQueue(16, index=50, name="unit")
    yield q
    q.close()


def test_roundtrip_and_depth(q):
    """put → depth=1（LLEN 真实等待数）→ get → 归零；payload 完整、_q_ts 用后即除。"""
    q.put_nowait({"req_id": "r1", "idx": 0})
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and q.stats()["depth"] != 1:
        time.sleep(0.01)                      # 容量门/RPUSH 已完成，此处只是读侧节奏
    assert q.stats()["depth"] == 1
    got = q.get(timeout=2)
    assert got["req_id"] == "r1" and got["idx"] == 0
    assert "_q_ts" not in got                 # 用后即除：不污染下游载荷
    assert q.stats()["depth"] == 0            # 空闲必归零（无记账漂移）


def test_in_out_counters(q):
    """in/out = 服务端 INCR 累计（受理/交付口径），与服务端列表长度一致。"""
    for i in range(3):
        q.put_nowait({"i": i})
    assert q.stats()["in"] == 3 and q.stats()["out"] == 0
    for i in range(2):
        assert q.get(timeout=2)["i"] == i
    st = q.stats()
    assert st["in"] == 3 and st["out"] == 2 and st["depth"] == 1


def test_full_gate_put_nowait(q):
    """容量门：LLEN 判满 → put_nowait 立即 queue.Full；消费释放后可再入。"""
    for i in range(16):                       # cap=16
        q.put_nowait({"i": i})
    with pytest.raises(queue.Full):
        q.put_nowait({"over": 1})
    assert q.get(timeout=2)
    q.put_nowait({"after": 1})                # 释放 1 格后可入
    assert q.stats()["depth"] == 16


def test_full_gate_put_timeout(q):
    """put(timeout) 满时等待至超时抛 Full，不永久阻塞。"""
    for i in range(16):
        q.put_nowait({"i": i})
    t0 = time.monotonic()
    with pytest.raises(queue.Full):
        q.put({"over": 1}, timeout=0.3)
    assert time.monotonic() - t0 >= 0.25      # 确实等满了窗口


def test_bounded_get_empty(q):
    """get(timeout) 空队列到期抛 queue.Empty，不永久阻塞。"""
    t0 = time.monotonic()
    with pytest.raises(queue.Empty):
        q.get(timeout=0.2)
    assert 0.15 <= time.monotonic() - t0 <= 2.0


def test_poll_loop_wakes_on_put(q):
    """出队轮询循环：有界 get 等待中，put 到达即被取走（demand-pull 语义）。"""
    got = {}

    def worker():
        got["item"] = q.get(timeout=5)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    time.sleep(0.2)                           # worker 已进入轮询等待
    q.put_nowait({"wake": 1})
    t.join(timeout=3)
    assert not t.is_alive()
    assert got["item"]["wake"] == 1


def test_ema_first_sample(q):
    """avg_wait_ms EMA：首样本 = 实际等待（压着不取 ~150ms）。"""
    q.put_nowait({"e": 1})
    time.sleep(0.15)
    q.get(timeout=2)
    assert q.stats()["avg_wait_ms"] >= 100, q.stats()


def test_close_idempotent(q):
    """close 幂等：重复调用无害，之后 stats 仍可惰性重建连接。"""
    q.close()
    q.close()
    q.put_nowait({"after-close": 1})
    assert q.get(timeout=2)["after-close"] == 1


def test_spawn_pickle_roundtrip(fakeredis_client, monkeypatch):
    """spawn 传递：__getstate__ 只带键名元数据，子进程按需重建连接。"""
    import pickle

    q = pc.InstrumentedQueue(16, index=51, name="pk")
    state = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(q))))
    assert state._key == q._key and state._cap == q._cap
    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q.put_nowait({"spawn": 1})
    assert q.get(timeout=2)["spawn"] == 1
    q.close()


def test_tts_claim_put_nowait(fakeredis_client, monkeypatch):
    """TtsClaimQueue.put_nowait 回归（20261111 事故）：此类曾缺 put_nowait
    ——ship_batch 派发 asr_job 时 AttributeError 被降级路径吞成 500。
    FakeQueue 有 put_nowait 会掩盖，必须对真类锁定。"""
    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(2, index=55, name="pnw")
    q.put_nowait({"req_id": "a", "idx": 0, "ps": "x", "voice": "v",
                  "speed": 1.0, "origin": 0, "need": 1, "text": "t"})
    q.put_nowait({"req_id": "b", "idx": 0, "ps": "y", "voice": "v",
                  "speed": 1.0, "origin": 0, "need": 1, "text": "t"})
    with pytest.raises(queue.Full):
        q.put_nowait({"req_id": "c", "idx": 0, "ps": "z", "voice": "v",
                      "speed": 1.0, "origin": 0, "need": 1, "text": "t"})
    assert q.depth() == 2


def test_cancelled_set_batch_lookup(fakeredis_client, monkeypatch):
    """断连标记批量查询：单次 RTT 返回已取消子集；空集/未取消返回空。"""
    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(16, index=51, name="unit-cancel")
    pc.cancel_add("r-gone")
    try:
        assert q.cancelled_set(["r-gone", "r-alive", "r-gone"]) == {"r-gone"}
        assert q.cancelled_set(["r-alive"]) == set()
        assert q.cancelled_set([]) == set()
    finally:
        pc.shared_redis().zrem("kokoro:cancel", "r-gone")


# ── 逐条排队等待随消息带回（/tts 日志阶段分解的 q_* 口径）──

def test_get_injects_qw_ms(q):
    """LIST 队列 get：_q_ts 换算的逐条等待注入 _qw_ms，戳本身用后即除。"""
    q.put_nowait({"req_id": "r1"})
    got = q.get(timeout=2)
    assert isinstance(got.get("_qw_ms"), float) and got["_qw_ms"] >= 0.0
    assert "_q_ts" not in got


def test_external_item_without_stamp_gets_no_qw_ms(q):
    """外部直写/旧消息无 _q_ts 戳 → 不注入 _qw_ms（消费端按 0 兜底）。"""
    import pickle as _pickle
    q._redis().rpush(q._key, _pickle.dumps({"req_id": "r2"}))
    got = q.get(timeout=2)
    assert got["req_id"] == "r2" and "_qw_ms" not in got


def test_claim_injects_qw_ms(fakeredis_client, monkeypatch):
    """认领式队列：逐条等待随 chunk 回传（ship_batch 捎给网关），EMA 口径不变。"""
    monkeypatch.setattr(pc, "_make_redis_client", lambda: fakeredis_client)
    q = pc.TtsClaimQueue(16, index=53, name="qw")
    q.put_nowait({"req_id": "a", "idx": 0, "ps": "x", "voice": "v",
                  "speed": 1.0, "origin": 0, "need": 1, "text": "t"})
    claimed = q.claim(4, "tts-0-1")
    assert len(claimed) == 1
    _m, c = claimed[0]
    assert isinstance(c.get("_qw_ms"), float) and c["_qw_ms"] >= 0.0
    assert "_q_ts" not in c
    q.ack([_m], "tts-0-1")


def test_get_nowait_returns_queued_item(q):
    """get_nowait 必须先做一次 LPOP 再判 deadline（原实现必抛 Empty）。"""
    q.put_nowait({"x": 1})
    assert q.get_nowait()["x"] == 1
    with pytest.raises(queue.Empty):
        q.get_nowait()


def test_corrupt_message_is_dropped_not_raised(q, caplog):
    """坏 pickle：丢弃该条并继续，不能把 g2p/结果泵线程带走。"""
    q._redis().rpush(q._key, b"not-a-pickle")
    with pytest.raises(queue.Empty):
        q.get(timeout=0.2)
    assert q.stats()["out"] == 1
    # 后续消息仍可消费（worker 存活）
    q.put_nowait({"after": 1})
    assert q.get(timeout=2)["after"] == 1

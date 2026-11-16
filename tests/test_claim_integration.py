"""认领式传输的真实 Redis 深度集成测试（多进程并发/崩溃/重派/毒块）。

需要 redis-server：本地开发用 WSL 起 TCP 实例（端口 6399）或镜像内
（unix socket 直连）。Windows 无 redis-server 时 skip。"""
import multiprocessing as mp
import pickle
import struct
import shutil
import subprocess
import time

import pytest

import proc_common as pc
import redis_transport

_PORT = 6399
_IDX = 41          # 键位隔离：不与其他用例的 kokoro:tts 键串扰


def _redis_reachable() -> bool:
    """TCP 探活（不用 redis-py：它对连接失败会重试多次，skip 判定要快）。"""
    import socket

    s = socket.socket()
    s.settimeout(0.2)
    try:
        s.connect(("127.0.0.1", _PORT))
        return True
    except OSError:
        return False
    finally:
        s.close()


# 不在模块 import 期 skip：skipif 会在 fixture 之前生效，即使本机有
# redis-server 二进制也永远跑不到自起逻辑（镜像内 redis-server 是 unix
# socket + --port 0，6399 本就不通）。改为 fixture 内“已可达则复用，
# 否则有二进制就自起，否则 skip”。


def _redis():
    import redis

    return redis.Redis(host="127.0.0.1", port=_PORT, decode_responses=False)


@pytest.fixture()
def redis_proc():
    """已可达则复用外部实例（WSL/CI）；否则有 redis-server 二进制就自起；
    两者都没有才 skip（Windows 本地开发机的离线 fakeredis 用例仍覆盖）。"""
    proc = None
    if not _redis_reachable():
        if not shutil.which("redis-server"):
            pytest.skip("需要真实 redis-server（TCP 6399 或本机 redis-server 二进制）")
        proc = subprocess.Popen(
            ["redis-server", "--port", str(_PORT), "--save", "",
             "--appendonly", "no"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if _redis().ping():
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            proc.kill()
            pytest.skip("redis-server 自起后 10s 内未就绪（端口 %d 被占用？）" % _PORT)
    r = _redis()
    r.flushdb()
    yield proc
    try:
        r.flushdb()
    except Exception:  # noqa: BLE001  连接已被测试关闭：忽略
        pass
    if proc is not None:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


@pytest.fixture()
def cq(redis_proc, monkeypatch):
    monkeypatch.setattr(redis_transport, "_make_redis_client", lambda: _redis())
    return pc.TtsClaimQueue(64, index=_IDX)


def _chunk(rid="r", idx=0, n=30):
    return {"req_id": rid, "idx": idx, "ps": "ə" * n, "voice": "zf_001",
            "speed": 1.0, "origin": 0, "need": 1, "text": "句"}


def _claimer_worker(port, results_q, worker_id):
    """子进程认领者：认领 → ack → 回报 (req_id, idx) 对（按到达序）。"""
    import redis

    r = redis.Redis(host="127.0.0.1", port=port, decode_responses=False)
    redis_transport._make_redis_client = lambda: r
    q = pc.TtsClaimQueue(64, index=_IDX)
    wid = f"test-{worker_id}-9{worker_id}"   # role-i-pid 形态
    got = []
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and len(got) < 12:
        claimed = q.claim(4, wid)
        if not claimed:
            time.sleep(0.01)
            continue
        for m, c in claimed:
            got.append((c["req_id"], c["idx"]))
        q.ack([m for m, _c in claimed], wid)
    results_q.put((worker_id, got))


def test_concurrent_claim_exactly_once(redis_proc, cq):
    """3 个 worker 进程并发认领 12 条：全部恰好消费一次，零重复零丢失。"""
    ctx = mp.get_context("spawn")
    for rid in ("ra", "rb", "rc"):
        for idx in range(4):
            cq.put(_chunk(rid, idx))
    assert cq.depth() == 12

    results_q = ctx.Queue()
    procs = [ctx.Process(target=_claimer_worker,
                         args=(_PORT, results_q, w)) for w in range(3)]
    for p in procs:
        p.start()
    outs = [results_q.get(timeout=20) for _ in procs]
    for p in procs:
        p.join(timeout=5)

    pairs = sorted(p for _w, got in outs for p in got)
    assert len(pairs) == 12                            # 零丢失
    assert len(set(pairs)) == 12                       # 零重复
    assert pairs == sorted((rid, i) for rid in ("ra", "rb", "rc")
                           for i in range(4))          # 与投放完全对账
    assert cq.depth() == 0 and cq.stats()["in"] == 12


def _hold_and_die(holder_q):
    """认领整批后持握等待被 SIGKILL（模拟 worker 死亡）。"""
    import redis as _r

    r = _r.Redis(host="127.0.0.1", port=_PORT, decode_responses=False)
    redis_transport._make_redis_client = lambda: r
    q = pc.TtsClaimQueue(64, index=_IDX)
    claimed = q.claim(4, "tts-0-999")
    holder_q.put(len(claimed))
    time.sleep(60)


def test_worker_death_zero_loss_via_requeue(redis_proc, cq, monkeypatch):
    """worker 死亡 → requeue_worker 精确重派（秒级）→ 零永久丢失。
    BRPOPLPUSH 级追踪：重派条件是「worker 已死」而非「TTL 超时」。"""
    monkeypatch.setattr(redis_transport, "_make_redis_client", lambda: _redis())
    ctx = mp.get_context("spawn")

    for i in range(4):
        cq.put(_chunk("r", i))
    holder_q = ctx.Queue()
    p = ctx.Process(target=_hold_and_die, args=(holder_q,))
    p.start()
    assert holder_q.get(timeout=10) == 4
    assert cq.depth() == 0                              # 已出队
    assert _redis().zcard(f"{cq._key}:infl:tts-0-999") == 4   # per-worker 在途

    p.kill()
    p.join(timeout=3)

    n = cq.requeue_worker("tts-0-999")                  # 死亡精确重派
    assert n == 4
    assert cq.depth() == 4                              # 立即回队列（不等 TTL）
    claimed2 = cq.claim(4, "tts-1-1000")
    assert [c["idx"] for _m, c in claimed2] == [0, 1, 2, 3]   # 零丢失
    cq.ack([m for m, _c in claimed2], "tts-1-1000")



def test_poison_chunk_gives_up_after_max(redis_proc, cq):
    """确定性坏块：重派计数超上限 → 放弃（受影响请求 504），不再循环。"""
    cq.put(_chunk("bad", 0))

    # 模拟连续 5 轮"认领→死亡→sweeper 重派"：TTL=0（认领即过期）
    for i in range(5):
        claimed = cq.claim(4, f"tts-0-50{i}")
        assert len(claimed) == 1              # 每轮都被认领（毒块仍可取出）
        out = cq.sweep({500 + i}, time.time(), ttl=0.0, max_redispatch=4)
    # 5 轮 sweep：前 4 轮重派（计数 1-4 ≤ 上限），第 5 轮超限放弃
    assert out["gaveup"] == 1 and out["redispatched"] == 0
    # 放弃 = 从最后一个认领者的 per-worker 在途账本移除（毒块计数一并清）
    assert _redis().zcard(f"{cq._key}:infl:tts-0-504") == 0
    assert cq.depth() == 0                    # 队列不再收它（不再循环炸 worker）


def test_slow_worker_not_redispatched(redis_proc, cq):
    """F2/F3 根因消除的直接验证：慢 worker（持握 0.3s，未死亡）不被误
    重派——重派条件是「worker 已死」而非「TTL 超时」，重复前向/部分
    重叠聚合的触发面从"每次慢批"缩到"极端故障"。"""
    cq.put(_chunk("slow", 0))
    m1, c1 = cq.claim(4, "tts-0-777")[0]               # 慢 worker 认领
    time.sleep(0.3)                                    # 持握期间（<< ttl 兜底线）
    out = cq.sweep({777}, time.time(), ttl=5.0, max_redispatch=5)
    assert out["redispatched"] == 0                    # 存活 → 不重派
    assert cq.depth() == 0                             # 仍在其在途账本
    assert cq.ack([m1], "tts-0-777") == 1
    assert _redis().zcard(f"{cq._key}:infl:tts-0-777") == 0


def test_dead_worker_sweep_redispatches(redis_proc, cq):
    """sweep 的死亡判定路径：pid 不在存活集 → 全量重派（TTL 无关）。"""
    cq.put(_chunk("dead", 0))
    m1, _c = cq.claim(4, "tts-0-888")[0]
    out = cq.sweep(set(), time.time(), ttl=9999, max_redispatch=5)
    assert out["redispatched"] == 1                    # 死者立即重派
    assert cq.depth() == 1
    assert cq.ack([m1], "tts-1-999") == 0              # 已不在死者账本

def test_redispatch_preserves_fifo_order(redis_proc, cq):
    """深度审计 F4 回归：重派成员保持原 FIFO 位（score 复用 member 内
    序号）——曾用音素数作 score 导致重派 chunk 插队/最短优先。"""
    # 三个请求的 chunk：A(ps=30) B(ps=30) C(ps=200)，同到达序
    cq.put(_chunk("ra", 0, 30))
    cq.put(_chunk("rb", 0, 30))
    cq.put(_chunk("rc", 0, 200))
    # 认领并持握 rb（认领顺序：head=ra 类 30，claim 取 ra+rb）
    claimed = cq.claim(4, "tts-0-123")
    assert [c["req_id"] for _m, c in claimed] == ["ra", "rb"]
    # rb 所在批 TTL 兜底过期 → sweep 重派（score 应为原 seq，非音素数）
    infl_key = f"{cq._key}:infl:tts-0-123"
    for m, _c in claimed:
        _redis().zadd(infl_key, {m: time.time() - 999})
    out = cq.sweep({123}, time.time(), ttl=0.1, max_redispatch=5)
    assert out["redispatched"] == 2
    # 队列现按 FIFO：ra(seq1) rb(seq2) 在 rc(seq3) 之前
    scores = _redis().zrange(cq._key, 0, -1, withscores=True)
    order = [pickle.loads(m[pc._CLAIM_HEADER:])["req_id"] for m, _s in scores]
    assert order == ["ra", "rb", "rc"], (order, scores)   # F4 修复的直验
    cq.ack([m for m, _s in scores])


def test_ack_clears_redispatch_counter(redis_proc, cq):
    """:ratt 生命周期 = 重派窗口：成功交付的 ack 同批清计数。
    旧实现只 zrem，每次 worker 死亡重派都会在 hash 里留一条永久 member
    （容器生命周期内只增不减）。"""
    cq.put(_chunk("r-ack", 0))
    m, _c = cq.claim(4, "tts-0-321")[0]
    _redis().hincrby(cq._key_ratt, m, 1)       # 模拟一次死亡重派计数
    assert _redis().hlen(cq._key_ratt) == 1
    assert cq.ack([m], "tts-0-321") == 1
    assert _redis().hlen(cq._key_ratt) == 0
    assert _redis().zcard(f"{cq._key}:infl:tts-0-321") == 0


def test_requeue_worker_enforces_max_redispatch(redis_proc, cq):
    """worker 死亡主路径也计数：supervisor 在 respawn 时调用
    requeue_worker（此后账本已空、sweep 永远看不到这些 member），
    超限必须丢弃——否则"死亡→重派→再死"可无限循环炸整个 tts 池。"""
    cq.put(_chunk("poison", 0))
    assert len(cq.claim(4, "tts-0-1")) == 1
    assert cq.requeue_worker("tts-0-1", max_redispatch=1) == 1   # 计数 1 ≤ 1
    assert cq.depth() == 1
    assert len(cq.claim(4, "tts-0-2")) == 1
    assert cq.requeue_worker("tts-0-2", max_redispatch=1) == 0   # 计数 2 > 1 → 放弃
    assert cq.depth() == 0
    assert _redis().hlen(cq._key_ratt) == 0


def test_claim_drops_corrupt_member(redis_proc, cq):
    """真损坏 member（新旧两种头都解不出）：认领侧丢弃不炸 worker——
    否则每轮认领→崩→重派循环（put_nowait 事故同构）。旧 22B 格式
    member 不在此列（decode_member 单向兼容，见下个用例）。"""
    client = _redis()
    # 手工塞一个坏 member（合法 ZSET 成员、载荷错位）
    bad = (struct.pack(">H", 30) + b"0" + b"00000000000000001234"
           + b"\x80\x05corrupt")
    client.zadd(cq._key, {bad: 1234})
    claimed = cq.claim(4, "tts-0-7")            # 不抛 = worker 活着
    assert claimed == []                        # 坏 member 被丢弃
    assert client.zcard(cq._key) == 0           # 队列清掉（不再循环炸）
    assert client.zcard(f"{cq._key}:infl:tts-0-7") == 0  # 在途账本同步清
    assert cq.stats()["bad"] == 1               # 计数可见


def _legacy_member(rid="legacy", ps=30, seq=1234):
    """旧版本 member 格式：2B 音素数 + 20B 序号 + pickle（无精度字节）。"""
    chunk = {"req_id": rid, "idx": 0, "ps": "ə" * ps, "voice": "zf_001",
             "speed": 1.0, "origin": 0, "need": 1, "text": "句",
             "needs_align": False}
    return (struct.pack(">H", ps) + f"{seq:020d}".encode()
            + pickle.dumps(chunk))


def test_claim_accepts_legacy_22b_member(redis_proc, cq):
    """升级期单向兼容：新版本能消费旧 22B 头 member（无 dtype → 下游按
    fp32 处理），而不是丢弃让请求干等 WAIT_TIMEOUT 才 504。"""
    client = _redis()
    client.zadd(cq._key, {_legacy_member(): 1234})
    claimed = cq.claim(4, "tts-0-9")
    assert len(claimed) == 1
    m, chunk = claimed[0]
    assert chunk["req_id"] == "legacy" and chunk["dtype"] == "fp32"
    assert cq.stats()["bad"] == 0
    assert cq.ack([m], "tts-0-9") == 1


def test_requeue_worker_accepts_legacy_inflight(redis_proc, cq):
    """旧 worker 升级前认领的 legacy member 留在 infl 账本里：新版本
    精确重派时按旧头解出 seq 回队列（不被当坏块丢弃）。"""
    legacy = _legacy_member("legacy-infl", seq=4321)
    _redis().zadd(f"{cq._key}:infl:tts-0-7", {legacy: 1})
    assert cq.requeue_worker("tts-0-7") == 1
    assert cq.depth() == 1 and cq.stats()["bad"] == 0
    claimed = cq.claim(4, "tts-0-9")
    assert len(claimed) == 1 and claimed[0][1]["req_id"] == "legacy-infl"


def test_asr_queue_batch_claim_and_ack(redis_proc, cq):
    """asr 批推的队列侧前提：asr job（无 ps/dtype → member 头长度恒 0、
    精度恒 0）全员同凑批类——claim(4) 一次拿全批，整批一次 ack 清账本
    （compute.asr_main 的批循环依赖这两个语义）。"""
    import protocol

    for i in range(3):
        cq.put(protocol.asr_job("r", i, 0, need=1,
                                piece={"shm_name": "x", "offset": 0,
                                       "samples": 8},
                                text="句", sample_rate=24000))
    claimed = cq.claim(4, "asr-0-55")
    assert len(claimed) == 3                       # 同凑批类：一次全拿
    assert [c["idx"] for _m, c in claimed] == [0, 1, 2]
    assert cq.depth() == 0                         # 已全部出队
    members = [m for m, _c in claimed]
    assert cq.ack(members, "asr-0-55") == 3        # 整批一次 ack
    assert _redis().zcard(f"{cq._key}:infl:asr-0-55") == 0

"""Redis 传输跨进程崩溃安全（真实 redis-server；Windows 无 redis-server 时
skip，镜像内必跑）：消费者 SIGKILL → **队列内消息零丢失**（其他消费者接手），
仅其 in-flight（已 LPOP 未处理）1 条随死亡丢失——与文档化语义一致；
生产者全程不阻塞。"""
import multiprocessing as mp
import shutil
import subprocess
import time

import pytest

import proc_common as pc

_PORT = 6399


def _redis_reachable() -> bool:
    """TCP 探活（不用 redis-py：连接失败重试会让 skip 判定拖到十几秒）。"""
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


# 同 test_claim_integration：skip 放在用例内，先尝试复用/自起 6399。
# 模块级 skipif 会绕过“本机有 redis-server 二进制”的自起逻辑。


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _pop_one(port, key, signal_q):
    """消费者 A：弹出 1 条（in-flight）后持握等待被杀。"""
    import redis

    r = redis.Redis(host="127.0.0.1", port=port, decode_responses=False)
    item = r.lpop(key)
    signal_q.put(item is not None)
    time.sleep(3600)   # 持握在途消息，等待被 SIGKILL


def _drain(port, index, name, signal_q):
    """消费者 B：持续消费直到窗口关闭，回报消费条数（走 InstrumentedQueue
    真实 get 路径，验证 wrapper 在独立进程内的行为）。"""
    import redis

    r = redis.Redis(host="127.0.0.1", port=port, decode_responses=False)
    pc._make_redis_client = lambda: r      # 子进程内替换工厂（wrapper 惰性连接）
    q = pc.InstrumentedQueue(64, index=index, name=name)
    n = 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            q.get(timeout=0.2)
            n += 1
        except Exception:  # noqa: BLE001  Empty（队列已空）或 redis 瞬断重试
            pass
    signal_q.put(n)


def test_consumer_killed_zero_queue_loss():
    port = _PORT
    redis_proc = None
    if not _redis_reachable():
        if not shutil.which("redis-server"):
            pytest.skip("需要真实 redis-server（TCP 6399 或本机 redis-server 二进制）")
        redis_proc = subprocess.Popen(       # 本地有二进制则自起（镜像内）
            ["redis-server", "--port", str(port), "--save", "",
             "--appendonly", "no"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import redis

        r = redis.Redis(host="127.0.0.1", port=port, decode_responses=False,
                        socket_connect_timeout=1.0)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if r.ping():
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        else:
            pytest.skip("redis-server 10s 内未就绪（端口 %d 被占用？）" % port)
        pc._make_redis_client = lambda: redis.Redis(
            host="127.0.0.1", port=port, decode_responses=False)

        ctx = mp.get_context("spawn")
        q = pc.InstrumentedQueue(64, index=41, name="crash-test")
        signal_q = ctx.Queue()
        for i in range(20):
            q.put_nowait({"i": i})                # 全部瞬时入队（生产者不阻塞）

        a = ctx.Process(target=_pop_one, args=(port, q._key, signal_q))
        a.start()
        assert signal_q.get(timeout=5) is True    # A 已持有 1 条在途
        a.kill()                                  # SIGKILL：in-flight 1 条随死亡丢失
        a.join(timeout=3)

        b = ctx.Process(target=_drain, args=(port, 41, "crash-test", signal_q))
        b.start()
        b_count = signal_q.get(timeout=15)
        b.join(timeout=3)

        # 队列内 19 条零丢失、全部被 B 接手；A 的 in-flight 1 条随死亡丢失
        assert b_count == 19
        assert r.llen(q._key) == 0                # 最终排空
    finally:
        if redis_proc is not None:
            redis_proc.kill()

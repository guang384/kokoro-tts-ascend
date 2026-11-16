"""g2p → tts 认领队列投递失败：必须发 synth_error + cancel，不能只打日志。

原实现把 for c in chunks: tts_queue.put(c) 放在 process 外层 try 里，
部分 chunk 已入队后失败只记一行日志——网关等缺的 idx 到 120s 才 504。
"""
import proc_common
import protocol
from g2p import publish_chunks
from tests.fake_queue import FakeQueue


def _chunk(i):
    return protocol.chunk("r-pub", i, "ps", "zf_001", 1.0, 0, 2, "句",
                          needs_align=False)


class _FlakyQueue:
    def __init__(self, fail_at=2):
        self.n = 0
        self.fail_at = fail_at

    def put(self, c):
        self.n += 1
        if self.n == self.fail_at:
            raise RuntimeError("redis down")


def test_publish_chunks_success():
    q = FakeQueue()
    req = {"req_id": "r-pub", "origin": 0}
    log = proc_common.setup_logging("test-g2p-pub")
    assert publish_chunks([_chunk(0), _chunk(1)], req, q, [FakeQueue()], log) is True
    assert q.qsize() == 2


def test_publish_chunks_partial_failure_sends_error_and_cancel():
    q = _FlakyQueue(fail_at=2)      # 第 1 条已入队，第 2 条失败
    gq = FakeQueue()
    req = {"req_id": "r-pub", "origin": 0}
    log = proc_common.setup_logging("test-g2p-pub")
    assert publish_chunks([_chunk(0), _chunk(1)], req, q, [gq], log) is False

    msg = gq.get_nowait()
    assert msg["type"] == protocol.SYNTH_ERROR
    assert msg["req_id"] == "r-pub"
    assert msg["kind"] == protocol.KIND_COMPUTE
    # 已入队的第 1 条由 tts 认领侧按 cancel 标记过滤
    assert proc_common.is_cancelled("r-pub") is True
    proc_common.shared_redis().zrem("kokoro:cancel", "r-pub")

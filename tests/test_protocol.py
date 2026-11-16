"""跨进程消息协议一致性：构造函数字段 ↔ 消费者读取字段逐项对齐。

历史上 g2p/batcher/compute 三端各自手写字典导致字段漂移（missing
need/req_id、origin 取错下标），整条管道断裂——
本文件把"每一端到底读什么"写成显式断言，协议再漂移立即红。
"""
import protocol
from g2p import make_chunks


# ── 各消费端实际读取的字段清单（与代码逐行对应，故意重复声明）──
G2P_QUEUE_CONSUMER_FIELDS = {"req_id", "origin", "text", "voice", "speed",
                             "needs_align",
                             "dtype"}    # needs_align 信息性（gateway 用 _request_meta 决定）；dtype 由 make_chunks 透传进 chunk
TTS_CHUNK_FIELDS = {"req_id", "idx", "ps", "voice", "speed", "origin",
                    "need", "text", "needs_align", "dtype",
                    "sents"}    # compute.ship_batch（认领式组批；精度类键；
                                # sents=合并句数——chunk 行展示字段）
SYNTH_PART_CONSUMER_FIELDS = {"req_id", "need", "block"}              # gateway._agg_add
BLOCK_SEG_FIELDS = {"idx", "offset", "samples", "text"}               # gateway 视图构建
ASR_JOB_FIELDS = {"type", "req_id", "idx", "origin", "need", "piece",
                  "text", "sample_rate", "max_group_sec"}     # compute.asr_main
CHUNK_TS_FIELDS = {"type", "req_id", "idx", "need", "samples",
                   "timestamp", "error"}                       # gateway._on_chunk_ts
SYNTH_ERROR_CONSUMER_FIELDS = {"req_id", "error", "kind"}             # gateway._on_synth_error


def test_chunk_fields_match_consumer():
    c = protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 1, 3, "你好",
                       needs_align=True, sents=1)
    assert set(c) == TTS_CHUNK_FIELDS


def test_new_request_fields_match_consumer():
    r = protocol.new_request("r1", 0, "你好", "zf_001", 1.0, True)
    assert G2P_QUEUE_CONSUMER_FIELDS <= set(r)


def test_make_chunks_propagates_dtype():
    """g2p 是 chunk 唯一生产者：new_request.dtype 必须透传到 chunk
    （缺省 fp32），否则请求级精度在 g2p 一跳丢失。"""
    req = protocol.new_request("r1", 0, "你好", "zf_001", 1.0, False,
                               dtype="fp16")
    chunks = make_chunks(req, ["你好"], [["ə"]])
    assert len(chunks) == 1 and chunks[0]["dtype"] == "fp16"
    req32 = protocol.new_request("r2", 0, "你好", "zf_001", 1.0, False)
    assert make_chunks(req32, ["你好"], [["ə"]])[0]["dtype"] == "fp32"


def test_synth_part_fields_match_consumer():
    block = {"shm_name": "x", "segs": [{"idx": 0, "offset": 0, "samples": 8,
                                        "text": "你好"}]}
    m = protocol.synth_part("r1", 1, block)
    assert SYNTH_PART_CONSUMER_FIELDS <= set(m)
    assert m["n"] == 1
    for seg in m["block"]["segs"]:
        assert BLOCK_SEG_FIELDS <= set(seg)


def test_asr_job_and_chunk_ts_fields():
    job = protocol.asr_job("r1", 0, 1, 3, {"shm_name": "x", "offset": 0,
                                          "samples": 8}, "你好", 24000)
    assert ASR_JOB_FIELDS <= set(job)
    assert set(job["piece"]) == {"shm_name", "offset", "samples"}
    ts = protocol.chunk_ts("r1", 0, 3, 8, [[0, 1]])
    assert set(ts) == CHUNK_TS_FIELDS


def test_synth_error_fields():
    assert SYNTH_ERROR_CONSUMER_FIELDS <= set(
        protocol.synth_error("r1", "err", protocol.KIND_COMPUTE))



def test_all_payloads_picklable_as_queue_data():
    """回归（生产事故）：队列句柄作为数据 pickle 会炸（"Queue objects
    should only be shared between processes through inheritance"）。
    全部协议消息必须可被 _ForkingPickler 序列化——消息入 Redis 前要
    pickle.dumps，spawn 传参走 ForkingPickler，两条真实路径共用该约束。"""
    from multiprocessing.reduction import ForkingPickler

    msgs = [
        protocol.new_request("r1", 0, "你好", "zf_001", 1.0, False),
        protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 0, 1, "你好"),
        protocol.synth_error("r1", "err", protocol.KIND_COMPUTE),
        protocol.synth_part("r1", 1, {"shm_name": "x", "segs": [
            {"idx": 0, "offset": 0, "samples": 8, "text": "你好"}]}),
        protocol.asr_job("r1", 0, 1, 3, {"shm_name": "x", "offset": 0,
                                        "samples": 8}, "你好", 24000),
        protocol.chunk_ts("r1", 0, 3, 8, [[0, 1]]),
    ]
    for m in msgs:
        ForkingPickler.dumps(m)  # 生产事故即在此炸：align_group 曾携带队列句柄


# ── g2p chunk 戳语义（need/bucket 的历史断裂点）──

def _req(req_id="r1", origin=1):
    return {"req_id": req_id, "origin": origin, "text": "你好。",
            "voice": "zf_001", "speed": 1.0, "needs_align": False}


def test_g2p_need_is_total_not_index():
    """need 必须是请求 chunk 总数（历史上被截成运行下标 0,1,2… → 多批请求截断）。"""
    chunks = make_chunks(_req(), ["第一句。", "第二句。", "第三句。"],
                         [["ps1a", "ps1b"], ["ps2"], ["ps3"]])
    assert len(chunks) == 4
    assert [c["idx"] for c in chunks] == [0, 1, 2, 3]
    assert all(c["need"] == 4 for c in chunks)  # 不是 idx！
    assert all(c["origin"] == 1 for c in chunks)





def test_g2p_chunks_keep_text_for_alignment():
    chunks = make_chunks(_req(), ["第一句。"], [["ps1"]])
    assert chunks[0]["text"] == "第一句。"


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_protocol.py")


def test_put_result_drop_counter():
    """put_result 投递失败（入口积压）→ 返回 False 且丢弃计数 +1：
    丢弃是 at-most-once 设计内的受控行为，必须可观测（/health workers[].err）。"""
    from tests.fake_queue import FakeQueue

    class _Full(FakeQueue):
        def put(self, item, timeout=None):
            raise RuntimeError("mailbox full")

    before = protocol.drop_count()
    assert protocol.put_result(_Full(maxsize=1), {"req_id": "r"}, log=None) is False
    assert protocol.drop_count() == before + 1
    protocol.note_drop()
    assert protocol.drop_count() == before + 2


# ── 阶段耗时随行（/tts 日志分解；全部可选，缺省不落消息）──

def test_timing_fields_optional_not_serialized_by_default():
    """不传耗时不落消息（旧格式字节级不变，消息键集契约锁死）；传了才带上。"""
    c = protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 0, 1, "你好")
    assert "q_g2p_ms" not in c and "g2p_ms" not in c
    c2 = protocol.chunk("r1", 0, "ə", "zf_001", 1.0, 0, 1, "你好",
                        q_g2p_ms=12.5, g2p_ms=88.0)
    assert c2["q_g2p_ms"] == 12.5 and c2["g2p_ms"] == 88.0

    block = {"shm_name": "x", "segs": [{"idx": 0, "offset": 0, "samples": 8,
                                        "text": "你好"}]}
    assert "timings" not in protocol.synth_part("r1", 1, block)
    sp = protocol.synth_part("r1", 1, block, timings={
        "q_g2p": 12.5, "g2p": 88.0, "q_tts": 3.0, "tts": 810.0})
    assert sp["timings"] == {"q_g2p": 12.5, "g2p": 88.0,
                             "q_tts": 3.0, "tts": 810.0}

    assert "timings" not in protocol.chunk_ts("r1", 0, 3, 8, [[0, 1]])
    ct = protocol.chunk_ts("r1", 0, 3, 8, [[0, 1]],
                           timings={"q_asr": 1.0, "asr": 250.0})
    assert ct["timings"] == {"q_asr": 1.0, "asr": 250.0}


def test_make_chunks_propagates_g2p_timings():
    """g2p 整单一次算：每条 chunk 携带同值，网关按 max 聚合（不随 chunk 数
    翻倍）；不传 → 缺省不落消息。"""
    req = protocol.new_request("r1", 0, "你好", "zf_001", 1.0, False)
    chunks = make_chunks(req, ["一。", "二。"], [["ps1"], ["ps2"]],
                         q_g2p_ms=7.5, g2p_ms=120.0)
    assert len(chunks) == 2
    assert all(c["q_g2p_ms"] == 7.5 and c["g2p_ms"] == 120.0 for c in chunks)
    bare = make_chunks(req, ["一。"], [["ps1"]])
    assert "q_g2p_ms" not in bare[0] and "g2p_ms" not in bare[0]

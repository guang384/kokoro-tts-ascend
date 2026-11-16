"""全链路 dry-run：stub 掉模型（G2P/前向/FunASR），但跑真实的
g2p.make_chunks → TtsClaimQueue 认领组批 → compute.ship_batch（真 SHM
写入）→ gateway 聚合 → /v1 与 /tts 两条终态路径。

这是防"协议断裂"回归的最强一档：任何一个字段/路由/生命周期问题，
在这条真组件链路上都会暴露。"""
import asyncio
import threading
import time

import numpy as np
import pytest

import compute
import gateway
import gateway_state
import gateway_agg
import protocol
from gateway import BizError, TTSRequest, _synth, tts_business
from g2p import make_chunks
from tests.fake_queue import FakeQueue, Runner


class StubCompute:
    """synth_batch 替身：每句 800 采样，幅度=(句序号+1)×0.125（int16 安全域
    内的精确二进制值，PCM16 线格式量化后仍可逐句区分，跨批可断言）。"""
    device = "cpu"
    samples_per_sent = 800

    def __init__(self):
        self.dtypes = []          # 每次前向的 dtype（fp16 全链路断言用）

    def synth_batch(self, jobs, dtype="fp32"):
        self.dtypes.append(dtype)
        # 真实前向有可测墙钟；留 10ms 让 /tts done 的 tts 字段非零，
        # 否则阶段耗时断言只能验证"字段存在"，全 0 也能通过。
        time.sleep(0.01)
        # 新契约：直出 int16（量化已在卡上完成——旧 float 幅度 ×32767
        # 取整后的精确值，round 半到偶与 torch.round 一致）
        n = self.samples_per_sent
        return [np.full(n, round((int(ps[2:]) + 1) * 0.125 * 32767),
                        dtype="<i2") for ps, _v, _s in jobs]


def _ps_lists(sentences):
    return [[f"ps{i}"] for i, _s in enumerate(sentences)]


@pytest.fixture()
def pipeline(monkeypatch):
    """搭好全部队列 + 后台线程（g2p 喂食器 / tts 认领消费者）。

    认领式管线：g2p 喂食器把 chunk 直推 TtsClaimQueue；tts 消费者按
    「队列快照」认领组批。tts_gate 事件保证确定性：喂食器推完一个请求
    的全部 chunk 才放行消费者认领——批组成不与推送节奏竞速。"""
    import fakeredis

    import proc_common
    import redis_transport

    server = fakeredis.FakeServer()
    client = fakeredis.FakeStrictRedis(server=server, decode_responses=False)
    monkeypatch.setattr(redis_transport, "_make_redis_client", lambda: client)

    q = {
        "g2p": FakeQueue(), "tts": proc_common.TtsClaimQueue(
            64, index=60, name="tts"),
        "gateways": [FakeQueue(), FakeQueue()],
        "asr": FakeQueue(),
        "tts_gate": threading.Event(),
        "reqs": [],                  # 原始请求留档（校验网关透传字段）
    }
    runners = []

    # g2p 喂食器：拉原始请求 → 真实 make_chunks → 直推认领队列 → 放行消费者
    def g2p_stub():
        while True:
            req = q["g2p"].get()
            q["reqs"].append(req)
            for c in protocol_chunks_source(req):
                q["tts"].put(c)
            q["tts_gate"].set()

    # tts 消费者：认领→组批→真 compute.ship_batch（stub 前向，真 SHM 写入）。
    # 前向替身经 q["stub"] 可替换（测试中途换替身而非加第二个消费者，
    # 避免同一队列两个消费者的非确定性路由）
    q["stub"] = StubCompute()

    def tts_stub():
        import proc_common

        log = proc_common.setup_logging("dryrun-tts")
        from multiprocessing import shared_memory

        batch_no = 0
        while True:
            q["tts_gate"].wait()
            q["tts_gate"].clear()
            # 让「首个 chunk 入队 → 最后一个 chunk 认领」的跨度可测
            # （纯本地 fakeredis 下 put/claim 可能亚毫秒，日志会显示 0.00）
            time.sleep(0.02)
            while True:                     # 排空队列快照（按 MAX_BATCH 认领）
                claimed = q["tts"].claim(4)
                if not claimed:
                    break
                members = [m for m, _c in claimed]
                for _m, c in claimed:
                    # 真实队列有排队等待；保底 2ms 让 chunk 行的 queue_ms
                    # 非零（否则全 0 也能通过字段正则）。
                    c["_qw_ms"] = max(float(c.get("_qw_ms") or 0.0), 2.0)
                batch_no += 1
                batch = {"batch_id": f"w0-{batch_no}",   # 与生产 bid 格式对齐（t=TTS）
                         "chunks": [c for _m, c in claimed]}
                compute.ship_batch(q["stub"], batch, q["gateways"],
                                   q["asr"], log, np, shared_memory)
                q["tts"].ack(members)

    runners.append(Runner(g2p_stub).start())
    runners.append(Runner(tts_stub).start())

    # asr 工人替身：按名 attach SHM → 恒定时间戳 → 回投
    runners.append(Runner(asr_stub, q["asr"], q["gateways"]).start())

    monkeypatch.setattr(gateway_state, "_pending", {})
    monkeypatch.setattr(gateway_state, "_request_meta", {})
    monkeypatch.setattr(gateway_state, "_agg_state", {})
    monkeypatch.setattr(gateway_state, "_g2p_queue", q["g2p"])
    monkeypatch.setattr(gateway_state, "_entry_wait_s", 10)
    monkeypatch.setattr(gateway, "_check_voice", lambda voice: None)
    monkeypatch.setattr(gateway, "_asr_ready", lambda: True)
    return q


_SENTENCES = ["第一句。", "第二句。", "第三句。", "第四句。", "第五句。", "第六句。"]


def protocol_chunks_source(req):
    """g2p 池主体（切句/G2P 用固定替身，盖章逻辑是真的）。"""
    speed = float(req["speed"])
    return make_chunks(dict(req, speed=speed), _SENTENCES, _ps_lists(_SENTENCES),
                       q_g2p_ms=10.0, g2p_ms=20.0)


def asr_stub(asr_q, gateways):
    """asr 工人替身：chunk 级流水——收 ASR_JOB（单 chunk）→ 校验 SHM 可读
    → CHUNK_TS 直投 origin 网关（恒定时间戳 [[0, 50]]，拼接偏移由 TS 账本
    按前序采样数计算）。timings 携带真实契约的跨度戳（认领/桶/完成）：
    20ms 间隔让 /tts done 的 asr 墙钟跨度在 %.2f 格式下非零。"""
    from multiprocessing import shared_memory

    while True:
        job = asr_q.get()
        piece = job["piece"]
        shm = shared_memory.SharedMemory(name=piece["shm_name"])
        try:
            # bytes() 拷贝立即释放 memoryview（裸切片的视图存活到 close 会
            # BufferError："cannot close exported pointers exist"）
            _ = bytes(shm.buf[piece["offset"]:piece["offset"] + piece["samples"] * 2])
        finally:
            shm.close()
        t_claim = time.time()
        time.sleep(0.02)   # asr 墙钟跨度（认领→完成）可测，否则 0.00
        gateways[job["origin"]].put(protocol.chunk_ts(
            job["req_id"], job["idx"], job["need"], piece["samples"],
            [[0, 50]], timings={"q_asr_first": t_claim, "q_asr_last": t_claim,
                                "asr_done": time.time(),
                                "asr_bucket": t_claim}))


async def _run_with_pump(coro_factory, result_q):
    """生产路径同款：结果泵线程 + call_soon_threadsafe 回投事件循环。"""
    gateway_state._main_loop = asyncio.get_running_loop()
    pump = threading.Thread(target=gateway_agg._result_pump, args=(result_q,),
                            daemon=True)
    pump.start()
    return await coro_factory()


def test_v1_end_to_end_multi_batch(pipeline):
    """6 句请求 → 2 个批次（4+2）→ gateway 收齐合成完整音频。"""
    q = pipeline
    audio, texts, fw, _rid, blocks, _timings, _agg_s = asyncio.run(_run_with_pump(
        lambda: _synth("六句话的请求", "zf_001", 1.0, needs_align=False),
        q["gateways"][0]))
    # 拼接：6×800 采样 + 5×60ms(1440) 间隙
    assert len(audio) == 6 * 800 + 5 * 1440
    assert texts == _SENTENCES
    assert fw == 2                       # 两批前向
    assert len(blocks) == 2              # 批内混批 → 每批每请求一块
    # 幅度检查：第 1 句 ≈ 1.0，第 6 句 ≈ 6.0（跨批正确排序）
    # /v1 返回 PCM16 int16（端到端直通）：0.125 步进 ×32767 取整
    assert audio.dtype == np.int16
    assert audio[0] == round(0.125 * 32767)
    assert audio[800 + 1440] == round(0.25 * 32767)
    assert audio[-1] == round(0.75 * 32767)
    # 无泄漏
    assert gateway_state._agg_state == {} and gateway_state._request_meta == {}


def test_v1_end_to_end_fp16_dtype(pipeline):
    """请求级 dtype=fp16 全链路：gateway → 真 make_chunks（chunk 携带
    dtype）→ 认领队列（member 精度字节）→ ship_batch → synth_batch
    (dtype="fp16")。锁住任何一跳丢失精度都不会静默按 fp32 跑。"""
    q = pipeline
    q["stub"].dtypes.clear()
    (audio, texts, _fw, _rid, _blocks,
     _timings, _agg_s) = asyncio.run(_run_with_pump(
        lambda: _synth("六句话的请求", "zf_001", 1.0, needs_align=False,
                       dtype="fp16"),
        q["gateways"][0]))
    assert texts == _SENTENCES and audio.dtype == np.int16 and len(audio) > 0
    assert q["stub"].dtypes and set(q["stub"].dtypes) == {"fp16"}


def test_tts_end_to_end_with_alignment(pipeline):
    """/tts：chunk 级流水（前向后派 asr job）→ TS 账本拼回 → 业务契约响应。"""
    q = pipeline
    resp = asyncio.run(_run_with_pump(
        lambda: tts_business(TTSRequest(text="六句话的请求", speed=1.0)),
        q["gateways"][0]))
    assert resp.status_code == 200
    import json

    d = json.loads(resp.body)
    assert d["code"] == 0 and d["message"] == "success"
    assert d["text"] == "六句话的请求"           # 契约字段
    # 对外 6 键契约（g2p 兼容别名已移除）
    assert set(d) == {"audio_base64", "code", "duration", "message",
                      "text", "timestamp"}
    # chunk 级流水拼接：6 chunk 各产 [[0,50]]，偏移=前序采样数换算
    # （800 采样 @24kHz ≈ 33ms/chunk，round 累加）
    assert d["timestamp"] == [[0, 50], [93, 143], [187, 237],
                              [280, 330], [373, 423], [467, 517]]
    assert d["duration"] > 0.4
    assert gateway_state._agg_state == {} and gateway_state._request_meta == {}
    assert gateway_state._pending == {}


def test_tts_done_line_stage_fields(pipeline, caplog):
    """日志重设计 v12：done 行只带网关独有字段（ep/req/voice/speed/
    audio_s/e2e_s/agg_ms/enc_ms）；批/句级明细由 TTS 批头行（bid/n/
    process_ms/audio_s/dt）+ chunk 行（chunk/bid/queue_ms/audio_s）承担
    ——bid 关联，chunk 键 {req}_{need}_{idx}。"""
    import logging
    import re

    q = pipeline
    with caplog.at_level(logging.INFO):   # 根级别：worker 线程的批头/chunk 行也进 caplog
        resp = asyncio.run(_run_with_pump(
            lambda: tts_business(TTSRequest(text="六句话的请求", speed=1.0)),
            q["gateways"][0]))
    assert resp.status_code == 200
    logs = [r.getMessage() for r in caplog.records]
    done = [x for x in logs if "ep=/tts" in x and "e2e_s=" in x]
    assert len(done) == 1, logs
    m = re.search(r"ep=/tts req=(\S+) voice=(\S+) speed=(\S+) "
                  r"audio_s=(\d+\.\d+) e2e_s=(\d+\.\d+) "
                  r"agg_ms=\d+ enc_ms=\d+", done[0])
    assert m, f"done 行字段缺失: {done[0]}"
    e2e = float(m.group(4))
    assert e2e > 0
    # TTS 批头行 + chunk 行（bid 关联；同一请求的 chunk 键前缀一致）
    hdr = [x for x in logs
           if re.search(r"bid=\S+ n=\d+ process_ms=[\d.]+ audio_s=[\d.]+ dt=", x)]
    chs = [x for x in logs
           if re.search(r"chunk=\S+_\d+_\d+ bid=\S+", x) and "queue_ms=" in x]
    assert hdr, "TTS 批头行缺失"
    assert len(chs) == 6, f"chunk 行数量（6 chunk）: {len(chs)}"
    rids = {x.split("chunk=")[1].split("_")[0] for x in chs}
    assert len(rids) == 1, "同一请求的 chunk 键前缀应一致"
    for x in chs:
        assert "bid=" in x and "queue_ms=" in x and "audio_s=" in x \
            and "process_ms=" in x


def test_tts_deadline_shared_across_stages(pipeline, monkeypatch):
    """单一截止线：音频段耗掉预算大头（2 批×0.12s/0.3s）后，对齐段只有
    剩余额度 → 504 总时长 ≈ WAIT_TIMEOUT_S 而非 2×WAIT（旧实现音频/对齐
    各一整段，本用例在旧实现下需 ~0.55s+，与 0.45s 断言线可区分）。"""
    import time as _time

    q = pipeline

    class SlowStub(StubCompute):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def synth_batch(self, jobs, dtype="fp32"):
            self.calls += 1
            _time.sleep(0.12)
            return super().synth_batch(jobs, dtype=dtype)

    slow = SlowStub()
    q["stub"] = slow
    monkeypatch.setattr(gateway_state, "_entry_wait_s", 0.3)
    q["asr"] = FakeQueue()   # asr 替身消费原队列对象 → 新作业无人消费，fut_ts 永不完成
    t0 = _time.perf_counter()
    with pytest.raises(BizError) as ei:
        asyncio.run(_run_with_pump(
            lambda: tts_business(TTSRequest(text="六句话的请求", speed=1.0)),
            q["gateways"][0]))
    elapsed = _time.perf_counter() - t0
    assert ei.value.status_code == 504
    # 音频段必须真的跑过（2 批 × 0.12s）——否则“立即 504”也能满足上界
    assert slow.calls == 2, f"音频批次数 {slow.calls}（期望 2）"
    assert 0.20 <= elapsed < 0.45, (
        f"总时长 {elapsed:.3f}s：截止线未共享或音频段未执行")
    assert gateway_state._agg_state == {} and gateway_state._pending == {}
    assert gateway_state._request_meta == {}


def test_long_request_groups_by_time(pipeline):
    """长请求的 chunk 级偏移累加（8s/句 + 60ms gap）。

    注意：本文件的 asr 替身不做真实切组；组切分数学由
    tests/test_funasr_math.py 的 _split_groups/_split_part 用例单独锁定。
    这里验证的是网关 TS 账本按前序采样数累加偏移（>30s 请求不漂移）。"""
    q = pipeline

    class LongStub(StubCompute):
        samples_per_sent = 24000 * 8  # 每句 8s → 6 句 48s → 4 句/组切 2 组

    q["stub"] = LongStub()
    resp = asyncio.run(_run_with_pump(
        lambda: tts_business(TTSRequest(text="长音频请求", speed=1.0)),
        q["gateways"][0]))
    import json

    d = json.loads(resp.body)
    ts = d["timestamp"]
    assert len(ts) == 6
    # chunk 级流水：每 chunk 8s + 60ms gap → 偏移 8060ms/chunk
    assert ts[4][0] == 32240
    assert ts[5] == [40300, 40350]
    assert all(e >= s for s, e in ts)


def test_request_id_echoed_on_both_endpoints(pipeline):
    """X-Request-ID 响应头贯通 /v1 与 /tts——客户端报障凭此对齐各 worker
    日志的 req= 行。"""
    import re

    from gateway import SpeechRequest, create_speech

    q = pipeline
    resp = asyncio.run(_run_with_pump(
        lambda: tts_business(TTSRequest(text="请求标识", speed=1.0)),
        q["gateways"][0]))
    assert re.fullmatch(r"[0-9a-f]{12}", resp.headers["X-Request-ID"])

    resp2 = asyncio.run(_run_with_pump(
        lambda: create_speech(SpeechRequest(input="请求标识二", speed=1.0)),
        q["gateways"][0]))
    assert re.fullmatch(r"[0-9a-f]{12}", resp2.headers["X-Request-ID"])
    assert resp2.headers["X-Request-ID"] != resp.headers["X-Request-ID"]


def test_gateway_forwards_g2p_overrides(pipeline):
    """请求级多音字覆盖：网关校验/规范化后原样进 g2p_queue（本测试的
    G2P 替身不消费读音，只锁协议透传——真实生效见 test_g2p_overrides）。"""
    q = pipeline
    resp = asyncio.run(_run_with_pump(
        lambda: tts_business(TTSRequest(
            text="六句话的请求", speed=1.0,
            g2p_overrides={"长乐": "zhang3 le4"})),
        q["gateways"][0]))
    assert resp.status_code == 200
    assert q["reqs"][-1]["g2p_overrides"] == {"长乐": ["zhang3", "le4"]}


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_dryrun.py")

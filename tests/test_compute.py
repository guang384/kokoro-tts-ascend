"""compute.ship_batch 混批语义：批内跨请求时每请求各自的 need/分块/排序。

历史 bug：need 取批内第一条 chunk 的值通提全批——混批时长单被截断、
短单被拖到超时。本文件用随机形状（固定 seed）直接驱动 ship_batch。"""
import random

import pytest
from multiprocessing import shared_memory

import numpy as np

import compute
import proc_common
import protocol
from tests import rnd
from tests.fake_queue import FakeQueue


class _PosStub:
    """synth_batch 替身：按批内位置返回幅度 4096×(pos+1) 的 800 采样 int16
    （新契约：fade/量化已在卡上完成，synth_batch 直出 int16——原 0.125
    float 步进 ×32767 量化后的精确值，逐位置可区分）。"""

    device = "cpu"
    samples = 800

    def synth_batch(self, jobs, dtype="fp32"):
        return [np.full(self.samples, round((pos + 1) * 0.125 * 32767),
                        dtype="<i2") for pos, _job in enumerate(jobs)]


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:  # noqa: BLE001
            return out


def test_ship_batch_mixed_requests_carry_own_need():
    """两个请求（句数随机且不同）混入一批：每条 synth_part 携带各自 need，
    分块按句序、幅度映射正确（批内位置 ↔ 音频内容）。"""
    rng = random.Random(97531)
    n_a, n_b = rnd.distinct_ints(rng, 1, 6, 2)
    chunks = []
    pos_of = {}
    for rid, n in (("reqA", n_a), ("reqB", n_b)):
        for i in range(n):
            chunks.append(protocol.chunk(
                req_id=rid, idx=i, ps=f"ps{i}", voice="zf_001", speed=1.0,
                origin=0, need=n, text=rnd.sentence(rng)))
    rng.shuffle(chunks)  # 批内请求间顺序随机（混批常态）
    for pos, c in enumerate(chunks):
        pos_of[(c["req_id"], c["idx"])] = pos

    gq = FakeQueue()
    log = proc_common.setup_logging("test-tts")
    compute.ship_batch(_PosStub(), {"batch_id": "b-1", "chunks": chunks},
                       [gq], FakeQueue(), log, np, shared_memory)

    parts = _drain(gq)
    by_rid = {p["req_id"]: p for p in parts}
    assert set(by_rid) == {"reqA", "reqB"}
    for rid, n in (("reqA", n_a), ("reqB", n_b)):
        p = by_rid[rid]
        assert p["need"] == n, (
            f"{rid} 的 need 必须是该请求句数 {n}，而非批首请求的值")
        assert [s["idx"] for s in p["block"]["segs"]] == list(range(n))
    # SHM 内容：段 idx 的幅度 = 该 chunk 在批内的位置+1（排序与映射双重验证）
    for rid, n in (("reqA", n_a), ("reqB", n_b)):
        shm = shared_memory.SharedMemory(name=by_rid[rid]["block"]["shm_name"])
        try:
            for seg in by_rid[rid]["block"]["segs"]:
                view = np.frombuffer(
                    shm.buf[seg["offset"]:seg["offset"] + seg["samples"] * 2],
                    dtype="<i2")
                amp = int(view[0])
                del view  # 释放导出指针再 close（mmap 语义）
                # PCM16 直写：幅度 (pos+1)×0.125 → round(×32767) = (pos+1)×4096
                assert amp == 4096 * (pos_of[(rid, seg["idx"])] + 1)
        finally:
            shm.close()
            try:
                shm.unlink()
            except Exception:  # noqa: BLE001  平台差异下幂等
                pass


def test_ship_batch_emits_span_stamps():
    """ship_batch 必须把 chunk 的入队/认领时刻带到 synth_part 的 seg 级
    q_enq/q_claim（网关算请求级跨度）+ 批级 tts_done 戳；求和口径字段
    （q_tts/tts/q_tts_first/q_tts_last）已删除，请求级耗时全由网关按戳算。"""
    chunk = protocol.chunk("r-span", 0, "ps0", "zf_001", 1.0, 0, 1, "句")
    chunk["_q_enq"] = 100.0
    chunk["_q_claim"] = 101.0
    gq = FakeQueue()
    log = proc_common.setup_logging("test-span")
    compute.ship_batch(_PosStub(), {"batch_id": "b-span", "chunks": [chunk]},
                       [gq], FakeQueue(), log, np, shared_memory)
    part = gq.get_nowait()
    assert "q_tts" not in part["timings"] and "tts" not in part["timings"]
    assert "q_tts_first" not in part["timings"]
    assert "q_tts_last" not in part["timings"]
    assert part["timings"]["q_g2p"] == 0.0 and part["timings"]["g2p"] == 0.0
    # tts 墙钟跨度终点戳（前向结束时刻；网关对同请求取 max）
    assert isinstance(part["timings"].get("tts_done"), float)
    seg = part["block"]["segs"][0]
    assert seg["q_enq"] == 100.0 and seg["q_claim"] == 101.0
    shm = shared_memory.SharedMemory(name=part["block"]["shm_name"])
    try:
        shm.unlink()
    except Exception:  # noqa: BLE001  Windows 句柄/已清理
        pass
    shm.close()


def test_ship_batch_forward_lines_chunk_level(caplog):
    """TTS 行重设计：批头行（bid/n/process_ms/audio_s/dt）+ 逐 chunk 行
    （chunk/bid/queue_ms/audio_s/chars/sents）——chunk 键 {req}_{need}_{idx}
    自包含；可推导量（tmax=max(a)、pad=tmax−a、chunk 分摊 ms）不上行；
    chunk 行不带 req=（键前缀即 req）。"""
    import logging

    class _MixedStub:
        device = "cpu"
        lens = [24000, 9000, 3000]   # 采样 → audio_s 1.0/0.375/0.125

        def synth_batch(self, jobs, dtype="fp32"):
            return [np.full(self.lens[pos % len(self.lens)], 100, dtype="<i2")
                    for pos, _job in enumerate(jobs)]

    chunks = [protocol.chunk(req_id="r-pad", idx=i, ps=f"ps{i}",
                             voice="zf_001", speed=1.0, origin=0, need=3,
                             text=f"第{i}句测试文本", sents=1)
              for i in range(3)]
    gq = FakeQueue()
    log = proc_common.setup_logging("test-pad")
    with caplog.at_level(logging.INFO):
        compute.ship_batch(_MixedStub(),
                           {"batch_id": "t0-7", "chunks": chunks},
                           [gq], FakeQueue(), log, np, shared_memory)
    logs = [r.getMessage() for r in caplog.records]
    hdr = [x for x in logs if "bid=t0-7" in x and "n=3" in x]
    ck = [x for x in logs if "chunk=r-pad_3_0" in x]
    assert len(hdr) == 1 and len(ck) == 1, logs
    assert "process_ms=" in hdr[0] and "audio_s=1.5" in hdr[0] \
        and "dt=fp32" in hdr[0]
    assert "queue_ms=0" in ck[0] and "audio_s=1.0" in ck[0] \
        and "chars=7" in ck[0] and "sents=1" in ck[0]
    # chunk 行带批级 process_ms（单行自包含，不必回查批头行）
    assert "process_ms=" in ck[0]
    # chunk 行不带 req=（键前缀即 req）
    assert "req=r-pad" not in ck[0]
    part = _drain(gq)[0]
    shm = shared_memory.SharedMemory(name=part["block"]["shm_name"])
    try:
        shm.unlink()
    except Exception:  # noqa: BLE001  Windows 句柄/已清理
        pass
    shm.close()


# ── NPU 初始化挂死的看门狗覆盖 ──

def test_with_busy_covers_attempt():
    """_with_busy：段内 stuck_s 增长（心跳可见 → supervisor 按 STUCK_KILL_S
    击杀）、段外归零、异常路径也归零。NPU aicore 坏态存在「不抛错、永久
    阻塞」模式——不覆盖则心跳新鲜 + stuck=0 的僵尸 worker 永远占位。"""
    import time

    b = compute._busy
    seen = {}

    def slow():
        time.sleep(0.12)
        seen["stuck"] = b.stuck_s()

    compute._with_busy(slow)
    assert seen["stuck"] > 0            # 段内可见（0.1s 粒度）
    assert b.stuck_s() == 0.0           # 段外归零

    def boom():
        raise RuntimeError("init failed")

    with pytest.raises(RuntimeError):
        compute._with_busy(boom)
    assert b.stuck_s() == 0.0           # 异常路径同样归零


def test_npu_load_paths_wired_with_busy():
    """结构锁定：_load_compute/_load_asr 的 NPU 初始化段（设备探测/
    KModel 加载/冒烟前向/FunASR 加载与冒烟）必须经 _with_busy 接线。"""
    from pathlib import Path

    src = Path(compute.__file__).read_text(encoding="utf-8")
    for token in ("_with_busy(pipeline.get_compute)",
                  "_with_busy(compute.synth_batch_smoke)",
                  "_with_busy(funasr_ts.get_model)",
                  "_with_busy(_smoke)"):
        assert token in src, f"加载链失去看门狗覆盖: {token}"


# ── asr 批循环：_run_asr_batch 的回投/隔离/降级契约（funasr_ts 打桩，无模型） ──

_ALL_SHMS = []   # 本模块创建的 SHM（autouse 兜底清理：构造中途失败也不泄漏）


def _asr_job_block(rid, idx, need=1, origin=0, shm_ok=True):
    """建一个 SHM PCM16 块 + 对应 asr job。shm_ok=False 时给不存在的
    shm 名（模拟坏块——_read_job_audio 抛错路径）。返回 (job, shm|None)。"""
    shm = None
    if shm_ok:
        shm = shared_memory.SharedMemory(create=True, size=480 * 2)
        _ALL_SHMS.append(shm)
        shm.buf[:480 * 2] = np.full(480, 1234, dtype="<i2").tobytes()
        piece = {"shm_name": shm.name, "offset": 0, "samples": 480}
    else:
        piece = {"shm_name": "kokoro-nonexistent-shm-xyz", "offset": 0,
                 "samples": 480}
    job = protocol.asr_job(rid, idx, origin, need=need, piece=piece,
                           text="你好", sample_rate=24000)
    return job, shm


def _cleanup_shm(shms):
    for s in shms:
        if s is not None:
            try:
                _ALL_SHMS.remove(s)
            except ValueError:
                pass
            try:
                s.close()
            except Exception:  # noqa: BLE001  幂等
                pass
            try:
                s.unlink()
            except Exception:  # noqa: BLE001  平台差异下幂等
                pass


@pytest.fixture(autouse=True)
def _cleanup_shm_after_test():
    """兜底：即使用例在构造 job 列表中途异常，也不把命名 SHM 留给后续用例。"""
    yield
    while _ALL_SHMS:
        _cleanup_shm([_ALL_SHMS.pop()])


def test_run_asr_batch_replies_each_job(monkeypatch):
    """3 job 一批：单次批推、逐 job chunk_ts 顺序一一对应、timings 携带
    跨度戳（q_asr_first/last、asr_done）。funasr_ts.batch_group_timestamp
    打桩记录调用形态。"""
    import funasr_ts

    calls = []

    def _fake_batch(items, sample_rate, max_group_sec=None):
        calls.append({"n": len(items), "rate": sample_rate, "mgs": max_group_sec})
        return [{"text": "".join(t), "timestamp": [[0, 50]]} for _a, t in items]

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _fake_batch)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(3)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    msgs = _drain(gq)
    assert [m["req_id"] for m in msgs] == ["r0", "r1", "r2"]   # 顺序一一对应
    assert [m["idx"] for m in msgs] == [0, 1, 2]
    # 桩返回单元文本=chunk 原文"你好"（1 段）→ 绑定摊平为逐字 2 段
    assert all(m["error"] is None
               and m["timestamp"] == [[0, 25], [25, 50]] for m in msgs)
    # 求和口径字段已删；桩 job 无 _q_enq/_q_claim → None，asr_done 恒带
    assert all("q_asr" not in m["timings"] and "asr" not in m["timings"]
               and m["timings"]["q_asr_first"] is None
               and m["timings"]["q_asr_last"] is None
               and isinstance(m["timings"]["asr_done"], float)
               for m in msgs)
    assert calls == [{"n": 3, "rate": 24000, "mgs": 30.0}]     # 单次批推


def test_run_asr_batch_shm_failure_isolated(monkeypatch):
    """坏块只 error 自家：job0 SHM 不存在 → error 回投，job1 照常成功
    （不连坐——与 ship_batch 的批级连坐是两个不同故障域）。"""
    import funasr_ts

    monkeypatch.setattr(
        funasr_ts, "batch_group_timestamp",
        lambda items, rate, max_group_sec=None: [
            {"text": "".join(t), "timestamp": [[7, 9]]} for _a, t in items])
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    bad, _s = _asr_job_block("bad", 0, shm_ok=False)
    good, shm = _asr_job_block("good", 1)
    try:
        compute._run_asr_batch([bad, good], [gq], log)
    finally:
        _cleanup_shm([shm])
    msgs = {m["req_id"]: m for m in _drain(gq)}
    assert set(msgs) == {"bad", "good"}                        # 两个都有回投
    assert msgs["bad"]["error"] and msgs["bad"]["timestamp"] == []
    # "你好" 1 段 → 绑定摊平为逐字 2 段
    assert msgs["good"]["error"] is None \
        and msgs["good"]["timestamp"] == [[7, 8], [8, 9]]


def test_run_asr_batch_infer_failure_all_errored(monkeypatch):
    """批级推理失败 → 整批 error（ship_batch 同契约：compute 错误 →
    各受影响请求 502），逐 job 仍各自回投。"""
    import funasr_ts

    def _boom(items, rate, max_group_sec=None):
        raise RuntimeError("NPU OOM")

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _boom)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(3)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    msgs = _drain(gq)
    assert len(msgs) == 3                                      # 逐 job 回投不缺席
    assert all(m["error"] is not None and "NPU OOM" in m["error"] for m in msgs)


def test_run_asr_batch_degrade_effective_1(monkeypatch):
    """生效档 1（热身降级）：batch_group_timestamp 内部逐 job 走
    group_timestamp 单条路径（降级路径与降级后同一条代码）。"""
    import funasr_ts

    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 1})
    seen = []
    monkeypatch.setattr(
        funasr_ts, "group_timestamp",
        lambda parts, texts, sr, mgs=None: (seen.append((len(parts), texts)),
                                            {"text": "".join(texts),
                                             "timestamp": [[3, 4]]})[1])
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(2)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    msgs = _drain(gq)
    # "你好" 1 段 → 绑定摊平为逐字 2 段
    assert [m["timestamp"] for m in msgs] \
        == [[[3, 3], [3, 4]], [[3, 3], [3, 4]]]
    assert len(seen) == 2 and [t for _l, t in seen] == [["你好"], ["你好"]]
    assert all(l == 1 for l, _t in seen)   # parts=[整段音频] 列表（裸 ndarray 会 len 出采样数）
    # 逐 job 单条路径（每个 job 一次 group_timestamp 调用）


def test_run_asr_batch_to_batch_group_seam(monkeypatch):
    """接缝锁定（批路径）：_run_asr_batch → 真 batch_group_timestamp
    （_generate_batch 打桩）。job 音频必须以 parts=[整段音频] 列表形态
    进入分组数学——裸 ndarray 会被 len() 成采样点数并触发 parts/texts
    长度校验、整批 error（历史事故：单测全 mock 掉 batch_group_timestamp
    没拦住该参数形状错）。"""
    import funasr_ts

    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})
    monkeypatch.setattr(funasr_ts, "get_model", lambda: (object(), "cpu"))
    seen = []

    def _fake_gen(model, audios, texts, sr, mode=None):
        seen.append(([len(a) for a in audios], sr, list(texts)))
        return [{"text": t, "timestamp": [[0, 10]]} for t in texts]

    monkeypatch.setattr(funasr_ts, "_generate_batch", _fake_gen)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(3)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    # 单次批推：3 个单组 job 展平 3 条、24k、文本顺序一一对应
    assert seen == [([480, 480, 480], 24000, ["你好", "你好", "你好"])]
    msgs = _drain(gq)
    assert all(m["error"] is None
               and m["timestamp"] == [[0, 5], [5, 10]] for m in msgs)


def test_run_asr_batch_degraded_seam_real_group_timestamp(monkeypatch):
    """接缝锁定（降级路径）：_run_asr_batch → 真 batch_group_timestamp →
    真 group_timestamp（timestamp 打桩）——降级档同样要求
    parts=[整段音频] 列表形态。"""
    import funasr_ts

    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 1})
    monkeypatch.setattr(
        funasr_ts, "timestamp",
        lambda audio, text, sr: {"text": text, "timestamp": [[5, 6]]})
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(2)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    msgs = _drain(gq)
    assert all(m["error"] is None
               and m["timestamp"] == [[5, 5], [5, 6]] for m in msgs)


def test_ship_batch_result_drop_skips_asr_and_unlinks(monkeypatch):
    """synth_part 投递失败（结果队列满）：块无人注册 → 立即回收并跳过 ASR，
    不能继续派发引用该块的 asr_job。"""
    class _NoPut:
        def put(self, *a, **k):
            raise RuntimeError("result queue full")

    asr_q = FakeQueue()
    log = proc_common.setup_logging("test-tts-drop")
    chunk = protocol.chunk("r-drop", 0, "ps0", "zf_001", 1.0, 0, 1, "句",
                           needs_align=True)
    compute.ship_batch(_PosStub(), {"batch_id": "b-drop", "chunks": [chunk]},
                       [_NoPut()], asr_q, log, np, shared_memory)
    assert asr_q.qsize() == 0     # 不派发引用已回收块的 ASR job


def test_run_asr_batch_bad_metadata_isolated(monkeypatch):
    """一条畸形 sample_rate 只废自家，合法 job 照常批推（不整批丢弃）。"""
    import funasr_ts

    calls = []

    def _fake_batch(items, sample_rate, max_group_sec=None):
        calls.append((len(items), sample_rate))
        return [{"text": "".join(t), "timestamp": [[1, 2]]} for _a, t in items]

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _fake_batch)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr-meta")
    bad, shm_bad = _asr_job_block("bad", 0)
    bad["sample_rate"] = "not-a-number"
    good, shm_good = _asr_job_block("good", 1)
    try:
        compute._run_asr_batch([bad, good], [gq], log)
    finally:
        _cleanup_shm([shm_bad, shm_good])
    msgs = {m["req_id"]: m for m in _drain(gq)}
    assert set(msgs) == {"bad", "good"}
    assert msgs["bad"]["error"] and "参数非法" in msgs["bad"]["error"]
    assert msgs["good"]["error"] is None \
        and msgs["good"]["timestamp"] == [[1, 1], [1, 2]]
    assert calls == [(1, 24000)]   # 只有合法 job 进入批推


def test_run_asr_batch_mixed_rates_split_buckets(monkeypatch):
    """混采样率防御分桶：不同 rate 不同桶各推一次（今天恒 24000 一桶）。"""
    import funasr_ts

    calls = []

    def _fake_batch(items, sample_rate, max_group_sec=None):
        calls.append({"n": len(items), "rate": sample_rate})
        return [{"text": "".join(t), "timestamp": []} for _a, t in items]

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _fake_batch)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    a, s1 = _asr_job_block("ra", 0)
    b, s2 = _asr_job_block("rb", 1)
    b["sample_rate"] = 16000                # 防御性混率（生产恒 24k）
    try:
        compute._run_asr_batch([a, b], [gq], log)
    finally:
        _cleanup_shm([s1, s2])
    assert sorted(calls, key=lambda c: c["rate"]) == [
        {"n": 1, "rate": 16000}, {"n": 1, "rate": 24000}]


def test_run_asr_batch_oom_self_heal(monkeypatch):
    """运行期 OOM 自愈：整桶 error 预设 → 生效档减半 → 整桶重试一次 →
    成功则撤销错误照常回投（批推每桶恰两次调用：原档失败 + 新档成功）。"""
    import funasr_ts

    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})
    calls = []

    def _flaky(items, rate, max_group_sec=None):
        calls.append(funasr_ts.effective_batch())
        if len(calls) == 1:
            raise torch_npu_oom()
        return [{"text": "".join(t), "timestamp": [[1, 2]]} for _a, t in items]

    def torch_npu_oom():
        return RuntimeError("NPU out of memory. Tried to allocate 1.2GB")

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _flaky)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(3)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    assert calls == [4, 2]                       # 原档 4 失败 → 降档 2 重试
    assert funasr_ts.effective_batch() == 2      # 降档持久生效
    msgs = _drain(gq)
    assert all(m["error"] is None
               and m["timestamp"] == [[1, 1], [1, 2]] for m in msgs)


def test_run_asr_batch_oom_retry_failure_still_errors(monkeypatch):
    """自愈重试仍败：维持整桶 error（限一次重试，不无限循环）；降档持久
    生效但不再递减（重试失败不再降档——下一批仍按新档尝试）。"""
    import funasr_ts

    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})
    calls = []

    def _always_oom(items, rate, max_group_sec=None):
        calls.append(funasr_ts.effective_batch())
        raise RuntimeError("hbm alloc failed")

    monkeypatch.setattr(funasr_ts, "batch_group_timestamp", _always_oom)
    gq = FakeQueue()
    log = proc_common.setup_logging("test-asr")
    jobs, shms = zip(*[_asr_job_block(f"r{i}", i) for i in range(2)])
    try:
        compute._run_asr_batch(list(jobs), [gq], log)
    finally:
        _cleanup_shm(shms)
    assert calls == [4, 2]                       # 恰一次重试
    assert funasr_ts.effective_batch() == 2      # 降档一次后不再递减
    msgs = _drain(gq)
    assert all(m["error"] is not None for m in msgs)


def test_looks_like_oom_matching():
    """OOM 特征匹配：torch/CANN 文案命中；room/bloom/SharedMemory 等普通
    错误不误触发（裸 "oom" 子串曾会永久减半批宽）。"""
    assert compute._looks_like_oom("NPU out of memory. Tried to allocate ...")
    assert compute._looks_like_oom("Ascend hbm alloc failed")
    assert compute._looks_like_oom("RuntimeError: NPU OOM")
    assert compute._looks_like_oom("container OOMKilled (exit 137)")
    assert not compute._looks_like_oom("[Errno 2] SharedMemory not found")
    assert not compute._looks_like_oom("批推结果数不符 2 != 3")
    assert not compute._looks_like_oom("cannot open /tmp/room-1/audio.wav")
    assert not compute._looks_like_oom("zoom level mismatch")
    assert not compute._looks_like_oom("bloom filter init failed")


def test_asr_batch_wiring():
    """结构锁定：asr_main 必须批认领（宽度 = min(配置, 生效档)，OOM 自愈
    降档后认领随之收窄）+ 整批一次 ack（member 列表）——防止回退成逐条
    认领/逐条 ack。只解析 asr_main 段：ack(members,...) 在 tts_main 也有，
    全文件 grep 会漏掉 asr 回退。"""
    from pathlib import Path

    src = Path(compute.__file__).read_text(encoding="utf-8")
    asr_src = src[src.index("def asr_main"):]
    assert "min(_ASR_MAX_BATCH, funasr_ts.effective_batch()), worker_id)" in asr_src
    assert "claim_queue.ack(members, worker_id)" in asr_src
    assert "claim_queue.claim(1, worker_id)" not in asr_src


def test_sample_asr_mem_records_heartbeat_snapshot(monkeypatch):
    """asr worker 主线程采样 → 心跳 npu_mem（metrics 的 per-worker 明细源）。"""
    import pipeline

    monkeypatch.setattr(pipeline, "asr_budget_meta", lambda: (1818, 500, 30.0))
    monkeypatch.setattr(pipeline, "asr_budget_error", lambda: None)
    monkeypatch.setattr(
        pipeline, "npu_mem_estimate",
        lambda dev, budget, ctx: {"reserved_mb": 1780.0,
                                  "estimated_mb": 2230.0,
                                  "context_mb": ctx,
                                  "budget_mb": budget,
                                  "process_mb": 2196.0,
                                  "delta_mb": -35.6,
                                  "context_actual_mb": 416.0})
    monkeypatch.setattr(compute, "_asr_state", {"device": "npu:0"})
    monkeypatch.setattr(compute, "_stats", dict(compute._stats))

    compute._sample_asr_mem()
    assert compute._stats["npu_mem"]["estimated_mb"] == 2230.0
    assert compute._stats["npu_mem"]["context_mb"] == pipeline.ASR_CONTEXT_MB
    assert compute._stats["npu_mem"]["process_mb"] == 2196.0
    assert compute._stats["npu_mem"]["delta_mb"] == -35.6
    snap = compute._asr_snapshot()
    assert snap["npu_mem"]["reserved_mb"] == 1780.0
    assert snap["npu_mem"]["process_mb"] == 2196.0


def test_sample_npu_mem_uses_tts_budget_and_skips_cpu(monkeypatch):
    """tts 采样传 _last_budget/TTS_CONTEXT_MB；CPU 设备不调 NPU API。"""
    import pipeline

    calls = []
    monkeypatch.setattr(pipeline, "_last_budget", (9543, 126, 55.1))
    monkeypatch.setattr(pipeline, "budget_error", lambda: None)
    monkeypatch.setattr(
        pipeline, "npu_mem_estimate",
        lambda dev, budget, ctx: calls.append((dev, budget, ctx)) or {
            "reserved_mb": 9500.0, "process_mb": 9683.0,
            "delta_mb": -119.1})
    monkeypatch.setattr(compute, "_stats", dict(compute._stats))

    class _Npu:
        device = "npu:0"

    class _Cpu:
        device = "cpu"

    compute._sample_npu_mem(_Npu())
    assert calls == [("npu:0", 9543, pipeline.TTS_CONTEXT_MB)]
    assert compute._stats["npu_mem"]["process_mb"] == 9683.0
    compute._sample_npu_mem(_Cpu())
    assert len(calls) == 1


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_compute.py")

"""asr 对齐 worker：FunASR 加载 → 认领桶批 → 词级时间戳回投 origin 网关。"""
import logging
import os
import re
import time

import proc_common
import protocol

import worker_common
from worker_common import (SAMPLE_RATE, _ASR_MAX_BATCH, _busy, _chunk_key,
                           _queue_ms, _looks_like_oom, _reply_chunk_ts_safe,
                           _sample_worker_npu_mem, _snapshot_base, _stats,
                           _with_busy, _asr_warm)

def _sample_asr_mem() -> None:
    """asr 不跑批量前向之外的持续负载，采样点放在加载完成与每批推理之后；
    空闲期沿用最后一次快照（内存预占恒定，不会漂移）。"""
    import npu_ops

    meta = npu_ops.asr_budget_meta()
    _sample_worker_npu_mem(_asr_state.get("device"),
                           meta[0] if meta else None,
                           npu_ops.ASR_CONTEXT_MB, npu_ops.asr_budget_error)


def _asr_snapshot() -> dict:
    def _effective():
        try:
            import funasr_ts
            return funasr_ts.effective_batch()   # 单源活值（含 OOM 降档）
        except Exception:  # noqa: BLE001  观测失败不影响心跳
            return None   # 观测不可用诚实上报，不回退陈旧影子值

    def _patch_ok():
        try:
            import funasr_ts
            st = funasr_ts.batch_patch_status()
            return None if st is None else bool(st.get("ok"))
        except Exception:  # noqa: BLE001
            return None
    return _snapshot_base("asr", _asr_state["device"],
                          _asr_warm.get("funasr") == "ok", _asr_warm,
                          {"batch_effective": _effective(),
                           "batch_patch_ok": _patch_ok()})


_asr_state = {"device": None}


def _load_asr(log):
    """FunASR 加载+冒烟（仅 asr 池）。失败驻留重试（proc_common.retry_forever
    骨架，同 tts 池 fail-fast 语义）。返回实际设备字符串。"""
    import funasr_ts

    def _init():
        _model, used = _with_busy(funasr_ts.get_model)  # 失败直接抛（fail-fast）
        if str(used).startswith("npu"):
            _with_busy(lambda: npu_ops.prime_asr_memory(used))  # 模型确已上卡才预占预算
            # 无条件同步（成功为 None）：重试成功后必须清掉旧错误，
            # 否则心跳永远假报预算失败。
            _stats["budget_error"] = npu_ops.asr_budget_error()
        _asr_state["device"] = used or "cpu"

        def _smoke():
            funasr_ts.group_timestamp(
                [0.1 * funasr_ts.np.ones(SAMPLE_RATE * 2, dtype="float32")],
                ["你好"], SAMPLE_RATE)  # 冒烟（NPU 推理：挂死风险点）
        _with_busy(_smoke)
        # 批推最坏形状热身（OOM 阶梯自降级，任何档位不抛——见
        # funasr_ts.warmup_batch）：生效档入 warm/stats，认领量钳它。
        # 必须热身完再置 ready：网关 _scan_asr_ready 只看 warm.funasr，
        # 提前置 ok 会在热身期间放行 /tts，而 asr 还没进认领循环。
        _with_busy(lambda: funasr_ts.warmup_batch(_ASR_MAX_BATCH))
        _asr_warm["funasr"] = "ok"
        _patch = funasr_ts.batch_patch_status()
        log.info("FunASR ready (forced-alignment inference worker, device=%s,"
                 " batch_effective=%d, batch_patch_ok=%s)",
                 _asr_state["device"], funasr_ts.effective_batch(),
                 _patch.get("ok") if _patch else None)
        return _asr_state["device"]

    return proc_common.retry_forever(_init, "funasr", _asr_warm, "FunASR", log)


def _read_job_audio(job: dict):
    """读 asr job 的 SHM PCM16 段 → float32（÷32767，与写入端同缩放）。

    SHM 连接/解码失败抛错——调用方逐 job 捕获（坏块只伤自家，不连坐）。"""
    from multiprocessing import shared_memory

    import numpy as np

    piece = job["piece"]
    shm = shared_memory.SharedMemory(name=piece["shm_name"])
    try:
        # SHM 线格式 = PCM16：还原 float32（÷32767，与写入端同缩放）
        return np.frombuffer(
            shm.buf[piece["offset"]:piece["offset"] + piece["samples"] * 2],
            dtype="<i2").astype(np.float32) / 32767.0
    finally:
        shm.close()  # unlink 归 gateway（创建侧/终态接收方）


def _run_asr_batch(jobs: list, gateway_queues, log, bid: str = "") -> None:
    """一批 asr job：SHM 读取（逐条隔离）→ 批推理 → 逐 job 回投。

    错误契约与 tts ship_batch 同构：SHM 类单 job 失败只 error 该 job
    （坏块不连坐）；推理批级失败 error 整批（compute 错误 → 各受影响
    请求 502）。本函数正常路径不抛；畸形载荷的字段引用错误（如缺
    req_id）例外——落到 asr_main 外层 except 兜底。

    timings 只携带跨度戳（_q_enq/_q_claim/asr_done/asr_bucket）——请求级
    耗时跨度全在网关按戳计算；_stats.asr_ms 累计推理墙钟总和（按桶累加
    一次，不随 B 膨胀）。"""
    import funasr_ts

    n = len(jobs)
    audios = [None] * n
    replied = [False] * n
    ts_out = [None] * n
    err_out = [None] * n
    bucket_of = {}           # job idx → 桶起始时刻（网关去重计 asr_fw）
    # ① 逐 job 读 SHM：单 job 失败立即 error 回投并剔除，不连坐
    for i, job in enumerate(jobs):
        try:
            audios[i] = _read_job_audio(job)
        except Exception as exc:  # noqa: BLE001  坏块只伤自家
            err_out[i] = str(exc)[:200]
            _stats["jobs"] += 1
            _stats["job_errors"] += 1
            log.warning("chunk=%s bid=%s 读取失败: %s",
                        _chunk_key(job), bid, err_out[i])
            _reply_chunk_ts_safe(job, [], err_out[i], gateway_queues, log,
                                 warn="asr error reply failed")
            replied[i] = True
    todo = [i for i in range(n) if audios[i] is not None]
    if not todo:
        return
    # ② 按 (sample_rate, max_group_sec) 分桶批推（今天恒 24000/30 一桶；
    #    分桶防御混率/混组限作业同批——不同形状不同预算语义）
    buckets = {}
    for i in todo:
        job = jobs[i]
        # 单 job 元数据解析必须隔离：一条畸形 sample_rate/max_group_sec
        # 不能让整批（含其他请求的合法 job）落到 asr_main 外层 except 被
        # 整批丢弃——坏 job 单独回 error，其余照常推理。
        try:
            rate = int(job.get("sample_rate") or 24000)
            mgs = float(job.get("max_group_sec")
                        or proc_common.ASR_MAX_GROUP_SEC)
        except (TypeError, ValueError) as exc:
            err_out[i] = f"对齐作业参数非法: {str(exc)[:120]}"
            _stats["jobs"] += 1
            _stats["job_errors"] += 1
            log.warning("chunk=%s bid=%s 对齐参数非法: %s",
                        _chunk_key(job), bid, str(exc)[:160])
            _reply_chunk_ts_safe(job, [], err_out[i], gateway_queues, log,
                                 warn="asr bad-metadata reply failed")
            replied[i] = True
            continue
        buckets.setdefault((rate, mgs), []).append(i)
    infer_ms = 0.0
    for (rate, mgs), idxs in buckets.items():
        t1 = time.perf_counter()
        bucket_ts = time.time()   # 同桶共享一戳：网关去重计数 asr_fw
        bucket_of.update((i, bucket_ts) for i in idxs)
        try:
            res = funasr_ts.batch_group_timestamp(
                [([audios[i]], [jobs[i]["text"]]) for i in idxs], rate,
                max_group_sec=mgs)
            if len(res) != len(idxs):
                raise RuntimeError(f"批推结果数不符 {len(res)} != {len(idxs)}")
            for i, r in zip(idxs, res):
                # 单元→字符绑定：段数恒 == 可对齐字符数（"Token" 五字母
                # 共用一段这类粒度差在此摊平，bind/index 两侧计数一致）。
                # 单元清单按分词规则从输入文本重建——batch_group_timestamp
                # 回填的 text 是输入原文，不是分词回显，不可信。
                ts_out[i] = funasr_ts.bind_char_timestamps(
                    jobs[i]["text"], r.get("timestamp") or [])
        except Exception as exc:  # noqa: BLE001  批级失败整桶 error（ship_batch 同契约）
            msg = str(exc)
            log.warning("bid=%s 批推理失败 n=%d rate=%d mgs=%.0f: %s",
                        bid, len(idxs), rate, mgs, msg[:200])
            # OOM 自愈：热身已证明最坏形状可行——运行期 OOM 只能来自分配
            # 器碎片/异形形状序列。生效档减半并整桶重试一次（限一次，仍
            # 败则按错误回投）；降档持久生效，防 502 风暴循环到重启。
            if _looks_like_oom(msg) and len(idxs) > 1 \
                    and funasr_ts.effective_batch() > 1:
                for i in idxs:
                    err_out[i] = msg[:200]   # 先按失败预设：自愈成功再撤销
                funasr_ts.downscale_effective(msg)
                try:
                    res = funasr_ts.batch_group_timestamp(
                        [([audios[i]], [jobs[i]["text"]]) for i in idxs], rate,
                        max_group_sec=mgs)
                    if len(res) != len(idxs):
                        raise RuntimeError(
                            f"批推结果数不符 {len(res)} != {len(idxs)}")
                    for i, r in zip(idxs, res):
                        ts_out[i] = funasr_ts.bind_char_timestamps(
                            jobs[i]["text"], r.get("timestamp") or [])
                        err_out[i] = None   # 自愈成功：撤销预设错误
                    log.warning("asr batch OOM self-healed: effective=%d",
                                funasr_ts.effective_batch())
                except Exception as exc2:  # noqa: BLE001  仍败：维持整桶 error
                    for i in idxs:
                        err_out[i] = str(exc2)[:200]
                    log.warning("asr batch retry after downscale failed: %s",
                                str(exc2)[:200])
            else:
                for i in idxs:
                    err_out[i] = msg[:200]
        bucket_ms = (time.perf_counter() - t1) * 1000
        infer_ms += bucket_ms
    _stats["asr_ms"] = _stats.get("asr_ms", 0.0) + infer_ms
    # 批头行（bid 关联）：本批公共信息一行，job 行随后逐条
    log.info("bid=%s n=%d process_ms=%.1f", bid, n, infer_ms)
    # ③ 逐 job 回投（成功/失败统一路径）
    for i, job in enumerate(jobs):
        if replied[i]:
            continue
        replied[i] = True
        _stats["jobs"] += 1
        if err_out[i] is not None:
            _stats["job_errors"] += 1
            log.warning("chunk=%s bid=%s 对齐失败: %s",
                        _chunk_key(job), bid, err_out[i])
            _reply_chunk_ts_safe(job, [], err_out[i], gateway_queues, log,
                                 warn="asr error reply failed",
                                 bucket_ts=bucket_of.get(i))
        else:
            log.info("chunk=%s bid=%s process_ms=%.1f queue_ms=%.0f chars=%d",
                     _chunk_key(job), bid, infer_ms, _queue_ms(job),
                     len(ts_out[i] or []))
            _reply_chunk_ts_safe(job, ts_out[i] or [], None,
                                 gateway_queues, log,
                                 warn="asr reply failed",
                                 bucket_ts=bucket_of.get(i))
    n_err = sum(1 for i in range(n) if err_out[i] is not None)
    if n_err:
        log.warning("bid=%s 本批对齐失败 %d/%d", bid, n_err, n)


def asr_main(idx: int, claim_queue, gateway_queues) -> None:
    """ASR 池工人：FunASR chunk 级对齐（ASR_JOB → CHUNK_TS 直投 origin 网关）。

    chunk 级流水：tts worker 前向完一个 chunk 即派发对齐作业——ASR 与
    剩余 TTS 重叠执行。作业经认领式队列（in-flight 追踪）：worker 死亡
    由 supervisor 精确重派，作业零永久丢失（恢复时效秒级）。回投走
    gateway_queues[origin]（chunk 自带 origin）。

    批推（与 tts 池同款「队列里有什么拿多少」语义）：认领
    min(ASR_MAX_BATCH_SIZE, 热身生效档) 条——MonotonicAligner 批原生
    （一次 padded 批前向；显存预算 npu_ops._asr_auto_budget_mb 随
    ASR_MAX_BATCH_SIZE 缩放，热身 funasr_ts.warmup_batch 实测闭环）。
    整批一次 ack（认领队列 in-flight 账本按 member 列表原子清除，
    死亡重派语义与逐条 ack 完全一致）。"""
    import funasr_ts

    tag = f"asr-{idx}"
    log = proc_common.setup_logging(tag)
    proc_common.start_heartbeat(  # 先心跳后加载
        tag=tag, pid=os.getpid(), stats_fn=_asr_snapshot)

    _load_asr(log)
    _sample_asr_mem()   # 加载/热身完成即出首份内存快照

    worker_id = f"asr-{idx}-{os.getpid()}"   # 死亡重派按它定位在途账本
    batch_seq = 0   # 认领批序号 → bid（job 行/chunk 关联的批级键；含 pid 防重启重复）
    log.info("asr claim width %d (config B=%d, warmup effective %d)",
             min(_ASR_MAX_BATCH, funasr_ts.effective_batch()),
             _ASR_MAX_BATCH, funasr_ts.effective_batch())
    while True:
        try:
            # 认领宽每轮取 min(配置, 生效档)——OOM 自愈降档后认领随之收窄
            # （否则认领大批却只能分小桶推，在途多、批效率反而降）
            claimed = claim_queue.claim(
                min(_ASR_MAX_BATCH, funasr_ts.effective_batch()), worker_id)
        except Exception as exc:  # noqa: BLE001  redis 瞬断：等守护拉起
            log.warning("claim failed (redis restart window?): %s", exc)
            time.sleep(0.2)
            continue
        if not claimed:
            time.sleep(0.01)   # 队列空：轻轮询
            continue
        _busy.enter()  # 看门狗：FunASR 推理 hang 时心跳照常、stuck_s 增长
        members = [m for m, _j in claimed]
        jobs = [j for _m, j in claimed]
        _stats["batches"] += 1
        if len(jobs) > _stats["batch_max"]:
            _stats["batch_max"] = len(jobs)
        batch_seq += 1
        bid = f"a{idx}-{os.getpid()}-{batch_seq}"
        try:
            _run_asr_batch(jobs, gateway_queues, log, bid=bid)
        except Exception:  # noqa: BLE001  坏消息不杀工人（防御兜底，理论
            # 不可达——_run_asr_batch 内部全捕获；job 缺字段等畸形载荷
            # 落此整批计 drops，网关 fut_ts 超时 504 兜底）
            log.exception("asr batch handling error (dropped)")
            _stats["job_drops"] += len(jobs)
        finally:
            claim_queue.ack(members, worker_id)  # 交付确认（in-flight 移除）
            _sample_asr_mem()                    # 每批后刷新内存快照
            _busy.exit()



# 实际由 supervisor 调用 tts_main/asr_main

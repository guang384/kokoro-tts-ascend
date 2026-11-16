"""tts 推理 worker：KModel 加载 → 认领组批 → 批前向 → SHM 写回 + ASR 派发。"""
import logging
import sys
import time
import uuid

import proc_common
import protocol

import worker_common
from worker_common import (SAMPLE_RATE, _MAX_BATCH, _busy, _chunk_key,
                           _log_mem_sample, _queue_ms, _reqs_list,
                           _sample_worker_npu_mem, _snapshot_base, _stats,
                           _tts_warm, _with_busy, group_by_request)

_WIN_SHM_KEEPALIVE = []  # Windows 冒烟：写侧留置的 SHM 句柄（防命名映射提前销毁）


def _load_compute(tag: str):
    """KModel 加载+冒烟（仅 tts 池）。失败驻留重试（proc_common.retry_forever
    骨架，tts/asr/g2p 共用）。get_compute 只在成功后缓存单例，重试安全。"""
    import pipeline

    log = proc_common.setup_logging(tag)
    log.info("compute starting")

    def _init():
        compute = _with_busy(pipeline.get_compute)  # 探测+预算+预占+加载
        _with_busy(compute.synth_batch_smoke)
        _tts_warm["kokoro"] = "ok"
        log.info("compute engine ready: device=%s", compute.device)
        return compute

    return proc_common.retry_forever(_init, "kokoro", _tts_warm,
                                     "compute engine", log)


def _tts_snapshot(state: dict) -> dict:
    return _snapshot_base("tts", state["device"],
                          _tts_warm.get("kokoro") == "ok", _tts_warm, {})


def _sample_npu_mem(compute) -> None:
    import npu_ops

    _sample_worker_npu_mem(
        getattr(compute, "device", ""),
        npu_ops._last_budget[0] if npu_ops._last_budget else None,
        npu_ops.TTS_CONTEXT_MB, npu_ops.budget_error)


def ship_batch(compute, batch: dict, gateway_queues, asr_queue, log,
               np, shm_mod) -> None:
    """前向一整批 → 每请求一块 SHM → synth_part 逐 rid 直投其 origin 网关。

    批内可混请求（认领式组批按长度类，origin/voice 均可混）：按请求分块
    写入，块归属唯一请求 → 网关作为唯一清理者，/v1 与 /tts 两条终态路径
    都不会误清别人的块。前向失败 → 批内全部受影响请求各收一条
    synth_error（kind=compute）。回投路由按 chunk 自带 origin（批不再有
    统一 origin——认领式组批跨 origin 混批）。"""
    chunks = batch["chunks"]
    if not chunks:
        return
    by_req = group_by_request(chunks)
    batch_dtype = chunks[0].get("dtype") or protocol.DTYPE_DEFAULT
    # 认领类键已含 dtype（fp16/fp32 不混批）；防御：万一混入按头 chunk 精度
    # 整批执行（错精度好过崩 worker——fp16 下跑 fp32 语义只是慢，反之可能
    # 触发 NPU 算子不支持）
    t0 = time.perf_counter()
    try:
        audio_list = compute.synth_batch(
            [(c["ps"], c["voice"], c["speed"]) for c in chunks],
            dtype=batch_dtype)
        # 契约守卫放在 try 内：违约按"前向失败"给每个受影响请求回
        # synth_error——放在 try 外会被 tts_main 记日志后 ack 掉，请求
        # 无结果无错误地干等 504。长度也必须查：后续按 batch_pos 索引，
        # 短列表会在 per-request try 之外抛 IndexError（同样静默 504）
        if len(audio_list) != len(chunks):
            raise TypeError(
                f"synth_batch 返回 {len(audio_list)} 条 / 输入 {len(chunks)} 条")
        for a in audio_list:
            if a.dtype != np.int16:   # 坏 dtype 直写 SHM 会是静默爆音
                raise TypeError(f"synth_batch 必须返回 int16，得到 {a.dtype}")
    except Exception as exc:  # noqa: BLE001
        log.exception("bid=%s reqs=%s dt=%s 前向失败",
                      batch.get("batch_id"), _reqs_list(by_req), batch_dtype)
        _stats["forward_errors"] += 1
        for rid in by_req:
            rid_origin = int(by_req[rid][0][2].get("origin", 0))
            protocol.put_result(gateway_queues[rid_origin], protocol.synth_error(
                rid, f"前向失败: {str(exc)[:160]}", protocol.KIND_COMPUTE), log)
        return
    infer_s = time.perf_counter() - t0
    infer_ms = infer_s * 1000
    # 请求级 tts 墙钟跨度（首个 chunk 认领 → 末块前向完成）的时间戳：
    # 批内共享同一前向结束时刻；与 chunk 的 _q_enq/_q_claim 同为
    # time.time()，同机跨进程可比。
    tts_done_ts = time.time()
    _stats["jobs"] += 1
    _stats["forward_ms"] += infer_ms
    if batch_dtype == "fp16":
        _stats["fp16_jobs"] += 1
    _sample_npu_mem(compute)  # 主线程采样（仅 NPU 设备；心跳线程只读缓存）
    # SHM 线格式 = 交付终态（PCM16 int16）：fade/量化已整体上移 NPU，
    # synth_batch 直出 int16（D2H 字节数减半；原 CPU 侧 nan/clip/×32767/
    # rint/astype 5 遍全量扫描归零）——本函数只剩按帧数的纯 memcpy 裁剪
    # 直写 SHM。网关/asr 读取端按 <i2 还原 float32（÷32767）。
    audios = audio_list
    audio_s = sum(a.shape[0] for a in audios) / SAMPLE_RATE
    _stats["audio_s"] += audio_s
    # 批头行 + 逐 chunk 行（bid 关联两级）：批公共信息一行，每 chunk 的
    # 排队/产出一行。可推导量（tmax=max(a)、pad=tmax−a、chunk 分摊 ms、
    # Σ chunk ms=批墙钟）不上行。
    log.info("bid=%s n=%d process_ms=%.1f audio_s=%.1f dt=%s",
             batch.get("batch_id"), len(chunks), infer_ms, audio_s,
             batch_dtype)
    for i, c in enumerate(chunks):
        log.info("chunk=%s bid=%s process_ms=%.1f queue_ms=%.0f audio_s=%.1f "
                 "chars=%d sents=%d",
                 _chunk_key(c), batch.get("batch_id"), infer_ms, _queue_ms(c),
                 audios[i].shape[0] / SAMPLE_RATE,
                 len(c.get("text") or ""), int(c.get("sents") or 0))

    for rid, items in by_req.items():
        # need 是"该请求的 chunk 总数"（网关聚合收齐线），批内可混请求
        # ——必须按各请求自己的 chunk 取，绝不能拿批内第一条的 need 通提
        # （否则长单被短单的 need 截断、短单被长单的 need 拖到超时）。
        need = int(items[0][2].get("need", len(items)))
        total = sum(audios[pos].nbytes for _i, pos, _c in items)
        shm = None
        try:
            # 统一 kokora_ 前缀（历史拼写，网关清道夫按它匹配，勿改）：worker
            # 被 SIGBUS 杀死时来不及 unlink，半成品块成为无主孤儿——网关清道夫
            # 按前缀+超龄回收（SHM_ORPHAN_MAX_AGE_S），否则 /dev/shm 写满后
            # 重生 worker 即死即泄漏，池永远无法恢复
            shm = shm_mod.SharedMemory(
                create=True, size=max(total, 1),
                name=f"kokora_{uuid.uuid4().hex[:12]}")
            segs, off = [], 0
            for idx_, pos, c in items:
                a = audios[pos]
                # cast("B")：把 int16 视图按字节铺平后直写 SHM，省一次整句拷贝
                shm.buf[off:off + a.nbytes] = memoryview(a).cast("B")
                seg = {"idx": idx_, "offset": off,
                       "samples": a.shape[0], "text": c.get("text", "")}
                # 网关据此算请求级 q_tts 跨度；旧消息/chunk 无字段则缺省
                if isinstance(c.get("_q_enq"), (int, float)):
                    seg["q_enq"] = c["_q_enq"]
                if isinstance(c.get("_q_claim"), (int, float)):
                    seg["q_claim"] = c["_q_claim"]
                segs.append(seg)
                off += a.nbytes
            if sys.platform == "win32":
                # Windows：命名映射随最后一个句柄关闭而销毁——必须留住写侧
                # 句柄等 gateway attach（dev/长时间冒烟场景）；上限 64 个，
                # 淘汰最旧（早已被消费）防止长跑无限增长。
                # Linux 生产：POSIX shm 名字在 unlink 前恒存，写完即关不漏 fd
                _WIN_SHM_KEEPALIVE.append(shm)
                if len(_WIN_SHM_KEEPALIVE) > 64:
                    dead = _WIN_SHM_KEEPALIVE.pop(0)
                    try:
                        dead.close()
                    except Exception:  # noqa: BLE001
                        pass
            else:
                shm.close()  # 写完即关本进程句柄；块由终态接收方 gateway unlink
        except Exception as exc:  # noqa: BLE001
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()  # 写入失败的半成品当场回收
                except Exception:  # noqa: BLE001
                    pass
            log.exception("bid=%s req=%s SHM write failed",
                          batch.get("batch_id"), rid)
            rid_origin = int(items[0][2].get("origin", 0))
            protocol.put_result(gateway_queues[rid_origin], protocol.synth_error(
                rid, f"音频回传失败: {str(exc)[:120]}", protocol.KIND_COMPUTE), log)
            continue
        # 阶段耗时随行：q_g2p/g2p 整单一次算 → 各 chunk 同值取 max；
        # tts_done 为本批前向结束的绝对时刻（网关对同请求取 max，与 seg
        # 的 q_claim min 相减得请求级 tts 墙钟跨度）。请求级排队跨度由
        # seg 级 q_enq/q_claim（入队/认领时刻）在网关计算，不随消息重复携带。
        timings = {
            "q_g2p": max((c.get("q_g2p_ms") or 0.0 for _i, _p, c in items),
                         default=0.0),
            "g2p": max((c.get("g2p_ms") or 0.0 for _i, _p, c in items),
                       default=0.0),
            "tts_done": tts_done_ts,
        }
        delivered = protocol.put_result(
            gateway_queues[items[0][2].get("origin", 0)],
            protocol.synth_part(rid, need,
                                {"shm_name": shm.name, "segs": segs},
                                timings=timings),
            log)
        if not delivered:
            # 结果队列满/网关卡死：这条 synth_part 是 shm_name 的唯一载体，
            # 丢了就没人会注册/回收这个块；继续派 ASR 只会让 ASR 白读一个
            # 无人认领的块。立即回收并跳过本请求的 ASR（请求由网关 504 兜底）。
            log.warning("synth_part delivery failed req=%s; unlink block and "
                        "skip ASR dispatch", rid)
            if sys.platform == "win32":
                try:
                    _WIN_SHM_KEEPALIVE.remove(shm)
                except ValueError:
                    pass
            try:
                shm.unlink()
            except Exception:  # noqa: BLE001  已 unlink/Windows 句柄未释放
                pass
            try:
                shm.close()
            except Exception:  # noqa: BLE001
                pass
            continue
        # chunk 级对齐流水：needs_align 的 chunk 前向后即派发 ASR 作业
        # （ASR 与剩余 TTS 重叠执行；取消标记不在此查——网关 TS 账本对
        # 已断连请求走孤儿丢弃）
        for idx_, pos, c in items:
            if not c.get("needs_align"):
                continue
            seg = next(s_ for s_ in segs if s_["idx"] == idx_)
            try:
                asr_queue.put_nowait(protocol.asr_job(
                    req_id=rid, idx=idx_, origin=int(c.get("origin", 0)),
                    need=need,
                    piece={"shm_name": shm.name, "offset": seg["offset"],
                           "samples": seg["samples"]},
                    text=c.get("text", ""), sample_rate=SAMPLE_RATE,
                    max_group_sec=proc_common.ASR_MAX_GROUP_SEC))
            except Exception:  # noqa: BLE001  queue.Full：对齐降级，不拖住 tts worker
                protocol.put_result(
                    gateway_queues[int(c.get("origin", 0))],
                    protocol.chunk_ts(rid, idx_, need, seg["samples"], [],
                                      error="对齐服务不可用，请稍后重试"),
                    log)



def tts_main(idx: int, claim_queue, asr_queue, gateway_queues) -> None:
    """TTS 池工人：无状态批工人（聚合在网关，见 protocol.py 模块注释）。

    批由 worker 认领时组装（按队列快照取同类 chunk ≤MAX_BATCH）：
    饱和自动满批、空闲零攒批延迟——零定时窗。断连取消经
    kokoro:cancel ZSET 检查（认领后前向前过滤）。"""
    import numpy as np
    from multiprocessing import shared_memory

    tag = f"tts-{idx}"
    log = proc_common.setup_logging(tag)
    state = {"device": None}
    proc_common.start_heartbeat(  # 先心跳后加载：预热期可见、失败不隐身
        tag=tag, pid=os.getpid(), stats_fn=lambda: _tts_snapshot(state))

    compute = _load_compute(tag)
    state["device"] = compute.device
    _sample_npu_mem(compute)   # 加载完成即出首份内存快照（心跳可见）
    log.info("tts-%d ready, claiming chunks", idx)

    seq = 0
    worker_id = f"tts-{idx}-{os.getpid()}"   # 死亡重派按它定位在途账本
    while True:
        # claim 连接异常捕获：redis 重启窗口内 worker 等待守护拉起后重试
        # （与 InstrumentedQueue._recv 同语义，不崩）
        try:
            claimed = claim_queue.claim(_MAX_BATCH, worker_id)
        except Exception as exc:  # noqa: BLE001  redis 瞬断：等守护重启
            log.warning("claim failed (redis restart window?): %s", exc)
            time.sleep(0.2)
            continue
        if not claimed:
            time.sleep(0.01)   # 队列空：轻轮询（交接延迟 ≤10ms，量级无感）
            continue
        members = [m for m, _c in claimed]
        _busy.enter()  # 看门狗：认领即算"忙"，处理完清零（hang 时 stuck_s 增长）
        batch = {"batch_id": f"t{idx}-{os.getpid()}-{seq}", "chunks": []}  # 深度审计 F3：预绑定，
        try:                                                    # except 曾引用未定义变量杀工人
            # 断连取消：认领后前向前过滤（一次 pipeline 查批内全部唯一
            # req_id；批内 chunk 常来自同几请求），被取消的 chunk 直接
            # 清标记丢弃——请求已断开，结果无处可去。
            # fail-open：查询本身失败（redis 瞬断）时照常计算——最坏是
            # 白算一批已断连请求（结果由网关孤儿路径丢弃）；fail-closed
            # 会 ack 掉整批，请求无结果无错误地干等 WAIT_TIMEOUT
            try:
                dead = claim_queue.cancelled_set(
                    {c["req_id"] for _m, c in claimed})
            except Exception as exc:  # noqa: BLE001
                log.warning("cancel check failed (fail-open): %s", exc)
                dead = set()
            live = [(m, c) for m, c in claimed if c["req_id"] not in dead]
            dropped = len(claimed) - len(live)
            if dropped:
                _stats["cancelled_chunks"] += dropped
                log.info("req chunks dropped (cancelled before forward): %d",
                         dropped)
            if not live:
                continue
            batch["chunks"] = [c for _m, c in live]
            seq += 1
            ship_batch(compute, batch, gateway_queues, asr_queue,
                       log, np, shared_memory)
        except Exception:  # noqa: BLE001  单批账目异常不杀工人
            log.exception("batch handling error batch=%s", batch.get("batch_id"))
        finally:
            claim_queue.ack(members, worker_id)   # 交付确认（in-flight 移除）
            _busy.exit()

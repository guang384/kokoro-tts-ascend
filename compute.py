#!/usr/bin/env python3
"""compute worker：两个角色的模型工人（supervisor 按角色拉起）。

tts 池（tts_main）：拉批任务 → 前向 → 按请求分块写 PCM16 SHM →
synth_part 直投 origin 网关。完全无状态：批内可混请求（认领式组批
语义），聚合"按 req_id 收齐 need 句"由网关完成——本进程崩溃只丢当前
批，死者在途由 supervisor 精确重派（请求不丢），supervisor 自动
拉起新进程继续消费。

asr 池（asr_main）：FunASR 词级时间戳推理（chunk 级 job → chunk_ts
直投 origin 网关）。不加载 KModel。

前向 = batched_model.forward_batch（B≤MAX_BATCH、同 (音素长度类, dtype)
桶内的句），质量与历史版本完全一致（未采用 concat 多句单序列——时长预测
OOD，A/B 结论见 ab_quality.py）。
"""
import logging
import os
import re
import sys
import time
import uuid

import proc_common
import protocol

_mem_log = logging.getLogger("kokoro.compute")

_started_at = time.time()
SAMPLE_RATE = proc_common.env_int("KOKORO_SAMPLE_RATE", 24000, minimum=8000)
_MAX_BATCH = proc_common.env_int("TTS_MAX_BATCH_SIZE", 8, minimum=1)
_ASR_MAX_BATCH = proc_common.env_int("ASR_MAX_BATCH_SIZE", 8, minimum=1)  # asr 批认领上限（显存预算 pipeline._asr_auto_budget_mb 随它缩放）
_busy = proc_common.BusyWatch()  # 推理卡死看门狗：任务取出→enter，处理完→exit
# stats 每进程仅一个角色在跑，共用一份无妨。**全部键必须预置**：心跳线程
# 用 dict(_stats) 快照，主循环中途首插新键会触发 "dictionary changed size
# during iteration" → 心跳线程死亡 → supervisor 30s 后误杀健康 worker。
# jobs：tts=前向批数 / asr=对齐作业数（同 dict 但每进程只跑一个角色，无冲突）
_stats = {"jobs": 0, "forward_ms": 0.0, "audio_s": 0.0,
          "forward_errors": 0, "job_drops": 0, "fp16_jobs": 0,
          "asr_ms": 0.0, "job_errors": 0, "npu_mem": None,
          "budget_error": None, "cancelled_chunks": 0,
          "batches": 0, "batch_max": 0}
_tts_warm = {"kokoro": "pending"}   # tts 池预热项
_asr_warm = {"funasr": "pending"}   # asr 池预热项
_WIN_SHM_KEEPALIVE = []  # Windows 冒烟：写侧留置的 SHM 句柄（防命名映射提前销毁）


def group_by_request(chunks: list) -> dict:
    """批内 chunk 按请求分组、组内按句序排序（纯函数，供测试锁定混批语义）。

    返回 {req_id: [(chunk_idx, batch_pos, chunk), ...]}——batch_pos 是该
    chunk 在批中的位置（对应 synth_batch 返回音频的下标）。"""
    by_req: dict = {}
    for pos, c in enumerate(chunks):
        by_req.setdefault(c["req_id"], []).append((int(c["idx"]), pos, c))
    for rid in by_req:
        by_req[rid].sort(key=lambda t: t[0])
    return by_req


def _with_busy(fn):
    """看门狗覆盖段：NPU 初始化（设备探测/KModel 加载/冒烟前向/FunASR
    推理）存在「不抛错、永久阻塞」的坏态（aicore 挂起）。不进 BusyWatch
    的话心跳照常新鲜、stuck_s 恒 0——僵尸 worker 永远占位，产能静默损失
    （串行启动闸超时放行后无人再管它）。进入本段后 supervisor 按
    STUCK_KILL_S（默认 300s >> 正常加载 60-90s）击杀重生。"""
    _busy.enter()
    try:
        return fn()
    finally:
        _busy.exit()


def _load_compute(tag: str):
    """KModel 加载+冒烟（仅 tts 池）。失败驻留重试（指数退避 5s→120s）。

    看门狗 SIGKILL 前代工人后，NPU 显存回收有延迟、aicore 可能短暂坏态
    ——立即重生再初始化会失败。原"失败永久驻留不重试"会让一次坏态变成
    整池永久死亡（生产事故 20261108：6/6 tts 全部 warm=fail 僵尸、零容量，
    认领队列顶满全服务堵塞）。get_compute 只在成功后缓存单例，
    重试安全。

    **不降级**（fail-fast）：持续失败保持 warm=fail → /health 503 →
    编排层（K8s liveness）重启容器换取全新 NPU 上下文；悄悄降级 CPU 会
    把"卡坏了"伪装成"还能跑"，掩盖故障。"""
    import pipeline

    log = proc_common.setup_logging(tag)
    log.info("compute starting")
    delay = 5.0
    while True:
        try:
            compute = _with_busy(pipeline.get_compute)  # 探测+预算+预占+加载
            _with_busy(compute.synth_batch_smoke)
            _tts_warm["kokoro"] = "ok"
            log.info("compute engine ready: device=%s", compute.device)
            return compute
        except Exception as exc:  # noqa: BLE001
            _tts_warm["kokoro"] = "fail"
            log.error("compute engine init failed, retry in %.0fs (warm=fail"
                      " → /health 503 → orchestration restarts container): %s",
                      delay, str(exc)[:200])
            time.sleep(delay)
            delay = min(delay * 2, 120.0)




def _tts_snapshot(state: dict) -> dict:
    """tts 心跳快照：device/warm/stats + NPU 占用 + 看门狗（/health 诊断用）。

    注意：本函数在心跳线程运行，**绝不能调 torch.npu API**——设备坏态时
    副线程碰 NPU 会被 SIGBUS 直接杀进程（生产事故 20261109：5 个空闲
    tts worker 的心跳线程相继被带走）。占用数据由主循环每次前向后采样
    写入 _stats["npu_mem"]，这里只读缓存值。"""
    out = {"role": "tts", "device": state["device"],
           "ready": _tts_warm.get("kokoro") == "ok",
           "uptime_s": round(time.time() - _started_at, 1),
           "stuck_s": _busy.stuck_s(),
           "warm": dict(_tts_warm),
           "stats": {**dict(_stats), "ipc_dropped": protocol.drop_count()},
           "npu_mem": None}
    if _stats.get("npu_mem") is not None:
        out["npu_mem"] = _stats["npu_mem"]  # 主循环采样（非实时，5s 心跳粒度）
    return out


def _log_mem_sample(mem: dict, prev: dict | None) -> None:
    """实际 npu-smi 内存变化时打一行 INFO（30s 缓存天然限频）。

    metrics 是连续观测；日志给真机排障一个可直接 grep 的锚点：
    actual 与 planned 的差就是公式偏差。"""
    if not mem or mem.get("process_mb") is None:
        return
    if prev and prev.get("process_mb") == mem.get("process_mb"):
        return
    planned = (f"{mem['planned_mb']:.0f}"
               if mem.get("planned_mb") is not None else "n/a")
    delta = (f"{mem['delta_mb']:+.0f}"
             if mem.get("delta_mb") is not None else "n/a")
    ctx_actual = (f"{mem['context_actual_mb']:.0f}"
                  if mem.get("context_actual_mb") is not None else "n/a")
    _mem_log.info(
        "npu mem actual=%.0fMB reserved=%.0fMB estimated=%.0fMB "
        "planned=%s delta=%s context_actual=%s",
        mem["process_mb"], mem.get("reserved_mb", 0.0),
        mem.get("estimated_mb", 0.0), planned, delta, ctx_actual)


def _sample_npu_mem(compute) -> None:
    """主循环采样 tts worker 的 NPU 内存进 _stats——只有主线程碰 NPU API。

    估算口径由 pipeline.estimate_worker_mem 统一：reserved（torch 缓存池）
    + CANN 上下文常量（tts 450），并给出预算/整卡占比。心跳线程只读
    _stats["npu_mem"]。

    **仅 NPU 设备启用**：非 NPU（显式 DEVICE=cpu/测试替身/无驱动容器）直接跳过
    ——torch_npu 在无驱动环境调用会在 C 层段错误，except 拦不住
    （20261109 镜像内验证 Segmentation fault 的根因）。"""
    dev = str(getattr(compute, "device", "") or "")
    if not dev.startswith("npu"):
        return
    try:
        import pipeline

        budget = pipeline._last_budget[0] if pipeline._last_budget else None
        prev = _stats.get("npu_mem")
        mem = pipeline.npu_mem_estimate(dev, budget, pipeline.TTS_CONTEXT_MB)
        if mem is not None:
            _stats["npu_mem"] = mem
            _log_mem_sample(mem, prev)
        if pipeline.budget_error():
            _stats["budget_error"] = pipeline.budget_error()
    except Exception:  # noqa: BLE001  采样失败不影响服务
        pass


def _sample_asr_mem() -> None:
    """主循环采样 asr worker 的 NPU 内存（与 tts 同一估算口径）。

    asr 不跑批量前向之外的持续负载，采样点放在加载完成与每批推理之后；
    空闲期沿用最后一次快照（内存预占恒定，不会漂移）。"""
    dev = str(_asr_state.get("device") or "")
    if not dev.startswith("npu"):
        return
    try:
        import pipeline

        meta = pipeline.asr_budget_meta()
        budget = meta[0] if meta else None
        prev = _stats.get("npu_mem")
        mem = pipeline.npu_mem_estimate(dev, budget, pipeline.ASR_CONTEXT_MB)
        if mem is not None:
            _stats["npu_mem"] = mem
            _log_mem_sample(mem, prev)
        if pipeline.asr_budget_error():
            _stats["budget_error"] = pipeline.asr_budget_error()
    except Exception:  # noqa: BLE001  采样失败不影响服务
        pass


def _chunk_key(x: dict) -> str:
    """chunk 级关联键：{req_id}_{need}_{idx}——TTS chunk 行与 ASR job 行
    同值（同一块音频的两个阶段），grep 它直接对齐两段。"""
    return f"{x.get('req_id')}_{x.get('need')}_{x.get('idx')}"


def _queue_ms(x: dict) -> float:
    """认领等待毫秒：claim − enqueue（chunk/job 通用；缺戳按 0）。"""
    enq, claim = x.get("_q_enq"), x.get("_q_claim")
    if isinstance(enq, (int, float)) and isinstance(claim, (int, float)):
        return max(0.0, (claim - enq) * 1000)
    return 0.0


def _reqs_list(by_req: dict) -> str:
    """批次归属清单：rid(chunk 数) 用 + 连接（forward 失败行的关联字段）。"""
    return "+".join(f"{rid}({len(items)})" for rid, items in by_req.items())


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
        # 阶段耗时随行（升级期兼容字段）：v12 的 done 行不再展示这些跨度
        # （批/句级明细由 worker 的 bid=/chunk= 行承担），字段保留供旧网关
        # 滚动升级兜底。g2p 整单一次算 → 各 chunk 同值取 max；q_tts 为本批
        # 内该请求各 chunk 认领等待之和；tts 前向为批时按 chunk 数分摊。
        # 旧消息缺字段由消费端按 0 兜底。
        timings = {
            "q_g2p": max((c.get("q_g2p_ms") or 0.0 for _i, _p, c in items),
                         default=0.0),
            "g2p": max((c.get("g2p_ms") or 0.0 for _i, _p, c in items),
                       default=0.0),
            # 旧口径保留给未升级网关/旧消息兜底；新 q_tts 由 seg 的
            # q_enq/q_claim 在网关按"首个入队→最后出队"墙钟跨度计算。
            "q_tts": sum(c.get("_qw_ms") or 0.0 for _i, _p, c in items),
            "tts": infer_ms * len(items) / len(chunks),
            "tts_done": tts_done_ts,
        }
        enq = [c["_q_enq"] for _i, _p, c in items
               if isinstance(c.get("_q_enq"), (int, float))]
        claim = [c["_q_claim"] for _i, _p, c in items
                 if isinstance(c.get("_q_claim"), (int, float))]
        if enq and claim:
            timings["q_tts_first"] = min(enq)
            timings["q_tts_last"] = max(claim)
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
    return {"role": "asr", "device": _asr_state["device"],
            "ready": _asr_warm.get("funasr") == "ok",
            "uptime_s": round(time.time() - _started_at, 1),
            "stuck_s": _busy.stuck_s(),
            "warm": dict(_asr_warm),
            "npu_mem": _stats.get("npu_mem"),
            "stats": {**dict(_stats), "ipc_dropped": protocol.drop_count(),
                      "batch_effective": _effective(),
                      "batch_patch_ok": _patch_ok()}}


_asr_state = {"device": None}


def _load_asr(log):
    """FunASR 加载+冒烟（仅 asr 池）。失败驻留重试（指数退避），不降级
    （同 tts 池 fail-fast 语义）。返回实际设备字符串。"""
    import funasr_ts
    import pipeline

    delay = 5.0
    while True:
        try:
            _model, used = _with_busy(funasr_ts.get_model)  # 失败直接抛（fail-fast）
            if str(used).startswith("npu"):
                _with_busy(lambda: pipeline.prime_asr_memory(used))  # 模型确已上卡才预占预算
                # 无条件同步（成功为 None）：重试成功后必须清掉旧错误，
                # 否则心跳永远假报预算失败。
                _stats["budget_error"] = pipeline.asr_budget_error()
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
        except Exception as exc:  # noqa: BLE001
            _asr_warm["funasr"] = "fail"
            log.error("FunASR init failed, retry in %.0fs (warm=fail → /health"
                      " 503 → orchestration restarts container): %s",
                      delay, str(exc)[:200])
            time.sleep(delay)
            delay = min(delay * 2, 120.0)


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


_OOM_RE = re.compile(
    r"(?i)\bout of memory\b|\boom(?:[-_ ]?kill(?:ed)?)?\b"
    r"|hbm[^\n]{0,40}(?:alloc|memory)")


def _looks_like_oom(msg: str) -> bool:
    """识别显存耗尽类异常（torch "out of memory"、独立缩写 OOM、昇腾 HBM
    分配/内存文案）——"SharedMemory" 等普通错误不误触发自愈。

    不能用裸子串：`"oom" in msg` 会命中 room/zoom/bloom，误判会把
    effective_batch 永久减半（只降不升）。"""
    return bool(_OOM_RE.search(str(msg)))


def _reply_chunk_ts(job: dict, timestamp: list, error, asr_ms: float,
                    gateway_queues, log, bucket_ts=None) -> None:
    """chunk_ts 回投 origin 网关（成功/失败统一路径）。回投满计
    job_drops——网关 fut_ts 超时 504 兜底（at-most-once 设计内）。

    timings.asr_done = 请求级 asr 墙钟跨度终点（网关对同请求取 max，
    起点=min 的 q_asr_last 即认领时刻）；asr_bucket = 桶起始时刻，仅
    进过推理桶的 job 携带——网关按同请求去重计数 asr_fw（桶数）。"""
    timings = {"q_asr": job.get("_qw_ms") or 0.0,
               "asr": asr_ms,
               # 请求级 q_asr 跨度：网关对同请求取
               # min(首 job 入队)/max(末 job 出队)
               "q_asr_first": job.get("_q_enq"),
               "q_asr_last": job.get("_q_claim"),
               "asr_done": time.time()}
    if bucket_ts is not None:
        timings["asr_bucket"] = bucket_ts
    ok = protocol.put_result(
        gateway_queues[int(job.get("origin", 0))],
        protocol.chunk_ts(job["req_id"], job["idx"], job["need"],
                          job["piece"]["samples"], timestamp,
                          error=error,
                          timings=timings), log)
    if not ok:
        _stats["job_drops"] += 1


def _run_asr_batch(jobs: list, gateway_queues, log, bid: str = "") -> None:
    """一批 asr job：SHM 读取（逐条隔离）→ 批推理 → 逐 job 回投。

    错误契约与 tts ship_batch 同构：SHM 类单 job 失败只 error 该 job
    （坏块不连坐）；推理批级失败 error 整批（compute 错误 → 各受影响
    请求 502）。本函数正常路径不抛；畸形载荷的字段引用错误（如缺
    req_id）例外——落到 asr_main 外层 except 兜底。

    timings.asr = 该 job 按音频时长分摊的桶墙钟（同请求各 chunk 求和 ≈
    该请求实际占用的对齐算力；不再每 job 复制整批时间、也不跨桶叠加）；
    _stats.asr_ms 仍累计推理墙钟总和（按桶累加一次，不随 B 膨胀）。"""
    import funasr_ts

    n = len(jobs)
    audios = [None] * n
    replied = [False] * n
    ts_out = [None] * n
    err_out = [None] * n
    asr_ms_out = [0.0] * n   # 逐 job 摊销后的对齐计算耗时（毫秒）
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
            try:
                _reply_chunk_ts(job, [], err_out[i], 0.0, gateway_queues, log)
            except Exception as rexc:  # noqa: BLE001  畸形载荷只废自家回投
                _stats["job_drops"] += 1
                log.warning("asr error reply failed req=%s idx=%s: %s",
                            job.get("req_id"), job.get("idx"),
                            str(rexc)[:160])
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
            try:
                _reply_chunk_ts(job, [], err_out[i], 0.0, gateway_queues, log)
            except Exception as rexc:  # noqa: BLE001  畸形载荷只废自家回投
                _stats["job_drops"] += 1
                log.warning("asr bad-metadata reply failed req=%s idx=%s: %s",
                            job.get("req_id"), job.get("idx"),
                            str(rexc)[:160])
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
        # 按音频时长摊销：同请求各 chunk 求和 = 该请求在该桶占用的算力份额
        # （batch 内共享同一面墙钟，不能每个 job 都记全量）。
        total_samples = sum(len(audios[i]) for i in idxs) or 1
        for i in idxs:
            asr_ms_out[i] = bucket_ms * len(audios[i]) / total_samples
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
            try:
                _reply_chunk_ts(job, [], err_out[i], asr_ms_out[i],
                                gateway_queues, log,
                                bucket_ts=bucket_of.get(i))
            except Exception as rexc:  # noqa: BLE001  畸形载荷只废自家回投
                _stats["job_drops"] += 1
                log.warning("asr error reply failed req=%s idx=%s: %s",
                            job.get("req_id"), job.get("idx"),
                            str(rexc)[:160])
        else:
            log.info("chunk=%s bid=%s process_ms=%.1f queue_ms=%.0f chars=%d",
                     _chunk_key(job), bid, infer_ms, _queue_ms(job),
                     len(ts_out[i] or []))
            try:
                _reply_chunk_ts(job, ts_out[i] or [], None, asr_ms_out[i],
                                gateway_queues, log,
                                bucket_ts=bucket_of.get(i))
            except Exception as rexc:  # noqa: BLE001  畸形载荷只废自家回投
                _stats["job_drops"] += 1
                log.warning("asr reply failed req=%s idx=%s: %s",
                            job.get("req_id"), job.get("idx"),
                            str(rexc)[:160])
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
    （一次 padded 批前向；显存预算 pipeline._asr_auto_budget_mb 随
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

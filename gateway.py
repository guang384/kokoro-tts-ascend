"""Kokoro TTS 服务（昇腾 NPU 版）· 统一队列架构（入口角色）。

- 入口进程：HTTP 接入 → 整单入 g2p_queue → g2p 池切句/G2P → tts 池认领
  组批/前向 → synth_part 逐段直投本入口 → 【本进程按 req_id 聚合】
  → 纯 TTS 直接编码返回；/tts 的 chunk 前向后即派发对齐（tts → asr → chunk_ts，
    与剩余合成重叠执行），音频与时间戳两账本独立收齐。
- G2P 在 g2p 池、FunASR 在 asr 池——入口进程无模型，内存轻、启动快。
- 入口进程间以 SO_REUSEPORT 均匀分接连接；日志带 [gw-N] 标签。
- 消息契约见 protocol.py（单一事实来源）；环境变量见 README.md。
"""
import asyncio
import base64
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

# HF_HUB_OFFLINE=1 由 proc_common 模块级强制（全仓库唯一一处；本模块
# import pipeline 时已经传递性地执行过）

import audio_enc
import gateway_agg
import gateway_obs
import gateway_state as gst
from gateway_state import _ClientGone, _SynthError  # 再导出：测试/端点沿用 gateway._SynthError
import pipeline
import proc_common
import protocol

# origin = 本网关在 gateway_queues 里的下标（supervisor 经 WORKER_IDX 注入）；
# 未注入（异常环境）按 0 兜底——绝不让 int("") 这类解析错误在首个请求时炸掉受理路径


# 模块级只取 logger，不在 import 时 setup_logging：supervisor 父进程在
# spawn 前 import 本模块取 target，import 副作用会把父进程的日志 tag
# 改成 pid/角色名（实测 supervisor 行变 [pid1]、本模块子进程行变 [pid44]）。
# 子进程入口 gateway_main 在设置 WORKER_TAG 后统一初始化 tag。
log = logging.getLogger("kokoro.server")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """启动：登记事件循环 + 起心跳与结果泵；退出：摘心跳文件。
    （FastAPI lifespan 写法，替代已弃用的 on_event。）"""
    gst._main_loop = asyncio.get_running_loop()
    gst._heartbeat_task = asyncio.get_running_loop().create_task(gateway_obs._heartbeat_loop())
    threading.Thread(target=gateway_agg._result_pump, args=(gst._result_q,), daemon=True).start()
    gst._voices_cache = set(pipeline.list_voices())  # 预热：免首请求在事件循环扫盘（pipeline 已随模块导入）
    log.info("gateway ready (G2P in g2p pool, FunASR in asr pool, no model here)")
    yield
    if gst._heartbeat_task is not None:
        gst._heartbeat_task.cancel()
        try:
            # 等取消落地再关循环：只 cancel 不 await 会留下 pending task，
            # 关闭事件循环时刷 "Task was destroyed but it is pending"。
            await gst._heartbeat_task
        except asyncio.CancelledError:
            pass
    try:  # 摘掉自己的心跳文件，聚合视图立即少一个 worker
        os.unlink(os.path.join(gst._health_dir, f"worker-{os.getpid()}.json"))
    except OSError:
        pass


app = FastAPI(
    title="Kokoro TTS (Ascend NPU)",
    version=gst._image_version,
    description="OpenAI-compatible TTS API for hexgrad/Kokoro-82M-v1.1-zh with dynamic batching",
    lifespan=_lifespan,
)

# 内网服务：允许任意源（tts-player 调试页面从 file:// 或独立静态服务器
# fetch 本服务时需要 CORS；无认证的纯内网 API，放开无额外暴露面）
from starlette.middleware.cors import CORSMiddleware

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _check_voice(voice: str) -> None:
    if gst._voices_cache is None:
        gst._voices_cache = set(pipeline.list_voices())
    if voice not in gst._voices_cache:
        raise HTTPException(400, f"未知音色: {voice}，可通过 GET /v1/voices 查询全部音色")


@app.get("/health")
async def health():
    """K8s 三探针共用（readiness/liveness/startup）：
    - ready=false（预热期/某角色零就绪）→ 503（必须配 startupProbe 宽限，
      否则预热期会被 liveness 反复重启）
    - ready=true → 200；degraded（预热失败/部分缺员但可服务）也 200。
    body 为精简结论卡，完整数值见 /metrics。聚合扫盘（心跳目录 listdir +
    逐文件 json.load）下放线程池——与 _asr_ready 自立的"事件循环不逐请求
    扫盘"规范一致。"""
    payload = await run_in_threadpool(gateway_obs.health_payload)
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)




@app.get("/metrics")
async def metrics():
    if not gst.METRICS_ENABLED:
        raise HTTPException(404, "metrics disabled (METRICS_ENABLED=0)")
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    body = await run_in_threadpool(
        lambda: generate_latest(gateway_obs._prom_registry()))  # collect+序列化下放线程池
    return Response(body, media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "kokoro-v1.1-zh", "object": "model", "owned_by": "hexgrad"}],
    }


@app.get("/v1/voices")
def voices():
    if gst._voices_cache is None:
        gst._voices_cache = set(pipeline.list_voices())
    return {"voices": sorted(gst._voices_cache)}


async def _synth(text: str, voice: str, speed: float, needs_align: bool,
                 req_id: str | None = None, release_shm: bool = True,
                 request: Request | None = None,
                 g2p_overrides: dict | None = None,
                 dtype: str = protocol.DTYPE_DEFAULT,
                 deadline: float | None = None):
    """合成入口：整单原子入 g2p_queue → g2p 池切句/G2P → tts 认领组批 →
    前向 → synth_part 逐段直投本入口 → 聚合收齐 need 句后返回。

    deadline：单一请求级截止线（perf_counter 绝对时刻）。本函数的音频
    等待只用剩余预算；None 按入口+WAIT_TIMEOUT_S 自算。/tts 由 handler
    传入入口时刻的 deadline，把余额留给对齐段——总时长硬上限 =
    WAIT_TIMEOUT_S（音频/对齐不再各 120s 串联）。

    返回 (audio, texts, fw, req_id, blocks_desc, timings, agg_s)；
    timings 为请求级阶段耗时（毫秒，{"q_g2p","g2p","q_tts","tts","chunks"}，
    由网关按 worker 捎带的跨度戳计算）；agg_s 为网关侧 SHM 读出+拼接耗时（秒）；
    audio 恒为拼接后的新数组（不回传 SHM 视图，杜绝悬垂）。
    release_shm=True（/v1）：聚合完成即收尾统一 unlink SHM；
    release_shm=False（/tts）：对齐阶段 asr 还要按名 attach 读块——本函数
    不清理，由 /tts 端点在全部 chunk_ts 收齐后回收；本函数自身的异常
    路径仍就地清理
    （请求终止时无人接手）。
    """
    # 语速钳制：帧数 = 预测时长 ÷ 语速，speed 无下限会把帧数（进而显存/时延
    # 上限）放大到不可控。下限默认 0.5，可用 TTS_SPEED_FLOOR 调整
    speed = max(gst._speed_floor, min(speed, 4.0))
    loop = asyncio.get_running_loop()
    req_id = req_id or uuid.uuid4().hex[:12]
    deadline = deadline if deadline is not None else (
        time.perf_counter() + gst._entry_wait_s)
    fut = loop.create_future()
    gst._pending[("req", req_id)] = fut
    gst._request_meta[req_id] = {"needs_align": bool(needs_align), "created": time.time()}

    # 断连取消：客户端断开 → 立即摘除本请求状态（后续 synth_part/chunk_ts
    # 走既有孤儿路径被丢弃）→ 写 CANCEL 标记（tts 认领后前向前过滤），
    # 已派发的批无法撤回但结果会被网关丢弃，不再浪费编码/对齐。
    async def _on_client_gone() -> None:
        if fut.done():
            return
        gst._pending.pop(("req", req_id), None)
        gst._pending.pop(("ts", req_id), None)
        gst._request_meta.pop(req_id, None)
        abandoned = gst._agg_state.pop(req_id, None)   # 已聚合块即时回收
        if abandoned:
            gateway_agg._unlink_blocks(abandoned["blocks"])
        try:
            proc_common.cancel_add(req_id)   # g2p 推送前 / tts 前向前检查跳过
        except Exception as exc:  # noqa: BLE001  redis 抖动：标记尽力而为，
            # 不能让 watcher 任务死掉——否则 fut 永不 resolve，请求白等到截止线
            log.warning("req=%s cancel_add failed (disconnect path, ignored): %s",
                        req_id, exc)
        gst._entry_stats["cancelled"] = gst._entry_stats.get("cancelled", 0) + 1
        log.info("req=%s client disconnected, request cancelled", req_id)
        if not fut.done():
            fut.set_exception(_ClientGone("客户端已断开"))

    poll_task = _watch_disconnect_task(loop, request, _on_client_gone)

    from multiprocessing import shared_memory

    try:
        # 准入快速拒绝（背压闸门 1 前置）：请求路径各段 EMA 排队等待求和
        # （队列空的段计 0——EMA 仅认领时更新，空队视为陈旧值），超过
        # 预算×冗余系数 → 排到它必然超截止线，秒拒不受理（429+Retry-After，
        # 好过受理后骑满 120s 才 504）。asr 段仅 /tts（对齐）参与——/v1
        # 不付对齐成本。redis 瞬断 fail-open：放行，截止线+超时取消兜底。
        pressures = []
        try:
            g2p_st = gst._g2p_queue.stats()
            pressures.append(("g2p", int(g2p_st.get("depth") or 0),
                              float(g2p_st.get("avg_wait_ms") or 0)))
            pressures.append(("tts",) + proc_common.claim_queue_pressure(
                proc_common.QUEUE_INDEX["tts"], "tts"))
            if needs_align:
                pressures.append(("asr",) + proc_common.claim_queue_pressure(
                    proc_common.QUEUE_INDEX["asr"], "asr"))
        except Exception as exc:  # noqa: BLE001
            log.warning("fast reject probe failed (fail-open): %s", exc)
        est_ms = sum(w for _s, d, w in pressures if d > 0)
        threshold_ms = gst._entry_wait_s * 1000 * gst.FAST_REJECT_MARGIN
        if est_ms > threshold_ms:
            gst._entry_stats["fast_reject"] += 1
            log.info(
                "fast reject req=%s est=%.2fs > threshold=%.2fs "
                "(budget=%.1fs×%.2f) %s",
                req_id, est_ms / 1000, threshold_ms / 1000,
                gst._entry_wait_s, gst.FAST_REJECT_MARGIN,
                " ".join(f"{s}(d={d},w={w / 1000:.2f}s)"
                         for s, d, w in pressures))
            raise HTTPException(
                429, "服务繁忙（排队积压超过等待预算），请稍后重试",
                headers={"Retry-After": "1",
                         "X-Reject-Reason": "wait_budget"})
        # 背压闸门 1.5：/dev/shm 水位。音频块在聚合/对齐期间驻留 tmpfs，
        # 写满即 SIGBUS（tts worker rc=-7 的历史事故）。90% 时拒绝新请求，
        # 给在途请求留出回收余量；非 POSIX（Windows 开发机）返回 -1 不拦。
        shm_pct = gateway_obs._shm_used_pct()
        if shm_pct >= 90.0:
            gst._entry_stats["rejections"] += 1
            gateway_obs._throttled_warn("shm_reject",
                            "/dev/shm 水位 %.0f%% >= 90%%，拒绝新请求（SIGBUS 防护）",
                            shm_pct)
            raise HTTPException(
                429, "服务繁忙（共享内存水位高），请稍后重试",
                headers={"Retry-After": "1",
                         "X-Reject-Reason": "shm_pressure"})
        # 整单入队（文本 KB 级）；切句/G2P 由 g2p 池执行。
        # 背压闸门 1：g2p_queue 满 → 立即 429（不受理不排队，连文本都不处理）
        try:
            gst._g2p_queue.put_nowait(protocol.new_request(
                req_id, gst._worker_origin, text, voice, speed, needs_align,
                g2p_overrides=g2p_overrides, dtype=dtype))
        except queue.Full:
            gst._entry_stats["rejections"] += 1
            raise HTTPException(429, "服务繁忙，请稍后重试",
                                headers={"Retry-After": "1",
                                         "X-Reject-Reason": "queue_full"})
        gst._entry_stats["requests"] += 1
        # 音频段只用剩余预算（单一截止线），余额留给 /tts 的对齐段
        r = await asyncio.wait_for(
            fut, timeout=max(deadline - time.perf_counter(), 0.0))
    except asyncio.TimeoutError:
        gst._entry_stats["timeouts"] += 1
        # 超时即取消：与断连同路下发 CANCEL，在途 chunk 认领后前向前被
        # 丢弃——死请求不再浪费前向，过载恢复更快。redis 抖动时吞掉：
        # 取消是尽力而为，绝不能把 504 变成 500（也让 _record_failure 留着）。
        try:
            proc_common.cancel_add(req_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("req=%s cancel_add failed (timeout path, ignored): %s",
                        req_id, exc)
        _record_failure(504, "推理或对齐超时")
        raise HTTPException(504, "推理超时，请稍后重试")
    except _ClientGone:
        raise HTTPException(499, "客户端已断开")
    except _SynthError as exc:
        # 服务端留痕：否则坏输入只会 400 给客户端，服务端日志无线索
        log.info("req=%s synth failed kind=%s: %s", req_id, exc.kind, exc)
        _record_failure(400 if exc.kind == protocol.KIND_INVALID else 502,
                        str(exc))
        raise HTTPException(400 if exc.kind == protocol.KIND_INVALID else 502,
                            str(exc))
    finally:
        if poll_task is not None:
            poll_task.cancel()
        gst._pending.pop(("req", req_id), None)   # 任何路径都摘除：防 gst._pending 泄漏
        gst._request_meta.pop(req_id, None)
        # 中止路径（超时/取消/断连）已聚合的部分块立即回收：否则要等
        # _sweep_leaks 的 2×WAIT(240s) 才清，正是 SHM 写满事故的加压条件
        abandoned = gst._agg_state.pop(req_id, None)
        if abandoned:
            gateway_agg._unlink_blocks(abandoned["blocks"])

    fw, texts, blocks_desc = r["fw"], r["texts"], r["blocks"]
    t_agg = time.perf_counter()   # agg 计时起点 = 音频账本收齐（fut 交付）

    def _gather_audio() -> np.ndarray:
        """SHM 读出 + 拼接（线程池内执行）：长文本的 SHM 合计可达几十 MB，
        frombuffer 逐句拷贝 + concatenate 若在事件循环上跑会阻塞同网关全部
        在途请求——与编码下放线程池同一原则。"""
        shms_ = []
        try:
            # 逐块 append（不能用列表推导）：第 k 块 attach 失败时推导式
            # 中止、赋值不发生，前 k-1 个已打开句柄会整体逃出 finally 的
            # close——fd/mmap 泄漏（块名被并发回收/Windows keepalive 淘汰
            # 后 attach 落空是真实可触发场景）
            for b in blocks_desc:
                shms_.append(shared_memory.SharedMemory(name=b["shm_name"]))
            pieces = []
            for b, shm in zip(blocks_desc, shms_):
                for s in b["segs"]:
                    # SHM 线格式即交付格式（PCM16）：全程 int16 直通，不再做
                    # float32 往返（原实现每请求 3-4 次全量格式转换）。
                    # copy 成自有数组——视图绝不跨语句存活（BufferError 防护）
                    pieces.append((s["idx"],
                                   np.frombuffer(
                                       shm.buf[s["offset"]:
                                               s["offset"] + s["samples"] * 2],
                                       dtype="<i2").copy()))
            pieces.sort(key=lambda t: t[0])  # 按句序（跨块）
            return audio_enc._concat([p for _i, p in pieces])
        except Exception:
            # 全量回收（含尚未 attach 的块）：attach 中途失败时，未打开的
            # 块名仍只在这里可见——只 unlink 已 attach 句柄会留下孤儿，
            # 要等 SHM_ORPHAN_MAX_AGE_S（默认 300s）清道夫。_unlink_blocks 幂等，finally 再 unlink 无害。
            gateway_agg._unlink_blocks(blocks_desc)
            raise
        finally:
            for shm in shms_:
                if release_shm:
                    try:
                        shm.unlink()   # 复用已打开句柄（免每块二次 attach）
                    except Exception:  # noqa: BLE001  已清理/名不存在：幂等
                        pass
                shm.close()

    audio = await run_in_threadpool(_gather_audio)
    agg_s = time.perf_counter() - t_agg
    return (audio, texts, fw, req_id, blocks_desc,
            r.get("timings") or {}, agg_s)


def _record_failure(status: int, reason: str) -> None:
    """记录一次请求级失败（时间 + 状态码 + 截断原因），/health 的
    recent_failures 展示最近 3 条（跨网关合并）。"""
    gst._entry_stats["failed"] += 1  # 终态失败计数（supervisor 卡死检测用）
    if status >= 500:            # 4xx 是客户端问题，不能把 /health 打成 degraded
        gst._entry_stats["server_5xx"] = gst._entry_stats.get("server_5xx", 0) + 1
        gst._entry_stats["last_5xx_ts"] = time.time()
    gst._recent_failures.append({"time": time.strftime("%H:%M:%S"),
                                  "status": status,
                                  "reason": str(reason)[:80]})




def _log_request_text(req_id: str, text: str) -> None:
    """请求文本日志策略（应用方要求）：INFO 打完整 TEXT（单行、换行替换为空格）。"""
    flat = text.replace(chr(10), " ").replace(chr(13), "")
    log.info("req=%s chars=%d text=%s", req_id, len(text), flat)


def _validate_request(text_raw, g2p_overrides, dtype_raw, voice_raw,
                      label: str, err):
    """端点输入校验的单一实现（/v1 与 /tts 同规则，仅字段名与异常类型不同）：
    空/超长文本、g2p_overrides/dtype 合法性、voice 缺省。
    err 为异常工厂（HTTPException / BizError）。返回 (text, overrides, dtype, voice)。"""
    if text_raw is None:
        raise err(400, f"缺少 {label} 参数")
    text = text_raw.strip()
    if not text:
        raise err(400, f"{label} 不能为空")
    if len(text) > gst._max_chars:
        raise err(400, f"{label} 超长({len(text)} > {gst._max_chars} 字符)，请分段请求")
    try:
        overrides = protocol.normalize_overrides(g2p_overrides)
        dtype = protocol.normalize_dtype(dtype_raw)
    except (ValueError, TypeError) as exc:
        raise err(400, str(exc))
    return text, overrides, dtype, (voice_raw or gst._default_voice).strip()


def _watch_disconnect_task(loop, request, on_fire):
    """断连探测 watcher 公共骨架（_synth 音频段与 /tts 对齐段共用）：
    0.5s 轮询 is_disconnected，断连时执行 on_fire()。探测异常 ≠ 已断连：
    直接退出不误判（落穿会把活连接判 499），该请求由正常超时/完成路径
    收尾；watcher 死了只损失"提前取消"能力。request 为 None（测试直调）
    返回 None。"""
    if request is None:
        return None

    async def _watch() -> None:
        try:
            while not await request.is_disconnected():
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception(
                "disconnect watcher probe error (watcher exits, request continues)")
            return
        await on_fire()

    return loop.create_task(_watch())


def _log_request_done(ep: str, rid: str, voice: str, speed: float,
                      audio_s: float, e2e_s: float, agg_s: float,
                      enc_s: float) -> None:
    """成功收尾（results 计数 + 单行汇总）：/v1 与 /tts 同格式同口径。
    只带网关独有字段——批/句级明细（排队、前向、对齐）由 tts/asr worker
    的 bid=/chunk= 行承担——grep <req_id> 串全程，grep <bid> 看批次构成。"""
    gst._entry_stats["results"] += 1   # 成功口径 = 完整响应体已就绪（/tts = 音频+对齐+编码全部完成）
    log.info("ep=%s req=%s voice=%s speed=%.2f audio_s=%.1f e2e_s=%.2f "
             "agg_ms=%.0f enc_ms=%.0f",
             ep, rid, voice, speed, audio_s, e2e_s, agg_s * 1000, enc_s * 1000)


class _BodySizeLimitMiddleware:
    """请求体上限 + Content-Type 归一（纯 ASGI）：Pydantic/JSON 解析前先看
    content-length——超大或恶意请求体先占内存再在事件循环上解析的路径
    直接 413 拒绝（内网无害，对外暴露时的基本防线）。分块编码无
    content-length 的由框架自然限流。

    Content-Type 归一：本服务端点只收 JSON 体，而 curl --data-raw 不带
    Content-Type（curl --data-raw 默认 application/x-www-form-urlencoded），
    FastAPI 对 JSON 模型体遇 urlencoded 会按表单解析 → 422。缺失或
    urlencoded 时把头部改写为 application/json（纯头部操作，零缓冲、
    不碰 receive 通道）。仅处理 POST；multipart 等其他类型不干预。

    必须是纯 ASGI：@app.middleware("http") 会套 BaseHTTPMiddleware，
    它包裹 receive 通道导致 request.is_disconnected() 恒为 False——
    断连取消链路（499/CANCEL/即时回收）整体失效（E2E 实测发现，
    生产一直在浪费断开客户端的推理算力）。"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers")
            if scope.get("method") == "POST" and headers is not None:
                ct_idx = next((i for i, (k, _v) in enumerate(headers)
                               if k == b"content-type"), None)
                ct = (headers[ct_idx][1].decode("latin-1").lower()
                      if ct_idx is not None else "")
                if not ct or ct.startswith("application/x-www-form-urlencoded"):
                    new_ct = b"application/json"
                    if ct_idx is not None:
                        headers[ct_idx] = (b"content-type", new_ct)
                    else:
                        headers.append((b"content-type", new_ct))
            for k, v in scope.get("headers") or []:
                if k != b"content-length":
                    continue
                if v.isdigit() and int(v) > gst._MAX_BODY_BYTES:
                    # 丢弃已到达的 body 再回 413（有界 8MB / 2s）：HTTP/1.1
                    # 客户端通常先把整个 body 写完才读响应，服务端不读就关连接
                    # 会让客户端收到 RST 而不是 413（verify_boundaries 实网复现）。
                    # 2s 上限防止慢速发送者（slowloris）白占一个请求任务；超出
                    # 丢弃上限的恶意大体积仍以断连收场，属于可接受降级。
                    discarded = 0
                    deadline = time.monotonic() + 2.0
                    try:
                        while discarded < 8 * 2**20:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            msg = await asyncio.wait_for(receive(), timeout=remaining)
                            if msg["type"] == "http.disconnect":
                                break
                            discarded += len(msg.get("body") or b"")
                            if not msg.get("more_body", False):
                                break
                    except Exception:  # noqa: BLE001  超时/客户端断开/协议异常
                        pass
                    await JSONResponse(status_code=413, content={
                        "detail": f"请求体过大（上限 {gst._MAX_BODY_BYTES // 2**20}MB）"
                    })(scope, receive, send)
                    return
                break
        await self.app(scope, receive, send)


app.add_middleware(_BodySizeLimitMiddleware)


class SpeechRequest(BaseModel):
    model: str = "kokoro-v1.1-zh"
    input: str = Field(..., min_length=1, description="待合成文本")
    voice: str | None = None
    response_format: Literal["wav", "pcm", "mp3", "flac", "opus", "aac"] = "wav"
    speed: float = Field(1.0, gt=0, le=4.0)
    g2p_overrides: dict[str, str | list[str]] | None = Field(
        None, description="请求级多音字覆盖 {词组: TONE3 拼音串/列表}，仅本请求生效")
    # Any 而非 str：非字符串输入（16/[]/{...}）也交给 protocol.normalize_dtype
    # 统一报 400 并列出可选值——Pydantic 的 str 类型会先拦成 422（只说"应为字符串"）
    dtype: Any = Field(
        None, description=f"计算精度，可选 {'/'.join(protocol.DTYPE_CHOICES)}"
                          f"（缺省 {protocol.DTYPE_DEFAULT}）")


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest, request: Request = None):
    t_e2e = time.perf_counter()   # e2e 起点：进处理函数（校验/排队/合成/编码全程）
    text, overrides, dtype, voice = _validate_request(
        req.input, req.g2p_overrides, req.dtype, req.voice,
        label="input", err=HTTPException)

    t0 = time.perf_counter()
    try:
        _check_voice(voice)
        (audio, _texts, fw, rid, _blocks,
         timings, agg_s) = await _synth(
            text, voice, req.speed, needs_align=False,  # OpenAI 兼容接口不付对齐成本
            request=request, g2p_overrides=overrides, dtype=dtype,
            deadline=t_e2e + gst._entry_wait_s)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed voice=%s", voice)
        _record_failure(500, f"合成失败: {exc}")
        raise HTTPException(500, f"合成失败: {exc}")
    elapsed = time.perf_counter() - t0
    duration = len(audio) / gst._sample_rate
    gateway_obs._record_latency(elapsed, len(text))
    _log_request_text(rid, text)

    fmt = req.response_format
    t_enc = time.perf_counter()
    try:
        if fmt == "wav":
            mime = "audio/wav"
            body = await run_in_threadpool(audio_enc._encode_wav, audio)
        elif fmt == "flac":
            mime = "audio/flac"
            body = await run_in_threadpool(audio_enc._encode_flac, audio)
        elif fmt == "pcm":
            mime = "audio/pcm"
            # 一样下放线程池：满额请求 PCM 20-30MB，tobytes 一次拷贝 10-30ms
            body = await run_in_threadpool(audio_enc._encode_pcm, audio)
        else:
            mime = {"mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac"}[fmt]
            body = await run_in_threadpool(audio_enc._encode_ffmpeg, audio, fmt)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))
    enc_s = time.perf_counter() - t_enc
    _log_request_done("/v1", rid, voice, req.speed, duration,
                      time.perf_counter() - t_e2e, agg_s, enc_s)

    return Response(
        content=body,
        media_type=mime,
        headers={
            "X-Request-ID": rid,  # 与各 worker 日志的 req= 对齐，客户端报障可凭此定位
            "X-Audio-Duration": f"{duration:.3f}",
            "X-Synth-Time": f"{elapsed:.3f}",
            "X-Batch-Size": str(fw),  # 该请求的前向批次数（>1 = 多批聚合而成）
        },
    )


class TTSRequest(BaseModel):
    """业务兼容接口（对齐业务报文契约：成功返回 audio_base64/code/duration/
    message/text/timestamp 6 字段，错误返回 {"error": msg}）。"""
    text: str | None = None
    speed: float = Field(1.0, gt=0, le=4.0)
    voice: str | None = None
    g2p_overrides: dict[str, str | list[str]] | None = Field(
        None, description="请求级多音字覆盖 {词组: TONE3 拼音串/列表}，仅本请求生效")
    # Any 而非 str：非字符串输入（16/[]/{...}）也交给 protocol.normalize_dtype
    # 统一报 400 并列出可选值——Pydantic 的 str 类型会先拦成 422（只说"应为字符串"）
    dtype: Any = Field(
        None, description=f"计算精度，可选 {'/'.join(protocol.DTYPE_CHOICES)}"
                          f"（缺省 {protocol.DTYPE_DEFAULT}）")


class BizError(Exception):
    """业务接口错误：响应体 {"error": msg}（对齐 Flask jsonify 版本）。"""

    def __init__(self, status_code: int, message: str, headers: dict | None = None):
        self.status_code = status_code
        self.message = message
        self.headers = headers


@app.exception_handler(BizError)
async def biz_error_handler(request, exc: BizError):
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.message}, headers=exc.headers)


def _scan_asr_ready() -> bool:
    """健康目录里是否有就绪的 asr 工人（/tts 快速失败，避免对齐单永悬挂）。"""
    for _name, path, _age in gateway_obs._health_dir_entries():
        try:
            with open(path, encoding="utf-8") as f:
                w = json.load(f)
        except (OSError, ValueError):
            continue
        if w.get("role") == "asr" and (w.get("warm") or {}).get("funasr") == "ok":
            return True
    return False


def _asr_ready() -> bool:
    """带 TTL 缓存的 asr 就绪探测：扫描是 listdir+mtime+json.load 的同步
    磁盘 IO，且 /tts 每请求都要过这道闸——绝不能在事件循环上逐请求扫盘。"""
    now = time.time()
    c = gst._asr_ready_cache
    if now - c["ts"] >= gst._ASR_READY_TTL_S:
        c["val"] = _scan_asr_ready()
        c["ts"] = now
    return c["val"]


@app.post("/tts")
async def tts_business(req: TTSRequest, request: Request = None):
    """业务兼容接口：返回 {audio_base64, duration, text, timestamp}。

    错误响应体为 {"error": msg}（对齐 Flask 版本）：400 缺少/空 text、
    429 服务繁忙（Retry-After）、503 时间戳服务未就绪、504 超时。
    timestamp 为字级毫秒时间戳 [[起ms, 止ms], ...]，由 asr 池
    （FunASR speech_timestamp_prediction-v1-16k-offline）经 chunk 级流水产出。
    """
    t_e2e = time.perf_counter()   # e2e 起点：进处理函数（校验/排队/合成/对齐/编码全程）
    deadline = t_e2e + gst._entry_wait_s   # 单一请求级截止线：音频+对齐共享预算
    text, overrides, dtype, voice = _validate_request(
        req.text, req.g2p_overrides, req.dtype, req.voice,
        label="text", err=BizError)

    if not _asr_ready():
        raise BizError(503, "时间戳服务暂不可用（asr 池未就绪），请稍后重试")

    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    req_id = uuid.uuid4().hex[:12]
    # 时间戳 Future 必须在 _synth 之前注册：chunk_ts 可能先于音频聚合到达
    # 立刻就能命中，不存在"先收到通知后登记"的时序窗口
    fut_ts = loop.create_future()
    gst._pending[("ts", req_id)] = fut_ts
    blocks_desc = None
    try:
        _check_voice(voice)
        (audio, _texts, fw, _rid, blocks_desc,
         synth_timings, agg_s) = await _synth(
            text, voice, req.speed, needs_align=True, req_id=req_id,
            release_shm=False,  # 对齐期间 asr 还要按名读块，这里不回收
            request=request, g2p_overrides=overrides, dtype=dtype,
            deadline=deadline)
        # 对齐段断连 watcher：音频段完成后 _synth 的 watcher 已随 finally
        # 取消，此后的对齐等待（长文本可达几十秒）原感知不到断开——现在
        # 立即 499 释放连接与受理额度（对齐/asr 已派发部分照旧算完，结果
        # 按孤儿丢弃）
        async def _on_ts_client_gone() -> None:
            if not fut_ts.done():
                fut_ts.set_exception(_ClientGone("客户端已断开"))

        ts_poll = _watch_disconnect_task(loop, request, _on_ts_client_gone)
        try:
            # 对齐段用剩余预算（单一截止线）：音频耗掉多少，对齐就只剩多少
            ts = await asyncio.wait_for(
                fut_ts, timeout=max(deadline - time.perf_counter(), 0.0))
        finally:
            if ts_poll is not None:
                ts_poll.cancel()
    except asyncio.TimeoutError:
        # 只可能是对齐段超时（_synth 内部的音频超时已转 HTTPException）：
        # 必须计入 timeouts/failed，否则 /health、/metrics、wedge 都看不到
        # "音频成功、对齐永远等不到"的故障。
        gst._entry_stats["timeouts"] += 1
        _record_failure(504, "推理或对齐超时")
        raise BizError(504, "推理或对齐超时，请稍后重试")
    except _ClientGone:
        gst._entry_stats["cancelled"] = gst._entry_stats.get("cancelled", 0) + 1
        raise BizError(499, "客户端已断开")
    except _SynthError as exc:
        # 对齐服务故障 → 502 / 输入问题 → 400（原实现落到 generic 500）
        status = 400 if exc.kind == protocol.KIND_INVALID else 502
        log.info("req=%s alignment failed kind=%s: %s", req_id, exc.kind, exc)
        _record_failure(status, str(exc))
        raise BizError(status, str(exc))
    except HTTPException as exc:  # 429/502/504 等，转业务格式
        raise BizError(exc.status_code, str(exc.detail), exc.headers)
    except Exception as exc:  # noqa: BLE001
        log.exception("synthesis failed voice=%s", voice)
        _record_failure(500, f"TTS 生成失败: {exc}")
        raise BizError(500, f"TTS 生成失败: {exc}")
    finally:
        gst._pending.pop(("ts", req_id), None)
        gst._ts_state.pop(req_id, None)   # TS 账本就地回收（审计 F6 对称性）
        retired = gst._retired_blocks.pop(req_id, None)
        if retired:
            gateway_agg._unlink_blocks([b for b, _t in retired])
        if blocks_desc is not None:
            gateway_agg._unlink_blocks(blocks_desc)  # 收尾统一回收（chunk_ts 收齐后块已无人读）

    tts_cost = time.perf_counter() - t0
    duration = len(audio) / gst._sample_rate

    t1 = time.perf_counter()
    ts_list = ts.get("timestamp") or []
    # 大 payload 序列化整体下放线程池：base64 + json.dumps 都不在事件循环上
    # 跑（长文本的 base64 可达几十 MB，事件循环内同步 dumps 会阻塞同网关
    # 全部在途请求），回预构字节绕开 JSONResponse 的同步 render
    payload = {
        "audio_base64": None,  # 占位，线程内填充
        "code": 0,
        "duration": round(duration, 3),
        "message": "success",
        "text": req.text,   # 契约字段（README/冒烟测试）
        "timestamp": ts_list,
    }
    # g2p 兼容别名已移除：对外 6 键契约不含此字段，多键对严格
    # 校验的消费方是兼容性风险
    body = await run_in_threadpool(
        lambda: json.dumps(
            {**payload, "audio_base64": base64.b64encode(
                audio_enc._encode_wav(audio)).decode("utf-8")},
            ensure_ascii=False).encode("utf-8"))
    enc_cost = time.perf_counter() - t1
    gateway_obs._record_latency(tts_cost, len(req.text))  # 延迟直方图口径不变 = 合成+对齐全程
    _log_request_done(request.url.path if request is not None else "/tts",
                      req_id, voice, req.speed, duration,
                      time.perf_counter() - t_e2e, agg_s, enc_cost)
    _log_request_text(req_id, text)

    return Response(content=body, media_type="application/json",
                    headers={"X-Request-ID": req_id})


@app.exception_handler(Exception)
async def unhandled(request, exc):
    log.exception("unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def gateway_main(i: int, port: int, g2p_queue, result_q) -> None:
    """supervisor 经 mp.Process 启动网关进程：注入队列句柄 → 绑定 → 跑 uvicorn。

    Linux 生产：自建 socket 并设 SO_REUSEPORT——每个网关独立监听 socket，
    内核按四元组哈希均匀分发（多网关负载均衡的关键）；Windows 冒烟：
    无 SO_REUSEPORT，每网关绑定独立端口（port+1000+i）。"""
    gst._result_q = result_q
    gst._g2p_queue = g2p_queue
    gst._worker_origin = int(i)
    os.environ["WORKER_TAG"] = f"gw-{i + 1}"
    # 进程级日志 tag 在子进程入口初始化（import 时不能做：父进程会 import
    # 本模块取 target，import 副作用会污染 supervisor 的 [supervisor] 标签）。
    proc_common.setup_logging(f"gw-{i + 1}")
    import socket
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 重启后 TIME_WAIT 不阻塞 bind
    if hasattr(socket, "SO_REUSEPORT"):  # Linux 生产：同端口多网关，内核哈希分发
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        bind_port = port
    else:  # Windows 冒烟：每网关独立端口（跨进程共享 accept 与 asyncio proactor 不兼容）
        bind_port = port + 1000 + i
    sock.bind(("0.0.0.0", bind_port))
    sock.listen(2048)
    config = uvicorn.Config(app, log_level=proc_common.log_level_name(),
                            timeout_keep_alive=proc_common.env_int(
                                "KEEP_ALIVE_TIMEOUT_S", minimum=0),
                            # 优雅退出上限 4s < supervisor join(5)：在途响应
                            # 自己收干净，避免 5s 后被 SIGKILL 硬切
                            timeout_graceful_shutdown=4)
    server = uvicorn.Server(config)
    asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    raise SystemExit("gateway 由 supervisor.py 编排启动（注入队列句柄 + REUSEPORT 监听），"
                     "不可直接运行。本地联调用 supervisor.py；接口冒烟用 tests/test_tts_api.py。")

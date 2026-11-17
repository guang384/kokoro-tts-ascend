#!/usr/bin/env python3
"""text pool（Stage 1 文本段）：从前置队列拉原始请求 → 切句/G2P → 投音素队列。

流水线定位：gateway（HTTP）→ g2p_queue → 【本池】→ tts_queue（ZSET 认领式）→ tts 池。
- G2P_WORKERS 线程池在本池（消灭入口间 G2P 不均）；
- chunk 消息由 protocol.chunk 构造：need=请求 chunk 总数（全部句子出音素
  后才知晓）；相似长度凑批由认领脚本的类宽承担（chunk 不带桶）。
- 音色合法性校验在 gateway._check_voice（入队前拦截），本池不重复。
"""
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import proc_common
import protocol

log = proc_common.setup_logging(os.getenv("WORKER_TAG", "g2p"))

G2P_WORKERS = proc_common.env_int("G2P_WORKERS", minimum=1)   # 进程数（supervisor 用；此处仅作线程数默认值）
_G2P_THREADS = proc_common.env_int("G2P_THREADS", G2P_WORKERS, minimum=1)  # 本进程音素化线程池大小（默认随进程数）
_max_chunk = proc_common.env_int("TTS_CHUNK_CHARS", minimum=8)
_speed_floor = proc_common.env_float("TTS_SPEED_FLOOR", minimum=0.1)
_CHUNK_SEC_PER_CHAR = 17.5 / 80.0   # 0.21875s/字@speed1.0（17.5帧/token 校准 ÷80fps，memcalc 同锚点）
_g2p_cache_size = proc_common.env_int("G2P_CACHE_SIZE", minimum=0)
_SENT_RE = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|\n+|$)")
_started_at = time.time()
_stats = {"requests": 0, "chunks": 0, "g2p_ms": 0.0, "errors": 0,
          "override_requests": 0}  # 全键预置（心跳线程快照要求）
_warm = {"g2p": "pending"}
_ph_cache = None  # 本 worker 的带缓存 phonemize（main 里装配，心跳统计读）
_busy = proc_common.BusyWatch()  # 看门狗：G2P 内嵌 espeak 子进程等外呼卡死可检测


def make_phonemizer(engine, maxsize: int):
    """包装 engine.phonemize 成 LRU 缓存版（纯函数记忆化）。

    生产 TTS 流量中固定短语/重试/问候语占比可观，同句重复 tokenization
    （jieba+misaki，纯 CPU）是白烧的延迟。返回 list 防缓存条目被外改；
    maxsize=0 完全关闭缓存。cache_info 挂到返回值上供心跳统计。"""
    @lru_cache(maxsize=maxsize)
    def _cached(sentence: str):
        return tuple(engine.phonemize(sentence))

    def phonemize(sentence: str):
        return list(_cached(sentence))

    phonemize.cache_info = _cached.cache_info
    return phonemize


def _chunk_cap(speed: float) -> int:
    """语速感知切句上限（根因层）：audio_sec ≈ chars × _CHUNK_SEC_PER_CHAR
    ÷ speed → 上限取「单 chunk 音频 ≲ ASR_MAX_GROUP_SEC−2s（校准余量）」。
    默认上限 120 字 @speed1.0 = 26.25s 本就不超标；speed < 0.9375 起线性
    收紧（0.5 → 64 字 ≈ 28s）。floor 8 = TTS_CHUNK_CHARS 下限。
    funasr_ts._split_groups 另有超长兜底硬切（校准偏差/直调路径）——
    asr 组帧数硬封顶（npu_ops._asr_auto_budget_mb 预算前提）两层保证。"""
    return min(_max_chunk,
               max(8, int((proc_common.ASR_MAX_GROUP_SEC - 2.0)
                          / _CHUNK_SEC_PER_CHAR * speed)))


def split_text(text: str, max_chars: int) -> list:
    """按标点切句 + 合并到 max_chars + 超长硬切；直接返回句列表。"""
    sents = [m.group(0).strip() for m in _SENT_RE.finditer(text) if m.group(0).strip()]
    merged, buf = [], ""
    for s in sents:
        if not s:
            continue
        if len(buf) + len(s) <= max_chars:
            buf += s
            continue
        if buf:
            merged.append(buf)
        while len(s) > max_chars:
            merged.append(s[:max_chars])
            s = s[max_chars:]
        buf = s
    if buf:
        merged.append(buf)
    return merged if merged else [text]


def make_chunks(req: dict, sentences: list, ps_lists: list,
                q_g2p_ms: float | None = None,
                g2p_ms: float | None = None) -> list:
    """句子+音素列表 → 协议 chunk 消息列表（纯函数，供测试锁定契约）。

    need 在全部句子出音素后回填（请求 chunk 总数）。
    q_g2p_ms/g2p_ms：g2p 阶段耗时随行（队列等待/计算，毫秒）——每条 chunk
    携带同值，网关聚合取 max；None 不落消息。
    每 chunk 附 sents（该 chunk 合并的句子数，按句读正则计数）——chunk 行
    展示字段；chars 由消费端按 len(text) 派生，不进协议。"""
    chunks = []
    for chunk_text, ps_chunk in zip(sentences, ps_lists):
        for ps in ps_chunk:
            chunks.append({"text": chunk_text, "ps": ps})
    need = len(chunks)
    out = []
    for i, c in enumerate(chunks):
        out.append(protocol.chunk(
            req_id=req["req_id"], idx=i, ps=c["ps"], voice=req["voice"],
            speed=req["speed"], origin=req["origin"], need=need,
            text=c["text"], needs_align=bool(req.get("needs_align")),
            dtype=req.get("dtype") or protocol.DTYPE_DEFAULT,
            q_g2p_ms=q_g2p_ms, g2p_ms=g2p_ms,
            sents=len(_SENT_RE.findall(c["text"]))))
    return out


def publish_chunks(chunks: list, req: dict, tts_queue, gateway_queues,
                   log) -> bool:
    """把 chunk 逐条投进 tts 认领队列（阻塞 put = 背压闸门 2）。

    部分成功是危险态：已入队的 chunk 带 need=N，网关会一直等缺的 idx。
    失败时尽力做两件事——给 origin 网关发 synth_error（让请求快速 502，
    而不是干等 WAIT_TIMEOUT 504），并写 cancel 标记（tts 认领后、前向前
    过滤掉已入队的 chunk）。Redis 本身不可达时两者也会失败，此时由网关
    截止线兜底；但绝不能静默只打一行日志。"""
    req_id = req.get("req_id", "")
    try:
        for c in chunks:
            tts_queue.put(c)
        return True
    except Exception as exc:  # noqa: BLE001
        _stats["errors"] = _stats.get("errors", 0) + 1
        log.warning("chunk publish failed req=%s: %s", req_id, str(exc)[:160])
        try:
            protocol.put_result(
                gateway_queues[int(req.get("origin", 0))],
                protocol.synth_error(
                    req_id, f"文本分发失败: {str(exc)[:160]}",
                    protocol.KIND_COMPUTE), log)
        except Exception:  # noqa: BLE001  尽力而为
            pass
        try:
            proc_common.cancel_add(req_id)
        except Exception:  # noqa: BLE001  尽力而为
            pass
        return False


def main(idx: int, g2p_queue, tts_queue, gateway_queues) -> None:
    global _ph_cache

    import pipeline

    tag = f"g2p-{idx}"
    log = proc_common.setup_logging(tag)

    def _stats_snapshot():
        out = dict(_stats)
        out["ipc_dropped"] = protocol.drop_count()
        if _ph_cache is not None:
            ci = _ph_cache.cache_info()
            out["cache_hits"], out["cache_misses"] = ci.hits, ci.misses
        return out

    proc_common.start_heartbeat(  # 先心跳后加载：预热期可见、失败不隐身
        tag=tag, pid=os.getpid(),
        stats_fn=lambda: {"role": "g2p", "ready": _warm["g2p"] == "ok",
                          "uptime_s": round(time.time() - _started_at, 1),
                          "stuck_s": _busy.stuck_s(),
                          "warm": dict(_warm), "stats": _stats_snapshot()},
    )
    # 初始化失败驻留重试（proc_common.retry_forever 共用骨架）：失败瞬态
    # （依赖/资源未就绪）可自愈，不再永久躺平——get_g2p 只在成功后缓存
    # 单例，重试安全
    def _init():
        g2p = pipeline.get_g2p()
        g2p.phonemize("你好。")  # 预热（jieba 词典/misaki 首载）
        _warm["g2p"] = "ok"
        log.info("G2P text engine ready")
        return g2p

    g2p = proc_common.retry_forever(_init, "g2p", _warm, "G2P", log)

    pool = ThreadPoolExecutor(max_workers=_G2P_THREADS, thread_name_prefix="g2p")
    _ph_cache = make_phonemizer(g2p, _g2p_cache_size)
    if _g2p_cache_size:
        log.info("g2p sentence cache enabled: maxsize=%d", _g2p_cache_size)

    def process(req: dict) -> None:
        origin = req["origin"]
        req_id = req["req_id"]
        t0 = time.perf_counter()
        # 看门狗只覆盖 G2P 计算段：tts 队列满时的阻塞 put 是设计内
        # 背压（等下游腾空间），不是 hang——盖进 busy 窗口会让过载期的
        # g2p 被看门狗误杀循环（生产事故 20261108：tts 池死亡 → 队列顶满
        # → g2p 每 300s 被杀一次，白丢请求还反复付预热代价）
        _busy.enter()
        try:
            speed = max(_speed_floor, min(float(req.get("speed", 1.0)), 4.0))
            sentences = split_text(req["text"], _chunk_cap(speed))
            raw_sents = sum(1 for m in _SENT_RE.finditer(req["text"])
                            if m.group(0).strip())
            ovr = req.get("g2p_overrides")
            if ovr:
                # 请求级多音字覆盖：音素依赖 overrides，同句不同覆盖结果
                # 不同——绕句子缓存（写缓存会污染后续无覆盖请求），一次
                # 注入/还原覆盖整单；覆盖请求罕见，串行执行即可
                _stats["override_requests"] += 1
                ps_lists = g2p.phonemize_many(sentences, overrides=ovr)
            else:
                # G2P 并行（CPU 线程池——全局共享的文本段算力；命中缓存的
                # 句子直接返回，不再过 jieba/misaki）
                ps_lists = list(pool.map(_ph_cache, sentences))
        except Exception as exc:  # noqa: BLE001
            _stats["errors"] = _stats.get("errors", 0) + 1
            # 输入格式问题（覆盖词典/编码异常）→ 400；G2P 依赖/内存等
            # 内部故障 → 502。原实现一律 KIND_INVALID，把服务故障伪装成
            # 客户端错误，健康指标也看不到。
            kind = (protocol.KIND_INVALID
                    if isinstance(exc, (ValueError, TypeError))
                    else protocol.KIND_COMPUTE)
            log.warning("text processing failed req=%s kind=%s: %s",
                        req_id, kind, str(exc)[:120])
            protocol.put_result(gateway_queues[origin], protocol.synth_error(
                req_id, f"文本处理失败: {str(exc)[:160]}", kind), log)
            return
        finally:
            _busy.exit()
        g2p_ms = (time.perf_counter() - t0) * 1000
        # 阶段耗时随行：_qw_ms 由队列 get 注入（g2p 队列等待）；缺戳消息/外部
        # 直写无此键 → None 不落消息，网关按 0 兜底
        chunks = make_chunks(dict(req, speed=speed), sentences, ps_lists,
                             q_g2p_ms=req.get("_qw_ms"), g2p_ms=g2p_ms)
        if not chunks:
            _stats["errors"] = _stats.get("errors", 0) + 1
            log.warning("no pronounceable content req=%s", req_id)
            protocol.put_result(gateway_queues[origin], protocol.synth_error(
                req_id, "文本不含可发音内容", protocol.KIND_INVALID), log)
            return
        # 逐句投认领队列（阻塞式 put=背压闸门 2：tts 队列满则等待，
        # 上游 g2p_queue 跟着填满 → 网关 429）。先确认全部投递成功再
        # 计 requests/chunks——部分投递失败由 publish_chunks 发 502。
        if not publish_chunks(chunks, req, tts_queue, gateway_queues, log):
            return
        _stats["requests"] += 1
        _stats["chunks"] += len(chunks)
        _stats["g2p_ms"] += g2p_ms
        log.info("req=%s queue_ms=%.1f process_ms=%.1f chunks=%d ovr=%d "
                 "chars=%d sents=%d",
                 req_id, float(req.get("_qw_ms") or 0.0), g2p_ms,
                 len(chunks), len(req.get("g2p_overrides") or {}),
                 len(req.get("text") or ""), raw_sents)

    while True:
        try:
            req = g2p_queue.get()
        except Exception:  # noqa: BLE001  坏消息/Redis 瞬断：保持工人存活
            log.exception("g2p queue get failed (worker kept alive, retry)")
            time.sleep(0.2)
            continue
        try:
            process(req)
        except Exception:  # noqa: BLE001  process 内部已兜主路径，这里防
            # make_chunks/投递等残余路径炸死工人（外层裸奔 = 每个意外异常
            # 都要付出一次进程重启+模型预热的代价）
            log.exception("request handling error req=%s", req.get("req_id"))


if __name__ == "__main__":
    raise SystemExit("g2p 池由 supervisor.py 编排启动（mp spawn 注入队列句柄），"
                     "不可直接运行。本地联调请用 supervisor.py 或 tests/ 下的单测。")

"""跨进程消息协议：类型常量 + 构造函数（单一事实来源）。

流水线全部跨进程队列消息的字段清单集中在此：任何一端要读的字段，
必须由这里的构造函数保证存在。历史上 g2p/batcher/compute 各自
手写字典，字段漂移导致整条管道断裂（missing bucket/need/req_id/
needs_align、origin 取错下标、busy 无 type 被丢弃）——本模块即防再犯。

消息一览（生产者 → 队列 → 消费者）：
  gateway → g2p_queue            NEW_REQUEST   原始请求（req_id/origin/needs_align/g2p_overrides
                                                由 gateway 定）
  g2p     → tts_queue(ZSET)      CHUNK         句子 chunk（need=请求总 chunk 数；score=FIFO 序号，
                                                member=长度头+精度+序号+载荷；认领式——worker
                                                按「队列快照」组批，无独立攒批进程）
  g2p     → gateway_queues[o]    SYNTH_ERROR   kind=invalid_input（文本不可发音等）
  gateway → kokoro:cancel(ZSET)  ──(非消息)──  断连取消标记（score=时刻）：tts 认领后、
                                                前向前检查跳过；600s 由 sweep 清扫
  tts     → {queue}:infl:{worker}(ZSET) ──(非消息)── 认领即标记（at-least-once）：
                                                worker 死亡 → requeue_worker 精确重派；
                                                TTL 只兜 supervisor 漏检的极端
  tts     → gateway_queues[o]    SYNTH_PART    每请求一段音频（SHM 块描述符），gateway 按 req_id 聚合
  tts     → gateway_queues[o]    SYNTH_ERROR   kind=compute（前向失败，批内全部受影响请求）
  tts     → asr_queue            ASR_JOB      chunk 级对齐作业（needs_align 的 chunk 前向后
                                                即派发——ASR 与剩余 TTS 重叠执行，无 aligner 进程）
  asr     → gateway_queues[o]    CHUNK_TS     单 chunk 字级时间戳（含 need，乱序到达，
                                                网关 TS 账本收齐拼接）

SHM 块归属：一个块只属于一个请求（tts 按请求分块写入）；清理者唯一
——终态接收方 gateway（/v1 编码完即清；/tts 在 align_done 后清，
任何超时/异常路径兜底清理）。
"""
import re

# ── 消息类型常量（gateway 结果队列里多态分派用）──

# ── 计算精度（请求级）：fp32 默认 | fp16 autocast（同一 fp32 权重，
#    前向按算子自动选型；凑批类键拼入 dtype 避免混批——autocast 是
#    整批上下文，一批只能一种精度）──
DTYPE_DEFAULT = "fp32"
DTYPE_CHOICES = ("fp32", "fp16")


def normalize_dtype(value, field: str = "dtype") -> str:
    """校验请求精度字段；非法值报错并列出可选值。"""
    if value is None or value == "":
        return DTYPE_DEFAULT
    v = str(value).strip().lower()
    if v not in DTYPE_CHOICES:
        raise ValueError(
            f"{field} 非法值 {value!r}，可选：{'/'.join(DTYPE_CHOICES)}"
            f"（缺省 {DTYPE_DEFAULT}）")
    return v


NEW_REQUEST = "new_request"
SYNTH_PART = "synth_part"
SYNTH_ERROR = "synth_error"
ASR_JOB = "asr_job"
CHUNK_TS = "chunk_ts"

# ── 错误分级（gateway 据此映射 HTTP 状态码）──
KIND_INVALID = "invalid_input"  # 客户端输入问题 → 400
KIND_COMPUTE = "compute"        # 推理侧故障 → 502

# ── 结果投递超时：目标入口卡死时丢弃本条，不冻结上游（该请求由入口侧超时兜底）──
RESULT_PUT_TIMEOUT_S = 30.0

# ── 请求级多音字覆盖（g2p_overrides）──
_OVR_MAX_WORDS = 200        # 单请求词条上限
_OVR_MAX_WORD_CHARS = 32    # 单词条字符上限
_OVR_PY_RE = re.compile(r"^[a-zü]+[1-5]$")   # pypinyin TONE3（带调数字，轻声 5）
# 词组限常用汉字区段：misaki 的中文路径只对 U+4E00–U+9FFF 音素化——
# 英文/数字走英文 G2P、ext-A 生僻字被 jieba 标 'x' 后跳过音素化（出 ❓），
# 两者覆盖都无法生效，校验期直接拒绝
_OVR_WORD_RE = re.compile(r"[\u4e00-\u9fff]+")


def normalize_overrides(overrides) -> dict:
    """校验并规范化 g2p_overrides：{词: "py1 py2" | ["py1", "py2"]}。

    返回 {词: [拼音…]}（TONE3 带调）。仅校验结构/格式，不动任何全局
    状态——实际生效由 G2PEngine 在请求内临时注入 pypinyin 词典。
    音节数须与字数一致：音素序列与文本按字对齐（时间戳依赖），
    数量不符会导致整句对齐漂移。
    """
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("g2p_overrides 必须是对象 {词组: 拼音}")
    if len(overrides) > _OVR_MAX_WORDS:
        raise ValueError(f"g2p_overrides 词条过多（{len(overrides)} > {_OVR_MAX_WORDS}）")
    out = {}
    for word, val in overrides.items():
        w = (word or "").strip()
        if not (1 <= len(w) <= _OVR_MAX_WORD_CHARS) or any(c.isspace() for c in w):
            raise ValueError(
                f"g2p_overrides 词组非法: {word!r}（1-{_OVR_MAX_WORD_CHARS} 字符且不含空白）")
        if not _OVR_WORD_RE.fullmatch(w):
            raise ValueError(
                f"g2p_overrides[{w!r}] 词组须为纯常用汉字（英文/数字/生僻字无法覆盖）")
        pys = val.split() if isinstance(val, str) else list(val or [])
        pys = [str(p).strip().lower() for p in pys if str(p).strip()]
        if not pys:
            raise ValueError(f"g2p_overrides[{w!r}] 缺少拼音")
        bad = [p for p in pys if not _OVR_PY_RE.match(p)]
        if bad:
            raise ValueError(
                f"g2p_overrides[{w!r}] 拼音须为 TONE3 带调格式（如 chang2）: {bad[:3]}")
        if len(pys) != len(w):
            raise ValueError(
                f"g2p_overrides[{w!r}] 音节数 {len(pys)} 与字数 {len(w)} 不一致")
        out[w] = pys
    return out


def new_request(req_id: str, origin: int, text: str, voice: str,
                speed: float, needs_align: bool,
                g2p_overrides: dict | None = None,
                dtype: str | None = None) -> dict:
    """gateway → g2p_queue。g2p_overrides：请求级多音字覆盖（已规范化）；
    dtype：请求级计算精度（已规范化，缺省 fp32）。"""
    msg = {"req_id": req_id, "origin": int(origin), "text": text,
           "voice": voice, "speed": float(speed),
           "needs_align": bool(needs_align), "dtype": dtype or DTYPE_DEFAULT}
    if g2p_overrides:
        msg["g2p_overrides"] = g2p_overrides
    return msg


def chunk(req_id: str, idx: int, ps: str, voice: str, speed: float,
          origin: int, need: int, text: str, needs_align: bool = False,
          dtype: str = DTYPE_DEFAULT,
          q_g2p_ms: float | None = None,
          g2p_ms: float | None = None,
          sents: int | None = None) -> dict:
    """g2p → tts_queue（ZSET 认领式）。need=该请求 chunk 总数；相似长度
    凑批由认领脚本的类宽（TTS_BATCH_BUCKET_CHARS）承担，chunk 本身无桶。
    needs_align：tts worker 前向后据此直接派发该 chunk 的 ASR 作业
    （chunk 级流水——对齐与剩余合成重叠，无 aligner 进程）。
    dtype：随请求进认领类键（fp16/fp32 不混批——autocast 是整批上下文）。
    q_g2p_ms/g2p_ms：g2p 阶段耗时随行（队列等待/计算，毫秒）——g2p 整单
    一次算，同请求各 chunk 携带同值，网关聚合取 max；None 不落消息
    （旧格式字节级不变，缺字段消费端按 0 兜底）。
    sents：该 chunk 合并的句子数（g2p 行/chunk 行的 sents 字段来源）；
    None 不落消息。"""
    msg = {"req_id": req_id, "idx": int(idx), "ps": ps, "voice": voice,
           "speed": float(speed), "origin": int(origin), "need": int(need),
           "text": text, "needs_align": bool(needs_align),
           "dtype": dtype}
    if q_g2p_ms is not None:
        msg["q_g2p_ms"] = float(q_g2p_ms)
    if g2p_ms is not None:
        msg["g2p_ms"] = float(g2p_ms)
    if sents is not None:
        msg["sents"] = int(sents)
    return msg


def synth_error(req_id: str, error: str, kind: str) -> dict:
    """g2p/tts → gateway_queues[o]。终态：该请求失败（有部分块由 gateway 兜底清）。"""
    return {"type": SYNTH_ERROR, "req_id": req_id, "error": error, "kind": kind}



def synth_part(req_id: str, need: int, block: dict,
               timings: dict | None = None) -> dict:
    """tts → gateway_queues[o]。block = {"shm_name", "segs": [{idx, offset,
    samples, text, q_enq?, q_claim?}]}（按句序）；need=该请求 chunk 总数
    （首段即定，后续一致）。q_enq/q_claim 为 chunk 入队/被认领的绝对时刻（秒）——网关对同
    请求取 min(入队)/max(认领) 得到请求级 q_tts 墙钟跨度。
    timings：可选阶段耗时（毫秒）{"q_g2p","g2p","tts_done"}；q_g2p/g2p 整单
    一次算（网关取 max）；tts_done 为本批前向结束的绝对时刻（秒，批内共享）
    ——网关对同请求取 max、与 seg 的 q_claim 的 min 相减得请求级 tts 墙钟
    跨度；None 不落消息（缺字段消费端按 0 兜底）。"""
    msg = {"type": SYNTH_PART, "req_id": req_id, "need": int(need),
           "block": block}
    if timings is not None:
        msg["timings"] = timings
    return msg


def asr_job(req_id: str, idx: int, origin: int, need: int, piece: dict,
            text: str, sample_rate: int,
            max_group_sec: float = 30.0) -> dict:
    """tts → asr_queue（chunk 级对齐作业）。piece = {shm_name, offset,
    samples}（音频在 SHM 块内的一段）；origin 是 chunk_ts 的回投网关；
    need=该请求 chunk 总数（chunk_ts 收齐判据）。"""
    return {"type": ASR_JOB, "req_id": req_id, "idx": int(idx),
            "origin": int(origin), "need": int(need), "piece": piece,
            "text": text, "sample_rate": int(sample_rate),
            "max_group_sec": float(max_group_sec)}


def chunk_ts(req_id: str, idx: int, need: int, samples: int,
             timestamp: list, error=None, timings: dict | None = None) -> dict:
    """asr → gateway_queues[o]。单 chunk 的字级时间戳（[[起ms,止ms],...]）
    或 error 字符串。need/samples 随行——网关 TS 账本据此在乱序到达下
    收齐拼接（偏移 = 前序 chunk 采样数换算；ASR 与剩余 TTS 重叠执行，
    chunk_ts 可能先于音频聚合完成到达）。
    timings：可选跨度戳（秒）{"q_asr_first","q_asr_last","asr_done",
    "asr_bucket"}——q_asr_first/q_asr_last 为该 ASR job 入队/被认领的绝对
    时刻，网关据此算请求级 q_asr 墙钟跨度；asr_done 为回投时刻（≈桶推理
    完成），网关取 max、与 q_asr_last 的 min 相减得请求级 asr 墙钟跨度；
    asr_bucket 为所属推理桶的起始时刻（同桶共享一戳），网关按同请求去重
    计数 asr_fw（桶数）；None 不落消息（缺字段消费端按 0 兜底）。"""
    msg = {"type": CHUNK_TS, "req_id": req_id, "idx": int(idx),
           "need": int(need), "samples": int(samples),
           "timestamp": timestamp, "error": error}
    if timings is not None:
        msg["timings"] = timings
    return msg


# 结果投递丢弃计数（at-most-once 设计内的受控丢弃）：各 worker 进程独立
# 计数，随心跳 stats 上报，经 _ROLE_SPEC err 元组聚合为 /metrics 的
# kokoro_worker_errors_total。键预置、进程内单线程递增——无跨线程竞态。
_DROPS = {"result": 0}


def drop_count() -> int:
    """本进程累计丢弃的结果消息数（put_result 失败 + ack 放弃）。"""
    return _DROPS["result"]


def put_result(q, msg: dict, log=None) -> bool:
    """带超时的结果投递：目标入口阻塞（如事件循环卡在编码）时宁可丢这一条
    ——消息级 at-most-once，该请求由入口侧超时兜底，绝不让单个入口的
    阻塞逐级冻结 g2p/tts/asr 全部工人。（注意：worker 死亡导致的在途丢失
    是另一条路径——由 supervisor 的 requeue_worker/sweep 精确重派兜底，
    见 proc_common.TtsClaimQueue。）
    线程约束：本函数只在各角色主循环线程调用——_DROPS 的
    读-改-写在 GIL 下非原子，挪进线程池会静默丢计数（易碎不变量）。"""
    try:
        q.put(msg, timeout=RESULT_PUT_TIMEOUT_S)
        return True
    except Exception as exc:  # noqa: BLE001  Full（入口积压）或队列关闭
        _DROPS["result"] += 1
        if log is not None:
            log.warning("result delivery failed (dropped, req=%s type=%s): %s",
                        msg.get("req_id"), msg.get("type"), str(exc)[:120])
        return False

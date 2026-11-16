"""FunASR 词级时间戳：直连 speech_timestamp_prediction-v1-16k-offline。

统一队列架构下本模块运行在独立的 asr 池（compute.asr_main，不加载
KModel）；/tts 经 gateway → tts → asr_queue（chunk 级） 往返调用。
用法（funasr 1.4.x）:
    model.generate(input=(wav_bytes_or_path, text), data_type=("sound", "text"))
输出: {"text": "你 好 欢 迎...", "timestamp": [[起毫秒, 止毫秒], ...]}（标点已过滤）

模型懒加载单例（双检锁）；group_timestamp 按句分组推理使 ASR 显存
不随请求总时长增长。批推（timestamp_batch / batch_group_timestamp，
生效档由 warmup_batch 实测写入）：多作业展平一次 padded 批前向——
数值一致性由 _install_batch_safe_predictor 补丁保证（funasr 的
cnn_blstm 对 padding 不掩码，混合长度批会漂移，补丁逐条分解 predictor
段）。推理输入默认走内存 bytes（省每组一次临时 WAV 落盘+读盘往返），
加载期冒烟探测可用性，funasr 版本不识别 bytes 时自动回退临时文件模式
（FUNASR_INPUT=auto|bytes|file 可显式指定）。
"""
import io
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import proc_common

log = logging.getLogger("kokoro.funasr_ts")

_model = None
_model_lock = threading.Lock()  # 预热线程与首个 /tts 可能并发首载，双检锁保证只加载一次
_input_mode = {"mode": "bytes"}  # 推理输入通道：加载期探测/回退后固化
_model_device = {"dev": None}    # 单例实际运行设备（/health 上报用）
# 批推生效档：warmup_batch 以最坏形状实测后写入（OOM 阶梯取最高通过档）。
# 1 = 单条路径（未热身/热身全败降级，与历史行为完全一致）。
_batch_effective = {"n": 1}
# 批推安全补丁状态：needed=cnn_blstm 上采样（有 padding 缺陷），ok=包装安装成功。
# **needed 且 not ok 时批推必须禁用**（effective_batch 强制 1）：等长热身走补丁
# 的透传分支，无法暴露混合长度漂移——不能拿热身成功当补丁有效的证据。
_batch_patch = {"checked": False, "needed": False, "ok": False}

DEFAULT_MODEL_DIRS = [
    os.getenv("FUNASR_MODEL_DIR", ""),
    os.path.expanduser("~/.cache/modelscope/hub/models/iic/speech_timestamp_prediction-v1-16k-offline"),
    "/opt/funasr-models/timestamp-v1",
]


def _resolve_model_dir() -> str:
    for d in DEFAULT_MODEL_DIRS:
        if d and Path(d, "model.pt").is_file():
            return d
    raise FileNotFoundError(
        "未找到 FunASR 时间戳模型（需含 model.pt），请设置 FUNASR_MODEL_DIR 指向 "
        "speech_timestamp_prediction-v1-16k-offline 的模型目录"
    )


def _resolve_funasr_device() -> str:
    """FUNASR_DEVICE: auto(默认) | cpu | npu:N | cuda:N。

    auto 跟随 KModel 引擎实际选中的设备（NPU 版镜像默认即上卡）；
    显式指定 npu/cuda 时同样 fail-fast（上卡/加载失败直接抛，由 _load_asr
    退避重试——与 tts 池一致，无 CPU 回退）；仅显式 cpu 时走 CPU。
    """
    explicit = os.getenv("FUNASR_DEVICE", "auto").strip()  # 默认 auto（与 TUNABLE_PARAMS/README 登记一致）
    if explicit and explicit.lower() != "auto":
        return explicit
    # 轻量设备探测（不加载 KModel——asr 池不装 KModel 但 FunASR 仍上卡）。
    # **只允许 torch_npu 缺失时回落 CPU**：pipeline._resolve_device 在
    # "torch_npu 已加载但无可用卡"时抛 RuntimeError（fail-fast），裸
    # except Exception 会把它吞成 CPU，让坏卡伪装成健康服务（warm.ok →
    # 网关继续接 /tts，实际对齐在 CPU 上超时）。
    try:
        import pipeline
    except ImportError:  # 本地开发机无项目模块/依赖：CPU
        return "cpu"
    dev = pipeline._resolve_device()
    return dev if dev.startswith(("npu", "cuda")) else "cpu"


def _load_auto_model(model_dir: str, device: str):
    from funasr import AutoModel

    # dither=0：WavFrontend 默认 dither=1.0（训练期增广）——knf 路径每次提
    # 特征叠 kaldi-native-fbank 内部 RNG 噪声，fbank 逐次不同 → encoder 放大
    # → 长序列字级时间戳 ±20-60ms 抖动（validate_asr_batch E 节实测定位）。
    # 推理必须关闭；按键合并不破坏 yaml 其余 frontend 参数（lfr_m/n_mels 等）。
    return AutoModel(model=model_dir, device=device, disable_update=True,
                     log_level="ERROR", disable_pbar=True,
                     frontend_conf={"dither": 0.0})


def _generate(model, audio, text: str, sample_rate: int, mode: str = None):
    """推理入口：bytes=内存 WAV 直传（默认）；file=临时文件兜底。

    两种通道的 WAV 字节内容完全一致（同 sf.write/PCM_16），时间戳数值
    不因通道而变；bytes 省掉每组一次 mkstemp+落盘+读盘+unlink。"""
    mode = mode or _input_mode["mode"]
    if mode == "bytes":
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
        return model.generate(input=(buf.getvalue(), text),
                              data_type=("sound", "text"))
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="ts_")
    os.close(fd)
    try:
        sf.write(wav_path, audio, sample_rate, subtype="PCM_16", format="WAV")
        return model.generate(input=(wav_path, text),
                              data_type=("sound", "text"))
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass


def _smoke_generate(model, mode: str = "bytes") -> None:
    """加载后立刻做一次最小推理：把"算子不支持/设备不可用/输入通道不兼容"
    这类问题暴露在加载期（启动预热即触发），而不是首个业务请求。"""
    sr = 16000  # 时间戳模型为 16k 训练（speech_timestamp_prediction-v1-16k-offline）
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr // 2) / sr)).astype(np.float32)
    _generate(model, tone, "你好", sr, mode=mode)


def _generate_batch(model, audios, texts: list, sample_rate: int, mode: str = None):
    """批推理入口：N 组 (音频, 文本) 一次批前向，返回逐条结果。

    bytes 模式逐条经 funasr 同款 load_bytes 预解码后再传 list——
    prepare_data_iterator 的 list 分支不自动解码 bytes（auto_model.py
    仅在 input_format 非 auto 时转换；原始 WAV bytes 落到 extract_fbank
    的 .shape 上直接崩），而单元素分支无条件 load_bytes。预解码后的
    float32 np 与现行单条路径走同一解码函数，数值路径零变化。
    file 模式路径列表直传（load_audio_text_image_video 原生支持）。

    batch_size=len(items) 必须显式传：AutoModel.inference 默认 1，
    list 输入会被静默切成逐条串行（auto_model.py:941）——批推的意义
    就在这一个参数。条数不符 = 契约违约，抛错由调用方整批兜底。

    返回 [{"text", "timestamp"}, ...]，顺序与输入一一对应。"""
    mode = mode or _input_mode["mode"]
    audios = list(audios)
    if len(audios) != len(texts):
        raise ValueError(f"audios/texts 长度不一致: {len(audios)} != {len(texts)}")
    if not audios:
        return []
    if mode == "bytes":
        from funasr.utils.load_utils import load_bytes

        audio_list = []
        for audio in audios:
            buf = io.BytesIO()
            sf.write(buf, audio, sample_rate, subtype="PCM_16", format="WAV")
            audio_list.append(load_bytes(buf.getvalue()))
        res = model.generate(input=(audio_list, list(texts)),
                             data_type=("sound", "text"),
                             batch_size=len(audio_list))
    else:
        paths = []
        try:
            for audio in audios:
                fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="ts_")
                os.close(fd)
                paths.append(wav_path)   # 先登记后写入：write 失败也进 finally 清理
                sf.write(wav_path, audio, sample_rate, subtype="PCM_16",
                         format="WAV")
            res = model.generate(input=(paths, list(texts)),
                                 data_type=("sound", "text"),
                                 batch_size=len(paths))
        finally:
            for wav_path in paths:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass
    if not isinstance(res, list) or len(res) != len(audios):
        raise RuntimeError(
            f"funasr 批推结果数不符: {len(res) if isinstance(res, list) else type(res)}"
            f" != {len(audios)}")
    return [{"text": r.get("text", ""), "timestamp": r.get("timestamp", [])}
            for r in res]


def timestamp_batch(audios, texts, sample_rate: int = 24000) -> list:
    """N 组 (波形, 文本) 一次批推理字级时间戳（timestamp 的批版）。

    返回 [{"text", "timestamp"}, ...] 顺序与输入一一对应；数组长度
    必须相等（同 group_timestamp 的防错位契约）。"""
    model, _dev = get_model()
    return _generate_batch(model, audios, texts, sample_rate)


def _install_batch_safe_predictor(model) -> bool:
    """批推数值一致性补丁：CIF predictor 的 upsample cnn_blstm 对 padding
    不做长度掩码（funasr get_upsample_timestamp 的 cnn_blstm 分支忽略
    mask——cnn_attn 分支反而带了），双向 LSTM 的反向从 padding 尾部把
    状态带回有效区：批内短条的时间戳相对单条漂移（validate_asr_batch
    实测 20~80ms，且随批内其他条的长度变化 = 同请求输出不确定）。

    补丁：包一层 get_upsample_timestamp——混合长度批退化为逐条 B=1
    调原函数（切到各自有效长，无 padding → 与单条路径逐位一致）再
    stack 回批形状；等长批（无 padding）仍走原批路径。encoder 段
    （长度掩码 + conv 零填充语义与单条一致）不受影响，仍整批前向；
    predictor 段占比小，逐条损失可接受（validate_asr_batch 的 timing
    关与生产 q_asr/asr 日志观测）。

    funasr 未来版本若掩码化该 blstm，本补丁自动冗余（无危害）；属性
    形状不符（版本漂移）则不装补丁告警回退原行为（可用性优先）。
    返回是否安装。

    安全契约：cnn_blstm 类型下未装成功 = 混合长度批无保护，调用方
    （effective_batch/warmup_batch）必须把批推降级为单条——等长热身无法
    发现该缺陷。非 blstm 类型（cnn_attn 自带 mask）needed=False，批推照常。"""
    # AutoModel 是包装器：MonotonicAligner 在 .model 下，predictor 在再下一层
    core = getattr(model, "model", model)
    pred = getattr(core, "predictor", None)
    orig = getattr(pred, "get_upsample_timestamp", None)
    upsample_type = getattr(pred, "upsample_type", None)
    # pred 缺失（版本漂移）无法判定缺陷类型：按 needed 处理，宁降单条不赌数值。
    _batch_patch.update(checked=True,
                        needed=(upsample_type == "cnn_blstm" or pred is None),
                        ok=False)
    if pred is None or orig is None:
        log.warning("funasr predictor patch skipped (no get_upsample_timestamp)")
        return False
    if upsample_type != "cnn_blstm":
        return False   # 非 blstm 上采样无此缺陷（cnn_attn 分支本就带 mask）
    import torch

    ut = int(getattr(pred, "upsample_times", 3))

    def patched(hidden, mask=None, token_num=None):
        if hidden.shape[0] == 1 or mask is None:
            return orig(hidden, mask, token_num)
        t_max = int(hidden.shape[1])
        lens = mask.squeeze(1).sum(-1).long()
        if bool((lens == lens[0]).all()) and int(lens[0]) == t_max:
            return orig(hidden, mask, token_num)   # 等长满宽批：无 padding，原路径精确
        outs = []
        for i in range(hidden.shape[0]):
            li = int(lens[i].item())
            # 切到各自有效长（B=1 无 padding）→ 与单条路径逐位一致
            outs.append(orig(hidden[i:i + 1, :li, :],
                             mask[i:i + 1, :, :li],
                             None if token_num is None else token_num[i:i + 1]))
        # 右侧零填充回批宽度（尾部本就被 mask2 清零，下游逐条切片，语义不变）
        def _pad(cols, width):
            return torch.cat(
                [torch.nn.functional.pad(c, (0, width - c.shape[1])) for c in cols],
                dim=0)
        return (_pad([r[0] for r in outs], t_max),
                _pad([r[1] for r in outs], t_max),
                _pad([r[2] for r in outs], t_max * ut),
                _pad([r[3] for r in outs], t_max * ut))

    pred.get_upsample_timestamp = patched
    _batch_patch["ok"] = True
    return True


def batch_patch_status() -> dict:
    """批推安全补丁状态快照（心跳/诊断用）：未检查为 None。"""
    return dict(_batch_patch) if _batch_patch["checked"] else None


def effective_batch() -> int:
    """批推生效批宽（warmup_batch 实测写入；未热身=1，认领量钳此值）。

    cnn_blstm 且补丁未装成功时强制 1——批推数值安全不可由等长热身背书。"""
    if _batch_patch["checked"] and _batch_patch["needed"] \
            and not _batch_patch["ok"]:
        return 1
    return max(1, int(_batch_effective["n"]))


def downscale_effective(reason: str = "") -> int:
    """运行期 OOM 自愈：生效批宽减半（下限 1），后续认领与分桶即生效。

    热身已证明最坏形状可行——运行期 OOM 只能来自分配器碎片/异形形状
    序列，减半缩前沿向；只降不升（重启后热身重新标定）。返回新档。"""
    new = max(1, int(_batch_effective["n"]) // 2)
    if new != _batch_effective["n"]:
        _batch_effective["n"] = new
        log.warning("funasr effective batch downscaled to %d (%s)",
                    new, str(reason)[:120])
    return _batch_effective["n"]


def _npu_reserved_mb(dev):
    """NPU 缓存池 reserved（MB）——热身后实测峰值，预算 vs 实测对照入
    日志/心跳（回填 memcalc 校准锚点）。非 npu 设备返回 None。"""
    if not str(dev).startswith("npu"):
        return None
    try:
        import torch

        idx = int(str(dev).split(":")[-1]) if ":" in str(dev) else 0
        return round(torch.npu.memory_reserved(idx) / 2**20, 1)
    except Exception:  # noqa: BLE001  观测失败不影响服务
        return None


def _warmup_tone(sample_rate: int, sec: float) -> np.ndarray:
    """热身合成音（与 _smoke_generate 同款 440Hz，时长对齐生产最坏组）。"""
    n = int(sample_rate * sec)
    return (0.1 * np.sin(2 * np.pi * 440.0 * np.arange(n) / sample_rate)
            ).astype(np.float32)


def warmup_batch(max_batch: int, max_group_sec: float = None) -> int:
    """加载后最坏形状热身 + OOM 阶梯自降级（解析预算的实测闭环）。

    合成 max_batch 条 max_group_sec 长度 tone（与生产同形状：24k WAV
    bytes → load_bytes → 16k → T=500 encoder 帧）一次批推——① allocator
    以真峰值形状预热，池一次到位，后续任何批不涨池、npu-smi 恒定；
    ② pipeline._asr_auto_budget_mb 解析预算的实测校验：热身成功即证明
    预算 ≥ 实测峰值，reserved 峰值入日志。OOM 阶梯降半重试，取最高
    通过档写 _batch_effective；=1 时与单条路径行为完全一致。任何档位
    都不抛错（批只是优化，可用性优先——与 FUNASR_INPUT auto 回退同
    风格）。CPU 设备同样执行（顺带验证 list 输入路径在所装 funasr
    版本可用——list 分支不自动解码 bytes 的坑在此提前暴露）。

    返回生效批宽。"""
    sec = float(max_group_sec if max_group_sec is not None
                else proc_common.ASR_MAX_GROUP_SEC)
    model, dev = get_model()
    sr = proc_common.env_int("KOKORO_SAMPLE_RATE", 24000, minimum=8000)
    b = max(1, int(max_batch))
    if _batch_patch["checked"] and _batch_patch["needed"] \
            and not _batch_patch["ok"]:
        if b > 1:
            log.error("funasr batch warmup skipped: cnn_blstm predictor patch"
                      " unavailable (funasr version drift?) → effective_batch=1")
        b = 1
    while b >= 1:
        t0 = time.perf_counter()
        try:
            res = _generate_batch(model, [_warmup_tone(sr, sec)] * b,
                                  ["你好"] * b, sr)
            if len(res) != b:
                raise RuntimeError(f"warmup 结果数不符 {len(res)} != {b}")
            log.info("funasr batch warmup ok: effective_batch=%d (%.0fms,"
                     " shape=%dx%.0fs) reserved=%sMB",
                     b, (time.perf_counter() - t0) * 1000, b, sec,
                     _npu_reserved_mb(dev))
            _batch_effective["n"] = b
            return b
        except Exception as exc:  # noqa: BLE001  OOM/批路径异常：降半重试
            log.warning("funasr batch warmup b=%d failed (%s), halving",
                        b, str(exc)[:160])
            b //= 2
    return effective_batch()


def _probe_input_mode(model) -> str:
    """FUNASR_INPUT: auto(默认) | bytes | file——决定推理输入通道。

    auto：先按 bytes 冒烟，不通过（funasr 版本不识别 bytes 输入等）自动
    回退 file 模式并告警；显式 bytes 失败直接抛（配置了就要求生效）；
    显式 file 跳过探测（兼容特殊挂载环境）。"""
    pref = (os.getenv("FUNASR_INPUT", "auto").strip().lower() or "auto")
    if pref == "file":
        _smoke_generate(model, "file")
        return "file"
    try:
        _smoke_generate(model, "bytes")
        return "bytes"
    except Exception as exc:  # noqa: BLE001
        if pref == "bytes":
            raise
        log.warning("FunASR bytes 输入不可用(%s)，回退临时文件模式", str(exc)[:120])
        _smoke_generate(model, "file")
        return "file"


def get_model():
    """进程内单例（双检锁）；首次调用加载（约数秒），之后复用。

    失败不降级（与 tts 池 fail-fast 一致）：NPU 上加载/自检失败直接抛错，
    由 asr worker 的重试链与编排层（容器重启）兜底——静默回退 CPU 会让
    /health 的 device 上报失真（假装在卡上跑）。仅当推理自检通过才发布
    单例，任何失败路径都不留半成品。
    返回 (model, 实际使用设备)；设备由 compute._load_asr 用于上报。
    """
    global _model
    if _model is not None:
        return _model, _model_device["dev"]
    with _model_lock:
        if _model is not None:
            return _model, _model_device["dev"]

        model_dir = _resolve_model_dir()
        device = _resolve_funasr_device()
        log.info("loading FunASR timestamp model: %s (device=%s)...", model_dir, device)
        t0 = time.perf_counter()
        model = _load_auto_model(model_dir, device)
        _input_mode["mode"] = _probe_input_mode(model)
        _batch_safe_pred = _install_batch_safe_predictor(model)
        if _batch_patch["needed"] and not _batch_patch["ok"]:
            _batch_effective["n"] = 1
            log.error("FunASR batch inference disabled (effective=1): cnn_blstm"
                      " predictor patch unavailable; timestamps would drift on"
                      " mixed-length batches (funasr version drift?)")
        # 先写 device 再发布 model：另一线程可能在锁外看到 _model 非空后
        # 立即读 _model_device["dev"]——顺序反了会返回 (model, None)，使
        # compute 误判为 CPU、跳过 prime_asr_memory 并上报错误设备。
        _model_device["dev"] = device
        _model = model
        log.info("FunASR timestamp model ready (%.1fs, device=%s, input=%s,"
                 " blstm_patch=%s, batch_effective=%d)",
                 time.perf_counter() - t0, device, _input_mode["mode"],
                 _batch_safe_pred, effective_batch())
        return _model, device


def timestamp(audio, text: str, sample_rate: int = 24000) -> dict:
    """合成波形(float32 单声道) -> FunASR 字级毫秒时间戳。

    返回 {"text": "你 好 ...", "timestamp": [[起ms, 止ms], ...]}（与外部
    FunASR 服务的历史输出格式一致）。
    """
    model, _dev = get_model()
    res = _generate(model, audio, text, sample_rate)
    r = res[0]
    return {"text": r.get("text", ""), "timestamp": r.get("timestamp", [])}


def _quietest_cut(audio: np.ndarray, lo: int, hi: int,
                  sample_rate: int) -> int:
    """[lo, hi) 采样点范围内 100ms RMS 最低段的中点（确定性下刀点）。

    语音短语边界近静音——静音下刀使「一个字的发音被切进两片」的概率
    最小。argmin 确定性 → 单/批路径（共用 _split_groups）对照可复现。"""
    win = max(int(0.1 * sample_rate), 1)
    lo, hi = max(lo, 0), min(hi, len(audio))
    if hi - lo <= win:
        return (lo + hi) // 2   # 窗口不足 100ms：取中点（lo>beg 由调用方保证）
    n_frames = (hi - lo) // win
    frames = audio[lo:lo + n_frames * win].reshape(n_frames, win)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return lo + int(np.argmin(rms)) * win + win // 2


def _split_part(audio: np.ndarray, text: str, sample_rate: int,
                max_group_sec: float) -> list:
    """超长单 part（> max_group_sec）硬切为 [(音频片, 文本片), ...]。

    兜底层：g2p._chunk_cap 语速感知切句已把典型 chunk 压到组上限内，此处
    只防语音实际速率偏离 17.5帧/token 校准（低速长句仍超标）与直调路径。
    循环切至剩余 ≤ 上限；文本按字符比例分配（语音速率近似均匀），两侧
    各留 ≥1 字。len(text) < 2 无法分配文本时安全阀：整体不切（旧行为，
    宁超预算不错位）。片时长之和 = 原长（无缝覆盖 → 偏移累进不变）。"""
    cap = int(max_group_sec * sample_rate)
    pieces, beg, ci = [], 0, 0
    sec_per_char = len(audio) / sample_rate / max(len(text), 1)
    while len(audio) - beg > cap and len(text) - ci >= 2:
        target = beg + cap
        win_lo = max(beg + sample_rate, target - 2 * sample_rate)
        cut = _quietest_cut(audio, win_lo, target, sample_rate)
        cj = ci + int(round((cut - beg) / sample_rate / sec_per_char))
        cj = min(max(cj, ci + 1), len(text) - 1)
        pieces.append((audio[beg:cut], text[ci:cj]))
        beg, ci = cut, cj
    pieces.append((audio[beg:], text[ci:]))
    return pieces


def _split_groups(parts, texts, sample_rate: int, max_group_sec: float) -> list:
    """按累计时长 ≤ max_group_sec 切组（group_timestamp /
    batch_group_timestamp 共用的分组数学）。

    组帧数硬封顶是 asr 显存预算（pipeline._asr_auto_budget_mb）的前提，
    由两层保证：g2p._chunk_cap 语速感知切句（典型路径不超标）+ 本函数
    超长单 part 硬切（_split_part，防御校准偏差）。多 part 也前置判满：
    20s+20s（cap=30）会切成两组，不会组成 40s 组。

    返回 [(组音频片段列表, 组文本列表), ...]——不拼接，批路径需要
    片段级偏移基准（组音频采样点数 = 组内片段之和）。"""
    if len(parts) != len(texts):
        raise ValueError(f"parts/texts 长度不一致: {len(parts)} != {len(texts)}")
    cap_samples = max_group_sec * sample_rate
    groups, cur_a, cur_t, cur_samples = [], [], [], 0
    for audio, text in zip(parts, texts):
        if len(audio) > cap_samples:
            # 超长 part：清空在攒组，硬切片各自成组（顺序保持 → 偏移无缝）
            if cur_a:
                groups.append((cur_a, cur_t))
                cur_a, cur_t, cur_samples = [], [], 0
            for piece_a, piece_t in _split_part(audio, text, sample_rate,
                                                max_group_sec):
                groups.append(([piece_a], [piece_t]))
            continue
        # 前置判满：追加后不得越过组上限（后置 flush 会多装一个 part）
        if cur_a and cur_samples + len(audio) > cap_samples:
            groups.append((cur_a, cur_t))
            cur_a, cur_t, cur_samples = [], [], 0
        cur_a.append(audio)
        cur_t.append(text)
        cur_samples += len(audio)
        if cur_samples >= cap_samples:
            groups.append((cur_a, cur_t))
            cur_a, cur_t, cur_samples = [], [], 0
    if cur_a:
        groups.append((cur_a, cur_t))
    return groups


def group_timestamp(parts, texts, sample_rate: int = 24000, max_group_sec: float = 30.0) -> dict:
    """逐组推理字级时间戳并按偏移拼回全局（单作业路径：批推降级档亦走此）。

    parts/texts 一一对应（切句后的音频片段与文本，长度必须相等——不匹配
    时 zip 会静默截断导致文本错位，这里显式报错）。按累计时长 ≤
    max_group_sec 切组、每组独立推理：单次输入形状固定 → ASR 显存有界，
    不随请求总时长增长（否则整段音频一次推理，激活值随总时长线性/平方级
    增长）。组起始偏移 = 前面各组采样点数累加，边界精确对齐无漂移。
    """
    groups = _split_groups(parts, texts, sample_rate, max_group_sec)

    all_ts = []
    offset_ms = 0.0
    for audio_list, text_list in groups:
        group_audio = np.concatenate(audio_list)
        res = timestamp(group_audio, "".join(text_list), sample_rate)
        for s, e in res.get("timestamp") or []:
            all_ts.append([s + round(offset_ms), e + round(offset_ms)])
        offset_ms += len(group_audio) / sample_rate * 1000
    return {"text": "".join(texts), "timestamp": all_ts}


def batch_group_timestamp(job_items, sample_rate: int = 24000,
                          max_group_sec: float = None) -> list:
    """多作业批推理：job_items = [(parts, texts), ...] 每 asr job 一项。

    与 group_timestamp 同一组切分/偏移拼接数学，差别仅在前向：全部
    job 的组展平后一次批前向（MonotonicAligner 批原生：extract_fbank
    对 list 走 pad_sequence → SANM encoder 一次 padded 批推，见
    pipeline._asr_auto_budget_mb 的形状推导）。生效批宽 ≤1（热身降级/
    未热身）时逐 job 走 group_timestamp——降级路径与降级后同一条代码，
    热身通过的生产路径才走批。组数超过生效档时分桶串推（热身只对
    ≤生效档的形状做过峰值预热，超档形状可能涨池）。

    返回逐 job {"text", "timestamp"}，顺序与 job_items 一一对应；
    偏移按各 job 自己的组累加，跨 job 不共享。
    """
    mgs = float(max_group_sec if max_group_sec is not None
                else proc_common.ASR_MAX_GROUP_SEC)
    if effective_batch() <= 1:
        return [group_timestamp(parts, texts, sample_rate, mgs)
                for parts, texts in job_items]
    # ① 各 job 切组（同 group_timestamp 数学），展平全部组并记录回拼键
    flat_audio, flat_text, back = [], [], []   # back: (job 下标, 组起始偏移ms)
    for ji, (parts, texts) in enumerate(job_items):
        offset_ms = 0.0
        for audio_list, text_list in _split_groups(parts, texts, sample_rate, mgs):
            group_audio = np.concatenate(audio_list)
            flat_audio.append(group_audio)
            flat_text.append("".join(text_list))
            back.append((ji, round(offset_ms)))
            offset_ms += len(group_audio) / sample_rate * 1000
    # ② 按生效档分桶批推（条数不符在 _generate_batch 内契约抛错 → 整批兜底）
    results = []
    step = effective_batch()
    model, _dev = get_model()
    for beg in range(0, len(flat_audio), step):
        results.extend(_generate_batch(model, flat_audio[beg:beg + step],
                                       flat_text[beg:beg + step], sample_rate))
    # ③ 按 job 拼回（同 group_timestamp 偏移数学：round 后整型相加无漂移）
    out = [{"text": "", "timestamp": []} for _ in job_items]
    for (ji, off_ms), res in zip(back, results):
        for s, e in res.get("timestamp") or []:
            out[ji]["timestamp"].append([s + off_ms, e + off_ms])
    for (_parts, texts), o in zip(job_items, out):
        o["text"] = "".join(texts)
    return out


# ── 字符级绑定：单元时间戳 → 逐字符时间戳（段数恒 == 可对齐字符数）──
#
# 强制对齐模型按输入文本分单元返回：中文逐字一段，英文词/数字串多字共
# 用一段（真实案例："Token" 五字母共用 860ms 一段 → 251 段 vs 255 个可
# 对齐字符）。消费端（播放器 bind/index）按"每字符一段"索引，单元粒度
# 直接造成段数不齐与逐字错位——bind_char_timestamps 把多字单元的窗口
# 均分到单元内各字符，让 /tts timestamp 段数恒等于可对齐字符数。

_ALIGN_RE = re.compile(r"[\u4e00-\u9fff0-9A-Za-z]")   # 与播放器 ALIGN_RE 同口径


def _slice_window(s_ms, e_ms, n: int) -> list:
    """[s,e) 均分 n 份（整毫秒线性取整，单调不减；n<=0 返回空）。"""
    if n <= 0:
        return []
    s_ms, e_ms = int(s_ms), int(e_ms)
    if n == 1 or e_ms <= s_ms:
        return [[s_ms, max(s_ms, e_ms)]]
    step = (e_ms - s_ms) / n
    return [[int(s_ms + k * step), int(s_ms + (k + 1) * step)]
            for k in range(n)]


def _monotone(segs: list) -> list:
    """就地钳制为起点单调不减、段内有序（异常输入的最后防线）。"""
    prev = 0
    for w in segs:
        if w[0] < prev:
            w[0] = prev
        if w[1] < w[0]:
            w[1] = w[0]
        prev = w[1]
    return segs


def _positional_bind(target: list, ts: list) -> list:
    """兜底：M 个字符按位置比例铺到 N 段窗口（吞字/变体/计数错乱）。

    字符 i 的锚点 (i+0.5)/M 映射到段下标；同段多字均分该段窗口，无字
    符的段窗口闲置（其时间由相邻段的钳制吸收）。输出段数 == len(target)
    且起点单调不减。"""
    m, n = len(target), len(ts)
    assign: list[list[int]] = [[] for _ in range(n)]
    for i in range(m):
        assign[min(int((i + 0.5) / m * n), n - 1)].append(i)
    out: list = [None] * m
    last_end = 0
    for j, grp in enumerate(assign):
        if not grp:
            continue
        s, e = ts[j]
        for k, (a, b) in enumerate(_slice_window(s, e, len(grp))):
            a, b = max(a, last_end), max(b, max(a, last_end))
            out[grp[k]] = [a, b]
            last_end = b
    return [w for w in out if w is not None]


def _tokenize_units(text: str) -> list:
    """按 FunASR 分词器的实证规则把文本拆成对齐单元。

    真机探针实证（20261116，speech_timestamp_prediction-v1，
    scripts/repro/probe_text_format.py）：
        "刷Token调用行为" → 刷 / TOKEN / 调 / 用 / 行 / 为
    即：中文字逐字、ASCII 字母串整体（FunASR 回显大写，绑定比较用
    casefold 归一）、ASCII 数字串整体；标点与其他符号不产生单元。
    """
    units: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if "\u4e00" <= ch <= "\u9fff":
            units.append(ch)
            i += 1
        elif ch.isascii() and (ch.isalpha() or ch.isdigit()):
            # 同类 ASCII 连跑为一段：字母串（含英文词）或数字串
            cls_alpha = ch.isalpha()
            j = i + 1
            while j < n and text[j].isascii() and (
                    text[j].isalpha() if cls_alpha else text[j].isdigit()):
                j += 1
            units.append(text[i:j])
            i = j
        else:
            i += 1
    return units


def bind_char_timestamps(text: str, ts: list) -> list:
    """单元级时间戳 → 字符级：输出段数恒等于 text 的可对齐字符数。

    单元清单按 FunASR 分词规则从**输入文本**重建（_tokenize_units），
    不依赖 generate 返回的 text 字段——group_timestamp/batch_group_timestamp
    回填的 text 是输入原文（"".join(texts)），从来不是分词回显
    （20261116 生产 WARNING 取证结论，探针见 scripts/repro/probe_text_format.py）。

    text：chunk 原文（含标点）
    ts  ：单元级时间戳 [[起ms, 止ms], ...]（与分词单元按序一一对应）

    路径：
      ① 重建单元数 == 时间戳数（常态）→ 一一精确绑定，多字单元（英文
         词/数字串）窗口均分到各字符；
      ② 单元数 != 时间戳数（对齐吞段/异常）→ 按位置比例铺，段数契约
         不破，WARNING 留痕。
    ts 空 / 文本无可对齐字符 → []（与对齐失败回空契约一致）。
    """
    target = _ALIGN_RE.findall(text or "")
    if not ts or not target:
        return []
    units = _tokenize_units(text or "")

    if len(units) == len(ts):
        out = []
        for u, (s, e) in zip(units, ts):
            out.extend(_slice_window(s, e, len(u)))
        return _monotone(out)
    log.warning("对齐单元数 %d != 时间戳数 %d（吞段/异常），按位置比例兜底",
                len(units), len(ts))
    return _monotone(_positional_bind(target, ts))

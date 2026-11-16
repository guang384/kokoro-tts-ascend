"""Kokoro-82M-v1.1-zh 推理引擎：共享 KModel + 中文 G2P + 批量前向接口。

- 设备自动探测可用 NPU → GPU → CPU（DEVICE 可强制指定，
  ASCEND_DEVICE_ID 指容器内逻辑卡号，越界/坏卡自动回退并告警）。
- torch 相关 import 全部在函数内部，保证 uvicorn 多 worker 场景下
  每进程独立初始化。
- 批量接口 synth_batch() 一次前向合成 B 个句子（见 batched_model.py）。
"""
import importlib
import logging
import math
import os
import re
import threading
import time
from typing import List, Sequence, Tuple

import npu_ops
import proc_common as _pc
import protocol

log = logging.getLogger("kokoro.pipeline")
_model_id = _pc.env_str("KOKORO_MODEL_ID")
_default_voice = _pc.env_str("KOKORO_VOICE")

_init_lock = threading.Lock()
_g2p: "G2PEngine | None" = None
_compute: "TTSCompute | None" = None


_SAMPLE_RATE = _pc.env_int("KOKORO_SAMPLE_RATE", minimum=8000)  # 声明模型真实采样率（非可调旋钮：改它会连锁破坏 fade 窗口/预算换算/WAV 头）
if _SAMPLE_RATE != 24000:
    # batched_model/memcalc/ASR 时间戳都按 24k 编译；允许改 env 只会产出
    # 错误 WAV 头/时长/时间戳偏移。启动即拒绝，别把配置错误拖到首个请求。
    raise RuntimeError(
        f"KOKORO_SAMPLE_RATE={_SAMPLE_RATE} 不支持：Kokoro-82M-v1.1-zh 与 "
        "batched_model/memcalc 固定为 24000Hz；请删除该 env 或设为 24000")
_FRAMES_PER_SEC = _SAMPLE_RATE / (10 * 6 * 5)  # 80：istftnet 上采样 10×6 × hop5 → 每秒 80 帧


def _build_en_callable():
    """英文 G2P 弹性构建（惰性）：spacy(en_core_web_sm) -> espeak-ng -> 关闭。

    只在模型已安装时使用 spacy，绝不运行期联网下载；KOKORO_EN_G2P
    可强制 spacy/espeak/off。关闭时中英混合文本里的英文会被丢弃。
    惰性：中文流量从不触发 spacy/torch 导入——g2p worker 只做文本，
    此前每个进程白载一份 torch（spacy→thinc 无条件 import torch）。
    """
    mode = os.getenv("KOKORO_EN_G2P", "auto").strip().lower()

    special = {"Kokoro": "kˈOkəɹO", "Sol": "sˈOl"}

    # 进程级锁：spacy Language 与 espeak-ng backend 都明确非线程安全
    # （C 库全局状态），G2P_THREADS>=2 时含英文的句子并发进入会串音甚至
    # 段错误。英文音素化本身毫秒级，串行化无感；中文主路径不持锁不受影响。
    en_lock = threading.Lock()
    resolved = {"fn": None}

    def _make_spacy():
        import spacy
        from misaki import en

        if not spacy.util.is_package("en_core_web_sm"):
            log.info("spacy model not installed, English G2P falls back to espeak-ng")
            return None
        from misaki import espeak

        try:
            fallback = espeak.EspeakFallback(british=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("EspeakFallback unavailable: %s", exc)
            fallback = None
        g2p = en.G2P(fallback=fallback, unk="")

        def fn(text: str) -> str:
            _, tokens = g2p(text)
            # 与 KPipeline.tokens_to_ps 逐字节一致（不 import kokoro：
            # kokoro/__init__ 无条件 import torch/KModel）
            return "".join(t.phonemes + (" " if t.whitespace else "")
                           for t in tokens).strip() if tokens else ""

        log.info("English G2P: spacy en_core_web_sm")
        return fn

    def _make_espeak():
        from misaki import espeak

        g2p = espeak.EspeakG2P(language="en-us")

        def fn(text: str) -> str:
            ps, _ = g2p(text)
            return ps

        log.info("English G2P: espeak-ng")
        return fn

    def _resolve():
        if resolved["fn"] is not None:
            return resolved["fn"]
        fn = None
        if mode in ("auto", "spacy"):
            try:
                if npu_ops._torch_import_ok():
                    fn = _make_spacy()
                else:
                    log.warning("spacy English G2P skipped: torch 不可导入"
                                "（QEMU/坏环境），改用 espeak-ng")
            except Exception as exc:  # noqa: BLE001
                log.warning("spacy English G2P unavailable: %s", exc)
        if fn is None and mode in ("auto", "espeak"):
            try:
                fn = _make_espeak()
            except Exception as exc:  # noqa: BLE001
                log.warning("espeak English G2P unavailable: %s", exc)
        if fn is None:
            log.warning("English G2P disabled: English text in mixed input will be"
                        " dropped (KOKORO_EN_G2P=auto|spacy|espeak|off)")
            def fn(text: str) -> str:
                return ""
        resolved["fn"] = fn
        return fn

    def en_callable(text: str) -> str:
        if text in special:
            return special[text]
        with en_lock:
            return _resolve()(text)

    return en_callable


# 多音字修正表（词组 → 拼音序列，TONE3 带调）：misaki 的中文 G2P 靠
# jieba 分词 + pypinyin 单字消歧，实测"很长/长文本"等组合切出单字"长"
# 时默认取 zhǎng（生长义）——高频误判内置修正；业务方经 KOKORO_G2P_DICT
# 扩展（每行：词组 拼音1 拼音2...，空格分隔）。
_POLYPHONE_FIXES = {
    "很长": ["hen3", "chang2"], "太长": ["tai4", "chang2"],
    "不长": ["bu4", "chang2"], "较长": ["jiao4", "chang2"],
    "长文本": ["chang2", "wen2", "ben3"],
    "长时间": ["chang2", "shi2", "jian1"],
    "悠长": ["you1", "chang2"], "长距离": ["chang2", "ju4", "li2"],
    "几长": ["ji3", "chang2"], "多长": ["duo1", "chang2"],
}


def _load_polyphone_fixes() -> dict:
    """内置表 + KOKORO_G2P_DICT（文件路径，或内联内容）。

    文件/内联格式：每行（或分号分隔）"词组 拼音1 拼音2 …"，拼音用
    pypinyin TONE3（带调数字：chang2/hang2/yue4；轻声 5）。
    """
    fixes = dict(_POLYPHONE_FIXES)
    raw = os.getenv("KOKORO_G2P_DICT", "").strip()
    if not raw:
        return fixes
    lines = []
    try:
        if os.path.isfile(raw):          # 文件模式（K8s ConfigMap 挂载）
            with open(raw, encoding="utf-8") as f:
                lines = f.readlines()
        else:                            # 内联模式（env 直接写内容）
            lines = raw.replace(";", chr(10)).splitlines()
    except OSError as exc:
        log.warning("G2P_DICT %s 读取失败（忽略）：%s", raw, exc)
        return fixes
    for ln, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) >= 2:
            fixes[parts[0]] = parts[1:]
        elif line.strip():
            log.warning("G2P_DICT 第 %d 行格式应为'词组 拼音…'，跳过: %r",
                        ln, line.strip()[:40])
    log.info("G2P 多音字词典已加载：%d 条（内置 %d + 自定义 %d）",
             len(fixes), len(_POLYPHONE_FIXES), len(fixes) - len(_POLYPHONE_FIXES))
    return fixes


def _apply_polyphone_fixes(fixes: dict) -> None:
    """双词典联动生效：jieba.add_word 保证词组被切成整词（pypinyin 的
    phrases 词典只对整词查询生效——"太长"默认被切成 ['太','长']，逐字
    转换时词组词条永远查不到）；pypinyin.load_phrases_dict 提供读音。"""
    try:
        import jieba
        from pypinyin import load_phrases_dict

        for word, pys in fixes.items():
            jieba.add_word(word, freq=1000)
        load_phrases_dict({w: [[p] for p in pys]
                           for w, pys in fixes.items()})
        if fixes:
            log.info("多音字修正已生效：%d 条", len(fixes))
    except Exception as exc:  # noqa: BLE001  词典失败不阻断 G2P（原读音兜底）
        log.warning("多音字修正加载失败（忽略，原读音兜底）：%s", exc)


# 请求级多音字覆盖（g2p_overrides）：pypinyin 的读音词典是进程级全局态，
# 覆盖只能「锁内临时注入 → 音素化 → 精确还原」。锁必须覆盖**所有**音素化
# 调用（无覆盖的请求同样持锁），否则注入窗口内其它请求会读到临时读音。
_G2P_LOCK = threading.Lock()
_OVR_MISSING = object()


def _apply_request_overrides(overrides: dict):
    """锁内注入（调用方持 _G2P_LOCK）：jieba 增词保证整词切分 +
    PHRASES_DICT 提供读音 + 临时替换 pypinyin 分词器。

    为什么不用 mmseg.retrain：retrain 会对整个短语词典重建分词 trie，
    在 g2p 进程（misaki 已 load 大词典）实测 ~100ms/次、每请求要两次
    （注入+还原）。而 misaki 是「jieba 切词 → 逐词 lazy_pinyin」，只要
    分词器对覆盖词返回整词，pypinyin 就会命中 PHRASES_DICT 的临时词条
    ——包装器仅对覆盖词走捷径（实测 0.3ms），其余原样交给原分词器。

    覆盖语义：最长匹配、显式覆盖优先。覆盖词嵌在更长 token 内（tone
    sandhi 合并词、或内置长词条如"朝阳区"含"朝阳"）时按覆盖词切开，
    因此用户覆盖能盖过内置短语词典——比 retrain（长词条优先）更符合
    “本次请求就要这个读音”的预期。
    返回 (saved, orig_seg) 还原句柄。"""
    import jieba
    import pypinyin.core as pinyin_core

    saved = {}
    orig_seg = pinyin_core.seg
    try:
        # 最长优先的覆盖词模式：处理"覆盖词嵌在更长 token 内"的形态
        # （tone sandhi 会把相邻词合并成更长的 word 再查读音）
        pat = re.compile("|".join(
            re.escape(w) for w in sorted(overrides, key=len, reverse=True)))

        def _seg_with_overrides(hans):
            if not isinstance(hans, str):
                return orig_seg(hans)
            if hans in overrides:
                return [hans]    # 整词 → _phrase_pinyin 命中临时读音
            if pat.search(hans) is None:
                return orig_seg(hans)   # 常规路径：零额外分词成本
            parts, pos = [], 0
            for m in pat.finditer(hans):
                if m.start() > pos:
                    parts.extend(orig_seg(hans[pos:m.start()]))
                parts.append(m.group(0))   # 命中段整词（临时读音）
                pos = m.end()
            if pos < len(hans):
                parts.extend(orig_seg(hans[pos:]))
            return parts

        for word, pys in overrides.items():
            saved[word] = (pinyin_core.PHRASES_DICT.get(word, _OVR_MISSING),
                           jieba.dt.FREQ.get(word, 0))
            pinyin_core.PHRASES_DICT[word] = [[p] for p in pys]
            jieba.add_word(word, freq=10000)
        pinyin_core.seg = _seg_with_overrides
        return saved, orig_seg
    except Exception:
        # 注入中途失败（异常/词典坏态）：已改的部分必须回滚——否则临时读音
        # 会永久留在全局词典（后续请求读错）且 jieba 词频被抬高
        _restore_request_overrides(saved, orig_seg)
        raise


def _restore_request_overrides(saved: dict, orig_seg) -> None:
    """锁内还原（finally 必达）：分词器/词典值逐词恢复，jieba 按注入前
    词频还原（原本不存在 del_word，原有词恢复原频——不回退会让后续请求
    的切分边界悄悄变化）。"""
    import jieba
    import pypinyin.core as pinyin_core

    pinyin_core.seg = orig_seg
    for word, (old, old_freq) in saved.items():
        if old is _OVR_MISSING:
            pinyin_core.PHRASES_DICT.pop(word, None)
        else:
            pinyin_core.PHRASES_DICT[word] = old
        try:
            if old_freq > 0:
                jieba.add_word(word, freq=old_freq)
            else:
                jieba.del_word(word)
        except Exception:  # noqa: BLE001  单词回滚失败不阻断其余词
            # 词频错位只影响切分边界（读音仍由 PHRASES_DICT 决定），
            # 但必须留痕——这是"字典脏了"的唯一线索
            log.warning("g2p override rollback failed for %r (jieba freq)", word)


class G2PEngine:
    """入口侧文本引擎：中文 G2P（misaki/jieba，纯 CPU，无模型权重）。"""

    def __init__(self):
        from misaki import zh

        log.info("initializing Chinese G2P pipeline...")
        # 直接构造 zh.ZHG2P，不 import kokoro：kokoro/__init__ 无条件
        # import torch/KModel（g2p worker 不做前向，白载 330MB 权重；
        # QEMU/无 NN 库环境还会在构造期段错误）。构造参数与
        # KPipeline 的 lang_code=='z' 分支逐字一致。
        self.g2p = zh.ZHG2P(
            version=None if _model_id.endswith("/Kokoro-82M") else "1.1",
            en_callable=_build_en_callable())
        # 预热 jieba 词典，避免首个请求冷启动 & G2P 线程池竞态
        try:
            import jieba

            jieba.initialize()
        except Exception:
            pass
        _apply_polyphone_fixes(_load_polyphone_fixes())
        log.info("G2P ready")

    def phonemize(self, text: str, overrides: dict | None = None) -> List[str]:
        """句子 -> 音素串列表（已按 510 截断）。输入按 \\n 分句。

        overrides：请求级多音字覆盖 {词: [拼音…]}（TONE3），仅本次调用
        生效——不写全局词典（并发请求会互相污染）。"""
        out = []
        for ps_list in self.phonemize_many(text.split("\n"), overrides):
            out.extend(ps_list)
        return out

    def phonemize_many(self, sentences, overrides: dict | None = None) -> List[List[str]]:
        """多句共享同一组 overrides 的音素化（一次注入/还原）。

        全局串行（_G2P_LOCK）：override 生效依赖 pypinyin 进程级
        PHRASES_DICT 临时改写，锁外并发会让其它请求读到注入读音。
        锁内仅做 jieba+misaki 音素化（毫秒级，G2P 需求远低于单核容量）。"""
        ovr = protocol.normalize_overrides(overrides)
        with _G2P_LOCK:
            applied = _apply_request_overrides(ovr) if ovr else None
            try:
                return [self._phonemize_one(s) for s in sentences]
            finally:
                if applied is not None:
                    _restore_request_overrides(*applied)

    def _phonemize_one(self, sent: str) -> List[str]:
        sent = sent.strip()
        if not sent:
            return []
        ps, _ = self.g2p(sent)
        ps = ps.strip() if ps else ""
        if not ps:
            return []
        if len(ps) > 510:
            log.warning("phonemes too long, truncated: %d > 510", len(ps))
            ps = ps[:510]
        return [ps]


class TTSCompute:
    """计算侧引擎：KModel（NPU/GPU/CPU）+ 音色包缓存 + 批量前向 + 显存预算。"""

    def __init__(self, device: str):
        from kokoro import KModel

        npu_ops._prime_memory(device)  # 预算硬限+预占先行（npu_ops）
        disable_complex = npu_ops._resolve_disable_complex(device)
        self.device = device
        log.info("initializing KModel: %s (disable_complex=%s)...", _model_id, disable_complex)
        self.model = KModel(repo_id=_model_id, disable_complex=disable_complex).to(device).eval()
        # 计算精度是请求级字段（protocol.normalize_dtype，chunk 携带，凑批
        # 类键隔离）——进程级 KOKORO_DTYPE 开关已移除。fp16 走 autocast：
        # 权重保 fp32（kokoro 上游有无 dtype 的 torch.zeros 与显式 fp32 的
        # hann_window，全模型 .half() 会被截回/混算炸），前向按算子自动选
        # 型，敏感算子（归一化/softmax）自动保 fp32；算力收益来自
        # matmul/conv（占比 95%+）。fp64 参考锚不再保留（请求级无法安全
        # 切 default dtype——加载期/前向期两段全局态，A/B 用 fp16 即可）。
        self._voice_cache = {}  # voice -> [N,256] 音色矩阵（已上卡），见 _voice_matrix
        log.info("compute engine ready: device=%s dtype=per-request(fp32|fp16)",
                 device)

    def _voice_matrix(self, voice: str):
        """音色包整体上卡并缓存：[N,256]（fp32 约 520KB/音色）。

        整包上卡后 make_ref 退化为纯卡上索引，消除每句一次的 H2D 同步点。
        并发首建的竞态只是幂等重建，无锁。
        dtype 跟随模型参数精度：当前恒 fp32。fp16 是前向 autocast（权重
        不动），音色也保 fp32 由 autocast 统一选型——不做全模型 .half()
        是因为上游会显式构造 fp32 张量（hann_window 等），与 fp16 权重
        混算会炸。
        """
        cache = self._voice_cache.get(voice)
        if cache is None:
            import torch

            pack = self.load_voice_file(voice)
            cache = (torch.stack(list(pack)).squeeze(1)   # [N,1,256] -> [N,256]
                     .to(device=self.device,
                         dtype=next(self.model.parameters()).dtype))
            self._voice_cache.setdefault(voice, cache)
            cache = self._voice_cache[voice]
        return cache

    def load_voice_file(self, voice: str):
        """直接读音色 .pt（免 KPipeline 实例化）。"""
        import torch  # noqa: F401  供 map_location 解析

        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(_model_id, local_files_only=True)
        path = os.path.join(snapshot, "voices", f"{voice}.pt")
        return torch.load(path, map_location=self.device, weights_only=True)

    def make_ref(self, ps: str, voice: str):
        """音色风格向量 voice_pack[len(ps)-1]，形状 [1, 256]，已在推理设备上。"""
        return self._voice_matrix(voice)[len(ps) - 1].unsqueeze(0)

    def synth_batch_smoke(self) -> int:
        """冒烟前向：单句最短序列，验证计算路径端到端（加载即验证）。"""
        return len(self.synth_batch([("ə", _default_voice, 1.0)]))

    def synth_batch(self, jobs: Sequence[Tuple[str, str, float]],
                    dtype: str = protocol.DTYPE_DEFAULT) -> List:
        """一次前向合成多个句子。

        jobs: [(phonemes, voice, speed), ...]，调用方按音素长度分桶凑批
        （认领类键已含 dtype，批内同精度）。dtype="fp16" 时前向套
        autocast（权重保 fp32，浮点算子按 NPU 支持度自动选型）。
        返回与 jobs 等长的 int16 PCM16 波形列表（24kHz，fade/量化在卡上
        完成，见 batched_model.finalize_audio）。
        """
        import torch

        from batched_model import forward_batch, phonemes_to_ids

        if not jobs:
            return []
        ids_list = [phonemes_to_ids(self.model, ps) for ps, _, _ in jobs]
        B = len(jobs)
        L_max = max(len(ids) for ids in ids_list)
        input_ids = torch.full((B, L_max), 0, dtype=torch.long)
        lengths = torch.empty(B, dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            lengths[i] = len(ids)
        ref_s = torch.cat([self.make_ref(ps, voice) for ps, voice, _ in jobs], dim=0)  # [B,256]
        speeds = torch.tensor([sp for _, _, sp in jobs], dtype=torch.float32)
        args = (self.model, input_ids.to(self.device),
                lengths, ref_s.to(self.device), speeds.to(self.device))
        if dtype == "fp16":
            # 请求级 autocast：权重不动（fp32 常驻），浮点算子按设备支持度
            # 自动选型（敏感算子保 fp32）。device_type 取实际设备（npu/cpu）
            # ——CPU 上 autocast 同样生效（fp16 算子覆盖不全，仅供本地验证），
            # 生产 NPU 由 CANN 选型
            dev_type = str(self.device).split(":")[0]
            with torch.autocast(device_type=dev_type, dtype=torch.float16):
                return forward_batch(*args)
        return forward_batch(*args)


def list_voices() -> List[str]:
    """从本地 HF 缓存列出全部音色（镜像已内置，离线可用）。"""
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(_model_id, local_files_only=True)
    voices_dir = os.path.join(snapshot, "voices")
    if not os.path.isdir(voices_dir):
        return []
    return sorted(f[:-3] for f in os.listdir(voices_dir) if f.endswith(".pt"))


def get_g2p() -> G2PEngine:
    """入口进程单例：文本引擎（G2P/音色清单）。"""
    global _g2p
    if _g2p is None:
        with _init_lock:
            if _g2p is None:
                _g2p = G2PEngine()
    return _g2p


def get_compute() -> TTSCompute:
    """计算进程单例；首次调用触发初始化（模型已在镜像内，离线可用）。"""
    global _compute
    if _compute is None:
        with _init_lock:
            if _compute is None:
                _compute = TTSCompute(npu_ops._resolve_device())
    return _compute

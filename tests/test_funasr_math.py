"""funasr_ts.group_timestamp 分组与偏移数学（timestamp 推理打桩，无模型）。"""
import numpy as np
import pytest

import funasr_ts


@pytest.fixture()
def fake_infer(monkeypatch):
    """打桩 timestamp：恒定返回两组固定时间戳，并记录每次调用的输入。"""
    calls = []

    def _ts(audio, text, sample_rate=24000):
        calls.append((len(audio), text))
        return {"text": text, "timestamp": [[0, 100], [200, 400]]}

    monkeypatch.setattr(funasr_ts, "timestamp", _ts)
    return calls


def test_group_offsets_accumulate(fake_infer):
    """两段各 0.6s（超过 max_group_sec=0.5 各自成组）：组间偏移 = 累计采样时长。"""
    a1 = np.zeros(14400, dtype=np.float32)  # 0.6s @24k
    a2 = np.zeros(14400, dtype=np.float32)
    out = funasr_ts.group_timestamp([a1, a2], ["甲", "乙"], 24000,
                                    max_group_sec=0.5)
    assert len(fake_infer) == 2
    ts = out["timestamp"]
    assert ts == [[0, 100], [200, 400],           # 组0：零偏移
                  [600, 700], [800, 1000]]        # 组1：+600ms
    assert out["text"] == "甲乙"


def test_single_group_no_shift(fake_infer):
    a = np.zeros(4800, dtype=np.float32)  # 0.2s，单组
    out = funasr_ts.group_timestamp([a], ["甲"], 24000, max_group_sec=30)
    assert len(fake_infer) == 1
    assert out["timestamp"] == [[0, 100], [200, 400]]


# ── 设备解析：坏卡必须 fail-fast，不能被吞成 CPU ──

def test_resolve_funasr_device_propagates_no_npu(monkeypatch):
    """torch_npu 已加载但无卡：npu_ops._resolve_device 的 RuntimeError
    必须上抛（ASR 池走退避重试/编排重启），不能静默回落 CPU。"""
    import npu_ops

    monkeypatch.setenv("FUNASR_DEVICE", "auto")
    monkeypatch.setattr(
        npu_ops, "_resolve_device",
        lambda: (_ for _ in ()).throw(RuntimeError("no usable npu")))
    with pytest.raises(RuntimeError, match="no usable npu"):
        funasr_ts._resolve_funasr_device()


def test_resolve_funasr_device_auto_prefers_npu(monkeypatch):
    """auto 且有可用 NPU：跟随 pipeline 上卡（不默认 CPU）。"""
    import npu_ops

    monkeypatch.setenv("FUNASR_DEVICE", "auto")
    monkeypatch.setattr(npu_ops, "_resolve_device", lambda: "npu:0")
    assert funasr_ts._resolve_funasr_device() == "npu:0"


def test_resolve_funasr_device_cpu_when_torch_npu_absent(monkeypatch):
    """无 torch_npu 的开发机：pipeline 返回 cpu，ASR 跟随 cpu。"""
    import npu_ops

    monkeypatch.setenv("FUNASR_DEVICE", "auto")
    monkeypatch.setattr(npu_ops, "_resolve_device", lambda: "cpu")
    assert funasr_ts._resolve_funasr_device() == "cpu"


def test_resolve_funasr_device_explicit_cpu(monkeypatch):
    """显式 FUNASR_DEVICE=cpu 是唯一人工逃生口，不再探测设备。"""
    monkeypatch.setenv("FUNASR_DEVICE", "cpu")
    assert funasr_ts._resolve_funasr_device() == "cpu"


def test_group_joins_texts_in_order(fake_infer):
    """组内多段：texts 按段序拼接后作为整组文本送推理。"""
    a = [np.zeros(1200, dtype=np.float32) for _ in range(4)]
    funasr_ts.group_timestamp(a, ["你", "好", "世", "界"], 24000, max_group_sec=30)
    assert len(fake_infer) == 1
    assert fake_infer[0][1] == "你好世界"


def test_mismatched_parts_texts_raises(fake_infer):
    """parts/texts 不匹配 → 显式报错（zip 静默截断会让文本错位）。"""
    with pytest.raises(ValueError):
        funasr_ts.group_timestamp([np.zeros(8, dtype=np.float32)],
                                  ["你", "好"], 24000)


# ── 推理输入通道：bytes 内存直传（默认）/ file 临时文件兜底 ──

class _StubModel:
    """记录 generate 收到的 input 元组；可配置对 bytes 输入抛错。"""

    def __init__(self, reject_bytes=False):
        self.reject_bytes = reject_bytes
        self.inputs = []

    def generate(self, input, data_type):  # noqa: A002  对齐 funasr 关键字名
        wav, _text = input
        if self.reject_bytes and isinstance(wav, (bytes, bytearray)):
            raise TypeError("bytes input not supported")
        self.inputs.append(wav)
        return [{"text": "你 好", "timestamp": [[0, 120]]}]


@pytest.fixture(autouse=True)
def _reset_mode(monkeypatch):
    monkeypatch.setattr(funasr_ts, "_input_mode", {"mode": "bytes"})


def test_generate_bytes_channel_sends_wav_bytes():
    """bytes 通道：input[0] 是完整 WAV 字节（可被 soundfile 读回原波形）。"""
    import soundfile as sf

    audio = np.linspace(-0.5, 0.5, 4800, dtype="float32")
    m = _StubModel()
    out = funasr_ts._generate(m, audio, "你好", 24000, mode="bytes")
    assert out[0]["timestamp"] == [[0, 120]]
    back, sr = sf.read(__import__("io").BytesIO(m.inputs[0]), dtype="float32")
    assert sr == 24000 and len(back) == 4800
    assert np.allclose(back, audio, atol=1e-4)   # PCM_16 量化误差内一致


def test_generate_file_channel_uses_and_cleans_tempfile():
    """file 通道：传路径字符串，用后临时文件即删。"""
    import glob
    import tempfile

    before = set(glob.glob(tempfile.gettempdir() + "/ts_*.wav"))
    m = _StubModel()
    funasr_ts._generate(m, np.zeros(8, "float32"), "你好", 16000, mode="file")
    assert isinstance(m.inputs[0], str) and m.inputs[0].endswith(".wav")
    after = set(glob.glob(tempfile.gettempdir() + "/ts_*.wav"))
    assert after == before   # 无新增残留


def test_probe_input_mode_auto_falls_back_on_bytes_reject():
    """auto：bytes 不可用 → 自动回退 file 并通过 file 冒烟。"""
    m = _StubModel(reject_bytes=True)
    mode = funasr_ts._probe_input_mode(m)   # FUNASR_INPUT 默认 auto
    assert mode == "file"
    assert len(m.inputs) == 1   # bytes 冒烟在 stub 处抛错未入账；file 冒烟成功一次


def test_probe_input_mode_explicit_bytes_raises_on_reject(monkeypatch):
    monkeypatch.setenv("FUNASR_INPUT", "bytes")
    m = _StubModel(reject_bytes=True)
    with pytest.raises(TypeError):
        funasr_ts._probe_input_mode(m)


def test_probe_input_mode_explicit_file_skips_bytes(monkeypatch):
    monkeypatch.setenv("FUNASR_INPUT", "file")
    m = _StubModel(reject_bytes=True)   # bytes 会炸也不该被碰到
    assert funasr_ts._probe_input_mode(m) == "file"
    assert len(m.inputs) == 1 and isinstance(m.inputs[0], str)


# ── 批推理：_generate_batch 契约 + batch_group_timestamp 分组/展平/偏移数学 ──

def test_load_auto_model_passes_dither_off(monkeypatch):
    """行为锁定：AutoModel 必须收到 frontend_conf={"dither": 0.0}——
    WavFrontend 默认 dither=1.0（训练增广）走 knf 路径时每次提特征叠
    内部 RNG 噪声，fbank 逐次不同 → 长序列时间戳 ±20-60ms 抖动。"""
    pytest.importorskip("funasr")
    import funasr

    seen = {}

    class _FakeAutoModel:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(funasr, "AutoModel", _FakeAutoModel)
    funasr_ts._load_auto_model("dummy-model-dir", "cpu")
    assert seen.get("frontend_conf") == {"dither": 0.0}
    assert seen.get("model") == "dummy-model-dir"


class _BatchStubModel:
    """记录 generate 调用形态的批替身：结果按输入文本派生（可区分顺序错位）。"""

    def __init__(self):
        self.calls = []

    def generate(self, input, data_type, batch_size=1):  # noqa: A002  对齐 funasr 关键字名
        audios, texts = input
        self.calls.append({"n_audio": len(audios), "n_text": len(texts),
                           "texts": list(texts), "batch_size": batch_size,
                           "audio_types": [type(a).__name__ for a in audios],
                           "audio_lens": [len(a) for a in audios]})
        return [{"text": t, "timestamp": [[100, 150]]}   # 恒定值：偏移成唯一区分项
                for t in texts]


@pytest.fixture()
def _batch_mode(monkeypatch):
    """热身生效档钳为 4 + model 单例替身（不加载真模型）。"""
    pytest.importorskip("funasr")   # bytes 预解码走 funasr.load_bytes
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})
    stub = _BatchStubModel()
    monkeypatch.setattr(funasr_ts, "_model", stub)
    monkeypatch.setattr(funasr_ts, "_model_device", {"dev": "cpu"})
    return stub


def test_generate_batch_decodes_bytes_before_list(monkeypatch):
    """bytes 批输入必须预解码：list 分支不自动 load_bytes（auto_model.py
    只在 input_format 非 auto 时转换），原始 WAV bytes 会带着文件头进
    extract_fbank 崩掉——进 generate 的 audio 列表必须是 float32 np，
    且 24k 输入已被解码重采样为 16k（与单条路径同一解码函数）。"""
    pytest.importorskip("funasr")
    m = _BatchStubModel()
    out = funasr_ts._generate_batch(
        m, [np.linspace(-0.5, 0.5, 4800, dtype="float32"),
            np.zeros(2400, "float32")], ["甲", "乙"], 24000, mode="bytes")
    assert len(out) == 2
    call = m.calls[0]
    assert call["batch_size"] == 2 and call["n_audio"] == call["n_text"] == 2
    assert call["audio_types"] == ["ndarray", "ndarray"]   # 预解码发生（非 bytes）
    assert call["audio_lens"] == [3200, 1600]              # 24k→16k 重采样（4800/2400 ×2/3）


def test_generate_batch_count_contract():
    """结果条数 ≠ 输入条数 = 契约违约：抛错让调用方整批兜底（不静默错位）。"""
    pytest.importorskip("funasr")

    class _ShortModel:
        def generate(self, input, data_type, batch_size=1):  # noqa: A002
            return [{"text": "t", "timestamp": []}]   # 输入 2 条只回 1 条

    with pytest.raises(RuntimeError):
        funasr_ts._generate_batch(_ShortModel(),
                                  [np.zeros(8, "float32"), np.zeros(8, "float32")],
                                  ["甲", "乙"], 24000, mode="bytes")


def test_batch_group_single_call_and_order(_batch_mode):
    """3 个单组 job：一次 generate、batch_size=3、顺序一一对应。"""
    items = [( [np.zeros(4800, "float32")], [t] ) for t in ("甲", "乙", "丙")]
    out = funasr_ts.batch_group_timestamp(items, 24000, max_group_sec=30)
    assert len(_batch_mode.calls) == 1
    assert _batch_mode.calls[0]["batch_size"] == 3
    assert _batch_mode.calls[0]["texts"] == ["甲", "乙", "丙"]
    assert [o["text"] for o in out] == ["甲", "乙", "丙"]
    assert all(o["timestamp"] == [[100, 150]] for o in out)   # 单组零偏移


def test_batch_group_multi_group_offsets(_batch_mode):
    """跨 job 展平 + 组间偏移按各自 job 累加：job0 两组（0.6s×2，
    max_group_sec=0.5 → 组1 偏移 +600ms），job1 单组零偏移。"""
    a = np.zeros(14400, dtype=np.float32)  # 0.6s @24k
    items = [([a, a], ["甲", "乙"]), ([np.zeros(4800, "float32")], ["丙"])]
    out = funasr_ts.batch_group_timestamp(items, 24000, max_group_sec=0.5)
    assert len(_batch_mode.calls) == 1                  # 3 组一批
    assert _batch_mode.calls[0]["texts"] == ["甲", "乙", "丙"]
    # 组结果恒 [[100,150]]：job0 组0 零偏移、组1 +600ms；job1 零偏移
    assert out[0]["timestamp"] == [[100, 150], [700, 750]]
    assert out[1]["timestamp"] == [[100, 150]]
    assert out[0]["text"] == "甲乙" and out[1]["text"] == "丙"


def test_batch_group_buckets_over_effective(monkeypatch):
    """组数超生效档时分桶串推（热身只预热 ≤ 档的形状，超档可能涨池）。"""
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 2})
    stub = _BatchStubModel()
    monkeypatch.setattr(funasr_ts, "_model", stub)
    monkeypatch.setattr(funasr_ts, "_model_device", {"dev": "cpu"})
    items = [([np.zeros(4800, "float32")], [t]) for t in ("甲", "乙", "丙")]
    out = funasr_ts.batch_group_timestamp(items, 24000, max_group_sec=30)
    assert [c["batch_size"] for c in stub.calls] == [2, 1]   # 档=2：3 组分两桶
    assert [o["text"] for o in out] == ["甲", "乙", "丙"]


def test_batch_group_degrade_to_per_job(monkeypatch):
    """生效档 1（未热身/热身全败）：逐 job 走 group_timestamp——降级
    路径与降级后同一条代码，不触碰 generate 批形态。"""
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 1})
    seen = []
    monkeypatch.setattr(funasr_ts, "group_timestamp",
                        lambda parts, texts, sr, mgs: (seen.append(texts),
                                                       {"text": "".join(texts),
                                                        "timestamp": [[1, 2]]})[1])
    stub = _BatchStubModel()
    monkeypatch.setattr(funasr_ts, "_model", stub)
    items = [([np.zeros(8, "float32")], ["甲"]), ([np.zeros(8, "float32")], ["乙"])]
    out = funasr_ts.batch_group_timestamp(items, 24000, max_group_sec=30)
    assert seen == [["甲"], ["乙"]]          # 逐 job 单条路径
    assert stub.calls == []                  # 批前向未被触碰
    assert [o["timestamp"] for o in out] == [[[1, 2]], [[1, 2]]]


def test_batch_group_mismatched_parts_raises(_batch_mode):
    """job 内 parts/texts 不匹配 → 显式报错（防 zip 静默截断文本错位）。"""
    with pytest.raises(ValueError):
        funasr_ts.batch_group_timestamp(
            [([np.zeros(8, "float32")], ["你", "好"])], 24000, max_group_sec=30)


def test_warmup_batch_ladder_halving(monkeypatch):
    """OOM 阶梯：B_max 失败 → 降半重试 → 最高通过档为生效批宽；全败 → 1。"""
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 1})

    class _Stub:
        device = "cpu"

    monkeypatch.setattr(funasr_ts, "get_model", lambda: (_Stub(), "cpu"))
    attempts = []
    fail_above = {"b": 2}

    def _fake_gen(model, audios, texts, sr, mode=None):
        b = len(audios)
        attempts.append(b)
        if b > fail_above["b"]:
            raise MemoryError("NPU OOM")
        return [{"text": t, "timestamp": []} for t in texts]

    monkeypatch.setattr(funasr_ts, "_generate_batch", _fake_gen)
    assert funasr_ts.warmup_batch(8) == 2        # 8→4→2（2 通过）
    assert attempts == [8, 4, 2]
    assert funasr_ts.effective_batch() == 2

    attempts.clear()
    fail_above["b"] = 0                          # 全败：4→2→1 全部 OOM
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 1})
    assert funasr_ts.warmup_batch(4) == 1
    assert attempts == [4, 2, 1]
    assert funasr_ts.effective_batch() == 1


def test_warmup_disabled_when_patch_unavailable(monkeypatch):
    """cnn_blstm 补丁装不上 → 等长热身不得把生效档写回 >1（热身走透传
    分支，证明不了混合长度安全）：warmup 只跑 B=1。"""
    monkeypatch.setattr(funasr_ts, "_batch_patch",
                        {"checked": True, "needed": True, "ok": False})
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})

    class _Stub:
        device = "cpu"

    monkeypatch.setattr(funasr_ts, "get_model", lambda: (_Stub(), "cpu"))
    attempts = []

    def _fake_gen(model, audios, texts, sr, mode=None):
        attempts.append(len(audios))
        return [{"text": t, "timestamp": []} for t in texts]

    monkeypatch.setattr(funasr_ts, "_generate_batch", _fake_gen)
    assert funasr_ts.warmup_batch(8) == 1
    assert attempts == [1]                       # 未尝试 8/4/2
    assert funasr_ts.effective_batch() == 1


def test_effective_batch_forced_1_when_patch_needed_missing(monkeypatch):
    """有效档的 fail-safe：needed 且未安装成功 → 无论 _batch_effective
    写成多少都返回 1；非 blstm（needed=False）不受影响。"""
    monkeypatch.setattr(funasr_ts, "_batch_patch",
                        {"checked": True, "needed": True, "ok": False})
    monkeypatch.setattr(funasr_ts, "_batch_effective", {"n": 4})
    assert funasr_ts.effective_batch() == 1
    monkeypatch.setattr(funasr_ts, "_batch_patch",
                        {"checked": True, "needed": False, "ok": False})
    assert funasr_ts.effective_batch() == 4


# ── 批推数值一致性补丁：_install_batch_safe_predictor 的路由逻辑 ──

def test_batch_safe_predictor_patch_routing():
    """混合有效长批 → predictor 段逐条 B=1（无 padding = 单条逐位）；
    等长满宽批 → 原批路径透传；B=1 → 直通。回批宽度按 upsample_times。"""
    pytest.importorskip("torch")
    import torch

    calls = []

    class _Pred:
        upsample_type = "cnn_blstm"
        upsample_times = 3

        def get_upsample_timestamp(self, hidden, mask=None, token_num=None):
            calls.append(int(mask.squeeze(1).sum(-1)[0]))
            w = hidden.shape[1]
            return (torch.zeros(1, w), torch.zeros(1, w),
                    torch.zeros(1, w * 3), torch.zeros(1, w * 3))

    class _Core:
        def __init__(self):
            self.predictor = _Pred()

    class _Model:
        def __init__(self):
            self.model = _Core()   # AutoModel 包装层：predictor 在 .model 下

    m = _Model()
    assert funasr_ts._install_batch_safe_predictor(m)
    pred = m.model.predictor
    B, T = 3, 10
    hidden = torch.zeros(B, T, 4)
    mask = torch.zeros(B, 1, T, dtype=torch.bool)
    for i, L in enumerate((4, 7, 10)):
        mask[i, 0, :L] = True
    ds, dsp, us, usp = pred.get_upsample_timestamp(hidden, mask)
    assert calls == [4, 7, 10]                 # 逐条切到各自有效长
    assert ds.shape == (B, T) and us.shape == (B, T * 3)   # 零填充回批宽度

    calls.clear()
    pred.get_upsample_timestamp(hidden, torch.ones(B, 1, T, dtype=torch.bool))
    assert calls == [T]                        # 等长满宽：原批路径一次

    calls.clear()
    pred.get_upsample_timestamp(hidden[:1], torch.ones(1, 1, T, dtype=torch.bool))
    assert calls == [T]                        # B=1 直通

    # 非 blstm 上采样（cnn_attn 带掩码，无缺陷）→ 不装补丁
    class _AttnPred(_Pred):
        upsample_type = "cnn_attn"

    m2 = _Model()
    m2.model.predictor = _AttnPred()
    assert not funasr_ts._install_batch_safe_predictor(m2)


# ── 超长 part 硬切：_split_part / _split_groups（asr 组帧数硬封顶的兜底层） ──

def _tone(sec: float, seed: int = 0) -> np.ndarray:
    """确定性正弦+固定种子噪声（与 validate_asr_batch 同风格）。"""
    rng = np.random.default_rng(seed)
    n = int(24000 * sec)
    return (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / 24000)
            + 0.05 * rng.standard_normal(n)).astype(np.float32)


def _tone_with_silence(sec: float, gap_at: float) -> np.ndarray:
    """[gap_at, gap_at+1) 秒处挖 1s 精确静音的测试音频。"""
    a = _tone(sec)
    a[int(gap_at * 24000):int((gap_at + 1) * 24000)] = 0.0
    return a


def test_split_part_cuts_at_silence():
    """60s 音频在 28s 处有 1s 静音：首刀落静音内（窗口 [28s,30s] 的 RMS
    argmin 命中零能量帧）；文本两侧非空、拼接无损；各片 ≤30s。"""
    sr = 24000
    audio = _tone_with_silence(60, gap_at=28)
    text = "字" * 100
    pieces = funasr_ts._split_part(audio, text, sr, 30)
    cut = len(pieces[0][0])
    assert 28 * sr <= cut <= 29 * sr           # 刀落静音内（100ms 窗中点）
    assert all(len(a) <= 30 * sr for a, _t in pieces)
    assert all(len(t) > 0 for _a, t in pieces)
    assert "".join(t for _a, t in pieces) == text
    assert sum(len(a) for a, _t in pieces) == len(audio)   # 无缝覆盖


def test_split_part_continuous_noise_caps():
    """无静音连续噪声（90s）：无静音可寻 → argmin 落窗内任意帧，但确定性；
    刀 ≤ target → 各片 ≤30s，循环至剩余 ≤ 上限。"""
    sr = 24000
    audio = _tone(90, seed=1)
    text = "字" * 200
    pieces = funasr_ts._split_part(audio, text, sr, 30)
    assert len(pieces) >= 3
    assert all(len(a) <= 30 * sr for a, _t in pieces)
    assert sum(len(a) for a, _t in pieces) == len(audio)
    assert "".join(t for _a, t in pieces) == text


def test_split_part_single_char_no_split():
    """1 字文本无法按比例分配 → 安全阀：整体不切（旧行为，宁超预算不错位）。"""
    audio = _tone(60, seed=2)
    pieces = funasr_ts._split_part(audio, "字", 24000, 30)
    assert len(pieces) == 1 and len(pieces[0][0]) == len(audio)


def test_split_groups_hard_splits_oversized_part():
    """_split_groups 集成：parts=[10s 正常, 60s 超长] → 超长项被硬切成
    ≤30s 的片段组；组音频时长之和 = 原始总时长（偏移累进基准不破）。"""
    sr = 24000
    normal = _tone(10, seed=3)
    long_audio = _tone_with_silence(60, gap_at=28)
    groups = funasr_ts._split_groups([normal, long_audio],
                                     ["甲", "乙" * 100], sr, 30)
    assert all(sum(len(a) for a in ga) <= 30 * sr for ga, _t in groups)
    assert sum(sum(len(a) for a in ga) for ga, _t in groups) == 70 * sr
    assert "".join(t for _ga, tl in groups for t in tl) == "甲" + "乙" * 100


def test_split_groups_multi_part_preflush_respects_cap():
    """多 part 前置判满：20s+20s（cap=30）必须切成两组，不能组成 40s 组
    （组帧数硬封顶是 ASR 显存预算的前提）。"""
    sr = 24000
    a = np.zeros(20 * sr, dtype=np.float32)
    groups = funasr_ts._split_groups([a, a], ["甲", "乙"], sr, 30)
    assert len(groups) == 2
    assert all(sum(len(x) for x in ga) <= 30 * sr for ga, _t in groups)
    assert "".join(t for _ga, tl in groups for t in tl) == "甲乙"


def test_batch_group_oversized_offsets_monotonic(_batch_mode):
    """批路径偏移数学在硬切下不破：恒定桩每组回 1 条 → 条数 = 组数；
    偏移严格递增（首刀落静音 ≈28.15s、第二片 ≈28.05s）、右端 ≤ 总时长。"""
    sr = 24000
    long_audio = _tone_with_silence(60, gap_at=28)
    out = funasr_ts.batch_group_timestamp([([long_audio], ["乙" * 100])],
                                          sr, max_group_sec=30)
    ts = out[0]["timestamp"]
    assert len(ts) == 3                        # 桩每组一条：60s 硬切恰 3 组
    assert all(s <= e for s, e in ts)
    assert all(ts[i][0] < ts[i + 1][0] for i in range(len(ts) - 1))
    assert ts[-1][1] <= 60 * 1000 + 1000       # ≤ 总时长（毫秒，含余量）


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_funasr_math.py")

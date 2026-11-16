"""量化上移 NPU 的逐字节等价性：finalize_audio（卡上 fade/量化，D2H int16）
与旧 CPU 路径（crop→fade→nan→clip→×32767→rint→astype）必须产出完全相同的
int16 波形。

旧路径顺序（重构前，逐段复刻）：
  forward_batch: crop(按帧) → seg[-fade:] *= ramp
  ship_batch:    nan_to_num → clip(±1) → ×32767 → np.rint → astype("<i2")
新路径顺序（finalize_audio，卡上）：
  fade(全张量上按每句 end 定位) → nan_to_num → clip → ×32767 → round → int16
关键不变量：np.rint 与 torch.round 同为 half-to-even；torch .to(int16) 对
整数值 float 与 C 截断一致（值已是 ±0.0 整数）。

本模块 import torch（batched_model 模块级依赖）——torch 不可安全导入的
环境（QEMU 仿真：xbyak 段错误）整文件跳过（段错误无法 try/except，只能
子进程探测；npu_ops._torch_import_ok 同款机制）。
"""
import numpy as np
import pytest

from npu_ops import _torch_import_ok

pytestmark = pytest.mark.skipif(
    not _torch_import_ok(),
    reason="torch 不可导入（QEMU/坏环境）；等价性对拍在真机/开发机跑")


def _finalize():
    from batched_model import finalize_audio

    return finalize_audio


def _torch():
    import torch

    return torch

RAMP = np.linspace(1.0, 0.0, int(0.005 * 24000), dtype=np.float32)  # 120 样本
RATIO = 300            # 每帧样本数（80fps × 300 = 24kHz）
FADE = len(RAMP)


def old_reference(audio_np: np.ndarray, frames, ratio: int) -> list:
    """重构前的 CPU 路径逐段复刻（forward_batch 尾 + ship_batch 量化）。"""
    out = []
    for i, fr in enumerate(frames):
        seg = audio_np[i, 0, :int(fr) * ratio].copy()
        if seg.shape[0] > FADE:
            seg[-FADE:] *= RAMP
        arr = np.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
        out.append(np.rint(np.clip(arr, -1.0, 1.0) * 32767).astype("<i2"))
    return out


def _assert_equal(audio_np, frames, ratio, tag):
    ref = old_reference(audio_np, frames, ratio)
    t = _torch()
    got = _finalize()(t.from_numpy(audio_np),
                      t.tensor(frames, dtype=t.long), ratio)
    assert len(got) == len(ref)
    for i, (g, r) in enumerate(zip(got, ref)):
        assert g.dtype == np.int16 == r.dtype, (tag, i, g.dtype)
        assert g.tobytes() == r.tobytes(), \
            f"{tag}[{i}] 不等价: len={g.shape[0]}/{r.shape[0]} " \
            f"first_diff={np.flatnonzero(g != r)[:5]}"


def test_equivalent_normal_signal():
    """常态信号（含正负、越界 ±1.5、0/满幅边界）逐字节一致。"""
    rng = np.random.default_rng(7)
    frames = [50, 120, 9, 301]
    s_max = max(frames) * RATIO
    audio = rng.uniform(-1.0, 1.0, size=(4, 1, s_max)).astype(np.float32)
    audio[0, 0, :100] = np.nan            # NaN：旧路径 astype 未定义区
    audio[1, 0, 500:510] = np.inf         # +Inf → 1.0
    audio[1, 0, 510:520] = -np.inf        # -Inf → -1.0
    audio[2, 0, :50] = 1.5                # 越界（clip 语义）
    audio[2, 0, 50:60] = -1.5
    audio[3, 0, 100] = 1.0                # 满幅正
    audio[3, 0, 101] = -1.0               # 满幅负
    _assert_equal(audio, frames, RATIO, "normal")


def test_equivalent_fade_region():
    """fade 窗内外的值（含窗内 NaN/Inf/越界）一致：fade 在量化前、裁剪区尾。"""
    rng = np.random.default_rng(11)
    frames = [200, 201]
    s_max = max(frames) * RATIO
    audio = rng.uniform(-0.8, 0.8, size=(2, 1, s_max)).astype(np.float32)
    end0 = frames[0] * RATIO
    audio[0, 0, end0 - FADE:end0] = np.linspace(-2.0, 2.0, FADE)  # 窗内越界
    audio[0, 0, end0 - 5] = np.nan                                # 窗内 NaN
    audio[0, 0, end0 - 6] = np.inf                                # 窗内 Inf
    _assert_equal(audio, frames, RATIO, "fade-region")


def test_equivalent_short_sentences_no_fade():
    """end ≤ fade 的极短句：旧路径不淡出，新路径同样跳过；零帧空段也对齐。"""
    rng = np.random.default_rng(13)
    s_max = 2 * RATIO
    audio = rng.uniform(-1.0, 1.0, size=(2, 1, s_max)).astype(np.float32)
    audio[0, 0, :] = np.nan               # 全 NaN 空段（end=0，无淡出）
    _assert_equal(audio, [0, 1], RATIO, "empty-and-short")
    # 缩小 ratio 构造极短（end=80/120 均 ≤ fade=120；严格 end>fade 才淡出，
    # 含 end==fade 边界）
    audio2 = rng.uniform(-1.0, 1.0, size=(2, 1, 200)).astype(np.float32)
    audio2[1, 0, 30:35] = np.nan
    _assert_equal(audio2, [2, 3], 40, "no-fade")   # 80/120 都不淡出


def test_equivalent_half_to_even_rounding():
    """半到偶舍入逐位一致：构造恰好落在 x.5 的样本。"""
    frames = [10]
    audio = np.zeros((1, 1, 10 * RATIO), dtype=np.float32)
    vals = [0.5 / 32767, 1.5 / 32767, 2.5 / 32767, -0.5 / 32767, -1.5 / 32767]
    for k, v in enumerate(vals):
        audio[0, 0, 100 + k] = v          # ×32767 = ±0.5/1.5/2.5 → rint 半到偶
    _assert_equal(audio, frames, RATIO, "rounding")

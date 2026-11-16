#!/usr/bin/env python3
"""批量前向 vs 官方单样本前向 的一致性验证（任意设备可跑，CPU 即可）。

三个关卡:
  A. B=1 退化回归:  相同随机种子下 forward_batch 必须与 model.forward 几乎逐位一致
     （时长逐位一致；音频经 PCM16 量化，非淡出区仅允许 ≤1 LSB 差异）
  B. 多样本一致性:  批量结果 vs 逐个单样本结果，时长/音频长度必须一致，
     波形相关性因 SineGen 噪声实现不同而要求 >= 阈值
  C. 批量吞吐:      同机同文本 批量 vs 串行单样本 的前向耗时对比

注意：synth_batch 直出 int16（fade/量化已上移设备侧），比较前需
/32767 还原 float；句尾 5ms fade 窗与单样本路径不同，A 关排除该窗。

用法:
    HF_HOME=.hf-cache DEVICE=cpu .venv/Scripts/python.exe tests/validate_batch.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", ".hf-cache")
os.environ.setdefault("DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pipeline import G2PEngine, TTSCompute  # noqa: E402

CORR_PASS = 0.85


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def single_forward(engine, ps, voice, speed, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    ref = engine.make_ref(ps, voice)
    with torch.no_grad():
        out = engine.model(ps, ref, speed, return_output=True)
    return (out.audio.detach().float().cpu().numpy(),
            int(out.pred_dur.sum().item()))


def batch_forward(engine, jobs, seed=None):
    """批量合成（可选先设随机种子，用于 B=1 逐位回归）。"""
    if seed is not None:
        torch.manual_seed(seed)
    return engine.synth_batch(jobs)


def main():
    import huggingface_hub.constants as _hfc

    print(f"env: offline={_hfc.HF_HUB_OFFLINE} home={_hfc.HF_HOME} endpoint={_hfc.ENDPOINT}")
    try:
        engine = TTSCompute(os.getenv("DEVICE", "cpu"))
        g2p = G2PEngine()
    except Exception as exc:  # noqa: BLE001
        print(f"模型/权重不可用，本数值门需真实权重（{exc}）；"
              f"请在已构建镜像或已下载 HF 缓存的环境运行。")
        return 2
    print(f"device={engine.device}\n")

    cases = [  # (句子, 音色, 语速)
        ("你好，欢迎使用昇腾算力。", "zf_001", 1.0),
        ("Kokoro 是一个开源的语音合成模型。", "zf_001", 1.0),
        ("今天天气不错，我们去公园散步吧！", "zm_009", 1.1),
        ("人工智能正在改变世界。", "zm_009", 0.9),
    ]
    # G2P（文本引擎独立于合成引擎）
    jobs = []
    for sent, voice, speed in cases:
        ps = g2p.phonemize(sent)[0]
        jobs.append((ps, voice, speed))
        print(f"  [{voice} x{speed}] {sent!r} -> ps({len(ps)})")

    all_pass = True

    # ---- 关卡 A: B=1 退化回归（同种子应几乎逐位一致）----
    print("\n== A. B=1 退化回归 ==")
    fade = int(0.005 * 24000)   # finalize_audio 的淡出窗（单样本路径无 fade）
    for i, (ps, voice, speed) in enumerate(jobs):
        single, _dur = single_forward(engine, ps, voice, speed, seed=1234)
        batched = batch_forward(engine, [(ps, voice, speed)], seed=1234)[0]
        bf = batched.astype(np.float32) / 32767.0
        n = max(0, min(len(single), len(bf)) - fade)
        diff = float(np.abs(single[:n] - bf[:n]).max()) if n else 1.0
        ok = (diff < 1e-4 and len(single) == len(batched)
              and int(batched[-1]) == 0)   # fade 末样本必须精确归零
        all_pass &= ok
        print(f"  case{i}: len {len(single)}=={len(batched)} "
              f"max|diff|={diff:.2e} tail={int(batched[-1])} "
              f"{'PASS' if ok else 'FAIL'}")

    # ---- 关卡 B: 多样本批量 vs 逐个单样本 ----
    print("\n== B. 批量 vs 单样本一致性（不同随机实现，看结构一致性）==")
    singles, single_frames = [], []
    for ps, voice, speed in jobs:
        audio, dur = single_forward(engine, ps, voice, speed)  # 各自独立噪声
        singles.append(audio)
        single_frames.append(dur)
    audio_list = engine.synth_batch(jobs)
    ratio = len(audio_list[0]) / single_frames[0] if single_frames[0] else 0
    for i, (audio_b, (ps, voice, speed)) in enumerate(zip(audio_list, jobs)):
        len_single = int(single_frames[i] * ratio)
        len_ok = abs(len(audio_b) - len_single) <= int(0.002 * 24000)  # ±2ms 内
        c = corr(singles[i], audio_b)
        ok = len_ok and c >= CORR_PASS
        all_pass &= ok
        print(f"  case{i}: len batch={len(audio_b)} single≈{len_single} corr={c:.4f} {'PASS' if ok else 'FAIL'}")

    # ---- 关卡 B2: 同批混合音色/语速 + 长短混合（分桶见 server，此处验证引擎不炸且合理）----
    print("\n== B2. 混合语速/音色批量 ==")
    mixed = [(jobs[0][0], "zm_009", 1.3), (jobs[3][0], "zf_001", 0.8)]
    single_m = [single_forward(engine, ps, v, s)[0] for ps, v, s in mixed]
    batch_m = engine.synth_batch(mixed)
    for i in range(2):
        c = corr(single_m[i], batch_m[i])
        ok = c >= CORR_PASS and abs(len(single_m[i]) - len(batch_m[i])) < 0.01 * 24000 * 4
        all_pass &= ok
        print(f"  case{i}: corr={c:.4f} len {len(single_m[i])}/{len(batch_m[i])} {'PASS' if ok else 'FAIL'}")

    # ---- 关卡 C: 前向耗时对比（吞吐信号）----
    print("\n== C. 前向耗时（CPU 上为相对信号，NPU 上差距更大）==")
    big = jobs * 4  # 16 句
    t0 = time.perf_counter()
    for ps, voice, speed in big:
        single_forward(engine, ps, voice, speed)
    t_single = time.perf_counter() - t0
    t0 = time.perf_counter()
    engine.synth_batch(big)
    t_batch = time.perf_counter() - t0
    print(f"  16 句: 串行={t_single:.2f}s  批量={t_batch:.2f}s  加速≈{t_single / t_batch:.1f}x")

    print("\n" + ("ALL PASS ✔" if all_pass else "存在 FAIL ✘"))
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

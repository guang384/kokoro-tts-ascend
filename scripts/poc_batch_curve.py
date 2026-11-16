#!/usr/bin/env python3
"""POC③：批量成本曲线 + padding 浪费率——吞吐杠杆 ①② 的定量依据（独立脚本）。

背景
    吞吐天花板 = decoder 的 fp32 执行路径（占前向 59%、不随流重叠）。
    两个尚无量化的杠杆：
    ① batch 大小曲线：更大的 B → conv GEMM 更"胖" → fp32 路径单 kernel
      利用率上升 → 每句成本下降？扫 B=1..32 测 per-chunk 时间曲线。
    ② padding 浪费：批内按最长句的 T_max 计算所有句——长短混批时短句
      按长句全价买单。测混合批 vs 统一长/短批的耗时 + 实际 pad 比例。

怎么跑（推荐空闲卡；同服务卡需降 --target-mb 6000 --max-b 8）
    docker run --rm -it \\
      --device /dev/davinci3 --device /dev/davinci_manager \\
      --device /dev/devmm_svm --device /dev/hisi_hdc \\
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\
      -v /usr/local/dcmi:/usr/local/dcmi:ro \\
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
      -v /root/cui:/poc -e PYTHONPATH=/app -e ASCEND_DEVICE_ID=0 -w /app \\
      --entrypoint /app/scripts/with_ascend_env.sh \\
      kokoro-tts-ascend:20261113-r2 \\
      python3 /poc/poc_batch_curve.py --device npu:0

判读
    ① chunk_ms 随 B 明显下降（B16/B32 比 B8 便宜 ≥15%）→ 大批喂饱
      fp32 路径，"2 worker × B16" 成立；曲线平 → batch 维度到头。
    ② 混合批耗时 ≈ 统一长句批（差 <5%）→ 短句全价买单实锤，
      组批按长度收紧的收益 = pad_ratio × decoder 份额(59%)。
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

_POC_LONG = ("华为昇腾处理器面向人工智能训练与推理场景，语音合成服务在"
             "容器队列中弹性扩容，批次吞吐随并发拐点稳定上升。")   # ~55 字
_POC_SHORT = "华为语音合成稳定。"                                    # ~9 字

FRAMES_PER_SEC = 80
SAMPLES_PER_FRAME = 300   # 24000/80


def parse_args():
    ap = argparse.ArgumentParser(description="批量成本曲线 + padding 浪费 POC")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--max-b", type=int, default=32, help="B 扫描上限")
    ap.add_argument("--iters", type=int, default=4, help="每档前向次数")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--target-mb", type=int, default=30000,
                    help="最大档库存预算 MB（空闲卡 30000；同服务卡降到 6000）")
    return ap.parse_args()


def fail(msg: str) -> None:
    print(f"VERDICT: INCONCLUSIVE — {msg}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        fail(f"import torch_npu 失败: {exc}")
    npu = torch.npu
    dev_index = int(args.device.split(":")[1]) if ":" in args.device else 0
    npu.set_device(dev_index)

    import pipeline

    pipeline._prime_memory = lambda device: None
    voice = pipeline._default_voice
    compute = pipeline.TTSCompute(args.device)

    from batched_model import phonemes_to_ids

    def g2p(text: str, fb_len: int) -> str:
        try:
            ps_list = pipeline.get_g2p().phonemize(text)
            if ps_list and ps_list[0]:
                return ps_list[0]
        except Exception as exc:  # noqa: BLE001
            print(f"g2p fallback: {exc}")
        return "ə" * fb_len

    ps_long = g2p(_POC_LONG, 72)
    ps_short = g2p(_POC_SHORT, 12)
    l_long = len(phonemes_to_ids(compute.model, ps_long))
    l_short = len(phonemes_to_ids(compute.model, ps_short))

    # 自动缩批：按最长句 B=1 的首前向库存 × 1.4 余量 ≤ target
    npu.synchronize()
    r0 = npu.memory_reserved()
    out1 = compute.synth_batch([(ps_long, voice, 1.0)], args.dtype)
    npu.synchronize()
    live1 = max((npu.memory_reserved() - r0) / 2**20, 64.0)
    if len(out1) != 1 or not hasattr(out1[0], "shape"):
        fail("warmup 前向返回异常——模型路径不通")
    b_cap = max(1, min(args.max_b, int(args.target_mb / (1.4 * live1))))

    def bench(jobs: list, iters: int) -> float:
        npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            compute.synth_batch(jobs, args.dtype)
        npu.synchronize()
        return time.perf_counter() - t0

    def pad_stats(out: list) -> tuple[int, int, float]:
        """从输出音频反推帧数：frames_i = samples/300；pad 比例按 T_max。"""
        frames = [round(len(a) / SAMPLES_PER_FRAME) for a in out]
        t_max = max(frames)
        total = sum(frames)
        pad = (t_max * len(frames) - total) / (t_max * len(frames))
        return t_max, total, max(0.0, pad)

    print(f"env    : torch={torch.__version__} device={args.device} "
          f" dtype={args.dtype} b_cap={b_cap} live/B1={live1:.0f}MB")
    print(f"tokens : long={l_long} short={l_short}")
    print(f"script : poc_batch_curve v1")

    # ── ① B 扫描（统一长句形状）─────────────────────────────────────
    sweep = [b for b in (1, 2, 4, 8, 16, 32) if b <= b_cap]
    print(f"\n== ① B 扫描（统一长句 {l_long} tokens，{args.iters} 次/档）==")
    print(f"{'B':>4} {'batch_ms':>10} {'chunk_ms/句':>12} {'audio/句':>9} "
          f"{'vs B=8':>8}")
    chunk_ms = {}
    audio_ref = None
    for b in sweep:
        jobs = [(ps_long, voice, 1.0)] * b
        compute.synth_batch(jobs, args.dtype)   # 形状首建/autotune（不入计时）
        npu.synchronize()
        dt = bench(jobs, args.iters)
        out = compute.synth_batch(jobs, args.dtype)
        audio_s = len(out[0]) / 24000.0
        audio_ref = audio_ref or audio_s
        chunk_ms[b] = dt / (args.iters * b) * 1000
        rel = f"{chunk_ms[b] / chunk_ms[8]:.2f}x" if 8 in chunk_ms else "-"
        print(f"{b:>4} {dt / args.iters * 1000:>10.0f} {chunk_ms[b]:>12.1f} "
              f"{audio_s:>9.1f} {rel:>8}")

    # ── ② padding 浪费（混合长度 vs 统一长/短）──────────────────────
    b2 = min(b_cap, 16)
    n_long = b2 // 2
    configs = [
        ("uniform-long", [(ps_long, voice, 1.0)] * b2),
        ("mixed(长+短)", [(ps_long, voice, 1.0)] * n_long
         + [(ps_short, voice, 1.0)] * (b2 - n_long)),
        ("uniform-short", [(ps_short, voice, 1.0)] * b2),
    ]
    print(f"\n== ② padding 浪费（B={b2}，{args.iters} 次/档）==")
    print(f"{'config':<16} {'组成':<22} {'T_max':>7} {'Σframes':>8} "
          f"{'pad_ratio':>9} {'batch_ms':>9} {'ms/句':>7}")
    stats = {}
    for name, jobs in configs:
        compute.synth_batch(jobs, args.dtype)   # 形状首建/autotune（不入计时）
        npu.synchronize()
        dt = bench(jobs, args.iters)
        out = compute.synth_batch(jobs, args.dtype)
        t_max, total_f, pad = pad_stats(out)
        stats[name] = dt / args.iters * 1000
        lens = [len(j[0]) for j in jobs]
        comp = f"{max(lens)}/{min(lens)}字符×{b2}"
        print(f"{name:<16} {comp:<22} {t_max:>7} {total_f:>8} "
              f"{pad:>9.1%} {stats[name]:>9.0f} {stats[name] / b2:>7.1f}")

    # ── 结论 ────────────────────────────────────────────────────────
    b_top = max(sweep)
    b_ref = 8 if 8 in chunk_ms else min(sweep)
    drop = 1 - chunk_ms[b_top] / chunk_ms[b_ref]
    print(f"\n== 结论 ==")
    if drop >= 0.15:
        print(f"① B 曲线：B{b_top} 每句成本比 B{b_ref} 便宜 {drop:.0%} → "
              f"大批喂饱 fp32 路径，'2 worker × B{b_top}' 值得上")
    elif drop >= 0.05:
        print(f"① B 曲线：B{b_top} 每句便宜 {drop:.0%}（5%~15%）→ 有收益但"
              f"幅度有限，按显存余量决定是否上大批")
    else:
        print(f"① B 曲线：平坦（差异 {drop:.0%}）→ batch 维度已到头，"
              f"fp32 路径在 B{b_ref} 就已饱和")
    ul, mx = stats["uniform-long"], stats["mixed(长+短)"]
    us = stats["uniform-short"]
    if abs(mx - ul) / ul < 0.05:
        print(f"② padding：混合批耗时 = 长句批（差 {abs(mx - ul) / ul:.0%}）→ "
              f"短句按最长句全价买单实锤；按长度收紧组批的收益 = "
              f"pad_ratio × 59%（decoder 份额）")
    else:
        print(f"② padding：混合批比长句批便宜 {1 - mx / ul:.0%} → 部分阶段"
              f"按真实长度工作，浪费低于理论值")
    print(f"② 参考：全短批耗时 = 长句批的 {us / ul:.0%} → 句长对成本的影响量级")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("POC 执行异常（见堆栈）")

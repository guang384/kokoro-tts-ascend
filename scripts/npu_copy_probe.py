#!/usr/bin/env python3
"""卡↔主机拷贝带宽实测（在含 torch_npu 的环境跑，容器内/真机）。

背景：Ascend 的 D2H/H2D 走 ACL memcpy，非 pinned（pageable）内存会经
CPU 侧中转，实测带宽常比 pinned 低数倍。本探针量化：
  1) H2D/D2H 带宽（1/4/16/64MB，pageable vs pinned）
  2) 音频真实形态：[8,1,600000] fp32（旧路径 19.2MB）与 int16（新路径 9.6MB）
  3) 小张量 .cpu() 同步延迟（forward_batch 中途 frames 同步的成本）
  4) 逐 batch 收益估算（新旧路径 D2H 字节差）

用法（容器内）：
    python3 scripts/npu_copy_probe.py
退出码恒 0；结论看打印。
"""
import time

import torch


def bench(fn, iters=30, warmup=5) -> float:
    """返回每次毫秒。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters


def main() -> int:
    __import__("torch_npu")   # 注册 torch.npu 后端

    dev = torch.device("npu:0")
    print(f"device={dev}  torch={torch.__version__}")
    print("=" * 72)

    # ── 1. 带宽矩阵 ──
    print(f"{'size':>8} {'dir':>4} {'pageable':>12} {'pinned':>12} {'speedup':>8}")
    for mb in (1, 4, 16, 64):
        n = mb * 2**20 // 4
        host = torch.randn(n)
        host_pinned = torch.randn(n).pin_memory()
        dev_t = torch.randn(n, device=dev)

        h2d_p = bench(lambda: host.to(dev, non_blocking=False))
        h2d_pin = bench(lambda: host_pinned.to(dev, non_blocking=False))
        d2h_p = bench(lambda: dev_t.cpu())
        buf = torch.empty(n, dtype=torch.float32).pin_memory()
        d2h_pin = bench(lambda: buf.copy_(dev_t))

        def gb(ms, nbytes):
            return nbytes / 2**30 / (ms / 1000)

        print(f"{mb:>6}MB {'H2D':>4} {h2d_p:>8.2f}ms {h2d_pin:>8.2f}ms "
              f"{h2d_p / h2d_pin:>7.1f}x   ({gb(h2d_p, mb * 2**20):.1f} vs "
              f"{gb(h2d_pin, mb * 2**20):.1f} GB/s)")
        print(f"{mb:>6}MB {'D2H':>4} {d2h_p:>8.2f}ms {d2h_pin:>8.2f}ms "
              f"{d2h_p / d2h_pin:>7.1f}x   ({gb(d2h_p, mb * 2**20):.1f} vs "
              f"{gb(d2h_pin, mb * 2**20):.1f} GB/s)")

    # ── 2. 音频真实形态（B=8，单句 8s = 192000 样本）──
    print("-" * 72)
    n_audio = 8 * 1 * 8 * 24000
    a_f32 = torch.randn(n_audio, device=dev)
    a_i16 = torch.randint(-32768, 32767, (n_audio,), dtype=torch.int16,
                          device=dev)
    t_f32 = bench(lambda: a_f32.cpu())
    t_i16 = bench(lambda: a_i16.cpu())
    print(f"音频批 D2H: fp32 {n_audio * 4 / 2**20:.1f}MB = {t_f32:.2f}ms | "
          f"int16 {n_audio * 2 / 2**20:.1f}MB = {t_i16:.2f}ms "
          f"(量化上移后每批省 {t_f32 - t_i16:.2f}ms)")

    # ── 3. 小张量同步延迟（frames.cpu() 的固有成本）──
    tiny = torch.zeros(8, dtype=torch.long, device=dev)
    t_sync = bench(lambda: tiny.cpu(), iters=200)
    print(f"小张量 D2H 同步延迟（frames.cpu 形态）: {t_sync:.3f}ms "
          f"——占 2.4s 前向的 {t_sync / 2400 * 100:.2f}%")

    # ── 4. pinned 是否可用 ──
    try:
        _ = torch.zeros(1024).pin_memory()
        print("pin_memory: 可用 —— 若 pageable/pinned 差距 >2x，建议给 "
              "finalize_audio 的 D2H 加常驻 pinned 暂存缓冲（下一版）")
    except Exception as exc:  # noqa: BLE001
        print(f"pin_memory: 不可用（{type(exc).__name__}）——pinned 优化不可行")

    print("=" * 72)
    print("判读：D2H pageable 带宽 <1GB/s 且 pinned 显著更快 → 回传结果，"
          "下一版加 pinned 暂存；差距小 → 拷贝不是瓶颈，优化重点在桶宽/"
          "批形状（见 README 调参表）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

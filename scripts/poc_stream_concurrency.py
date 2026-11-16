#!/usr/bin/env python3
"""POC：910B 单进程多 Stream 并发前向可行性验证（独立脚本，不进服务路径）。

背景
    统一显存池方案（单进程 + K 路并发前向）的成败点：同一进程里多路前向
    在各自独立的 torch_npu Stream 上能否真正重叠（默认流会串行化）。
    本脚本在真实 KModel + 真实形状上测 overlap 效率，直接给 PASS/PARTIAL/
    FAIL 结论——这决定"单进程统一池"是值得实施还是放弃。

怎么跑（在真机的服务容器里；运行期间保持服务空闲，无压测流量）
    # 容器名按实际改；POC 不预占预算（卡上需 ~5-7GB 空闲）
    docker cp scripts/poc_stream_concurrency.py kokoro-tts:/app/poc_stream_concurrency.py
    docker exec kokoro-tts scripts/with_ascend_env.sh python /app/poc_stream_concurrency.py

    可选参数：
      --device npu:0        设备
      --streams 2,4         并发路数档位（1/省略 = 只跑串行基线）
      --iters 6             每路前向次数（总工作 = K × iters）
      --batch 8             每 batch chunk 数上限（会按空闲显存自动下调）
      --dtype fp32          前向精度（fp16 可进一步降显存）
      --target-mb 6500      K 路并发激活总预算（含 1.4 库存余量）

判读标准（脚本自动输出 VERDICT）
    K=4 overlap 效率 ≥ 0.6   → PASS：多 Stream 并发有效，统一池方案可行
    0.3 ≤ 效率 < 0.6         → PARTIAL：有收益但受限，方案收益打折重估
    效率 < 0.3 / Stream API 缺失 / 报错 → FAIL：当前栈不支持 → 方案放弃

预期输出样例
    RESULT serial        : 6 forwards, 3.21s, 0.535s/fwd
    RESULT concur K=2    : wall 3.45s → speedup 1.86x (efficiency 0.93) ok=12 err=0
    RESULT concur K=4    : wall 1.98s → speedup 3.23x (efficiency 0.81) ok=24 err=0
    VERDICT: PASS — ...
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import traceback

_POC_TEXT = ("华为昇腾处理器面向人工智能训练与推理场景，语音合成服务在"
             "容器队列中弹性扩容，批次吞吐随并发拐点稳定上升。")  # ~55 字


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="910B 多 Stream 并发前向 POC")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--streams", default="2,4",
                    help="并发路数档位，逗号分隔；串行基线恒跑")
    ap.add_argument("--iters", type=int, default=6,
                    help="每路前向次数（串行基线同次数）")
    ap.add_argument("--batch", type=int, default=8,
                    help="每批 chunk 数上限（按空闲显存自动下调）")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--voice", default=None, help="默认取 KOKORO_VOICE/zf_001")
    ap.add_argument("--target-mb", type=int, default=6500,
                    help="K 路并发激活总预算 MB（含 1.4 库存余量）")
    ap.add_argument("--raw", action="store_true",
                    help="裸 kernel 模式：预构建输入张量，线程内只跑 "
                         "forward_batch——把 kernel 重叠与 Python/GIL 预处理拆开对比")
    return ap.parse_args()


def get_npu_api():
    """import torch_npu 并返回 torch.npu 命名空间；Stream API 缺失 = FAIL 结论。"""
    import torch

    try:
        import torch_npu  # noqa: F401  注册 torch.npu 后端
    except ImportError as exc:
        fail(f"import torch_npu 失败: {exc}")

    npu = getattr(torch, "npu", None)
    if npu is None:
        fail("torch.npu 命名空间不存在（torch_npu 未注册）")
    for attr in ("Stream", "stream", "synchronize", "set_device"):
        if not hasattr(npu, attr):
            fail(f"torch.npu.{attr} 缺失——多流并发不可行（POC 结论：FAIL）")
    return torch, npu


def fail(msg: str) -> None:
    print(f"VERDICT: FAIL — {msg}")
    sys.exit(1)


def build_ps(text: str) -> str:
    """真实 g2p 音素串；失败退合成 ps（同为 ~70 字符级序列，形状等价）。"""
    try:
        from pipeline import get_g2p

        ps_list = get_g2p().phonemize(text)
        ps = ps_list[0] if ps_list else ""
        if ps:
            print(f"g2p: {len(ps)} chars phonemes")
            return ps
    except Exception as exc:  # noqa: BLE001
        print(f"g2p fallback: {exc}")
    return "ə" * 72


class WorkerStats:
    def __init__(self):
        self.ok = 0
        self.errors: list[str] = []


def run_concurrent(npu, fn, expect_n: int, k: int, iters: int,
                   dev_index: int) -> tuple[float, list[WorkerStats]]:
    """K 线程 × 各自独立 Stream，Barrier 同步起跑，全局 sync 后计时。
    fn() = 单次前向调用（synth 路径或裸 forward_batch 路径）。
    返回 (墙钟秒, 各线程统计)。"""
    stats = [WorkerStats() for _ in range(k)]
    barrier = threading.Barrier(k + 1)

    def worker(tid: int) -> None:
        stream = None
        try:
            npu.set_device(dev_index)
            stream = npu.Stream()
            with npu.stream(stream):
                barrier.wait()
                for _ in range(iters):
                    try:
                        out = fn()
                        if len(out) == expect_n and all(
                                hasattr(a, "shape") for a in out):
                            stats[tid].ok += 1
                        else:
                            stats[tid].errors.append("返回数量/类型异常")
                    except Exception as exc:  # noqa: BLE001
                        stats[tid].errors.append(f"{type(exc).__name__}: "
                                                 f"{str(exc)[:120]}")
        except Exception as exc:  # noqa: BLE001  含 BarrierBrokenError
            stats[tid].errors.append(f"线程级: {type(exc).__name__}: "
                                     f"{str(exc)[:160]}")
        finally:
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception:  # noqa: BLE001
                    pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(k)]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    barrier.wait()
    for t in threads:
        t.join()
    npu.synchronize()
    wall = time.perf_counter() - t0
    return wall, stats


def main() -> None:
    args = parse_args()
    torch, npu = get_npu_api()

    dev_index = int(args.device.split(":")[1]) if ":" in args.device else 0
    npu.set_device(dev_index)
    ks = [k for k in (int(x) for x in args.streams.split(",")) if k > 1]
    if not ks:
        fail("--streams 需要 >1 的档位（串行基线恒跑）")
    k_max = max(ks)

    # 复用 TTSCompute 全部前向机制，但跳过预算/预占（POC 在服务容器里轻装跑）
    import pipeline

    pipeline._prime_memory = lambda device: None
    voice = args.voice or pipeline._default_voice
    compute = pipeline.TTSCompute(args.device)

    ps = build_ps(_POC_TEXT)
    jobs_b1 = [(ps, voice, 1.0)]

    # ① 首前向 = warmup + 形状首建：reserved 增量 ≈ B1 全套库存（激活 +
    #    fade ramp + workspace）。必须在前向之前取基线，否则形状已缓存量不到。
    npu.synchronize()
    r0 = npu.memory_reserved()
    out0 = compute.synth_batch(jobs_b1, args.dtype)
    npu.synchronize()
    live1 = max((npu.memory_reserved() - r0) / 2**20, 64.0)
    if len(out0) != 1 or not hasattr(out0[0], "shape"):
        fail("warmup 前向返回异常——模型路径不通")

    # ② 按空闲预算自动缩批：K 路并发库存 × 1.4 余量 ≤ target
    b_fit = max(1, min(args.batch,
                       int(args.target_mb / (1.4 * k_max * live1))))
    if b_fit < 1:
        fail(f"空闲显存不足以并发（live/B1≈{live1:.0f}MB）——请确认服务空闲，"
             f"或降 --batch/--dtype fp16")
    jobs = [(ps, voice, 1.0)] * b_fit
    for _ in range(2):   # 最终形状再 warmup（B 相关库存/算法已覆盖）
        out0 = compute.synth_batch(jobs, args.dtype)
    npu.synchronize()

    samples = int(out0[0].shape[0]) if hasattr(out0[0], "shape") else 0
    audio_s = samples / 24000.0

    # 前向调用闭包：synth 全路径 vs 裸 forward_batch（预构建输入，剥离
    # phonemes_to_ids/组 batch 的 Python+GIL 段）——两模式对照定位串行点
    from batched_model import forward_batch, phonemes_to_ids
    if args.raw:
        import torch as _t

        ids = phonemes_to_ids(compute.model, ps)
        l_ids = len(ids)
        input_ids = _t.zeros((b_fit, l_ids), dtype=_t.long)   # 0 = PAD_ID
        for i in range(b_fit):
            input_ids[i, :l_ids] = _t.tensor(ids, dtype=_t.long)
        lengths = _t.full((b_fit,), l_ids, dtype=_t.long)
        ref_s = _t.cat([compute.make_ref(ps, voice)] * b_fit, dim=0)
        speeds = _t.ones(b_fit, dtype=_t.float32)
        raw_args = (compute.model, input_ids.to(compute.device),
                    lengths, ref_s.to(compute.device), speeds.to(compute.device))
        fn = lambda: forward_batch(*raw_args)   # noqa: E731
        mode = f"raw 裸 kernel（预构建输入，B={b_fit}）"
    else:
        fn = lambda: compute.synth_batch(jobs, args.dtype)   # noqa: E731
        mode = f"synth 全路径（含 Python 预处理，B={b_fit}）"
    expect_n = b_fit

    print(f"\n== POC 配置 ==")
    print(f"env    : torch={torch.__version__} "
          f"torch_npu={getattr(__import__('torch_npu'), '__version__', '?')} "
          f"device={args.device} dtype={args.dtype}")
    print(f"mode   : {mode}")
    print(f"shape  : B={b_fit} ps={len(ps)}chars live/B1={live1:.0f}MB "
          f"(K={k_max} 预计 ≈{1.4 * k_max * live1:.0f}MB ≤ target {args.target_mb}MB)")
    print(f"audio  : {audio_s:.1f}s/句 × {b_fit} 句/批")
    if b_fit == 1:
        print("WARN   : batch 自动缩到 1——launch/GIL 主导形状，并发效率会被"
              "低估；空闲卡上建议 --target-mb 30000 --batch 16 重跑\n")

    # ③ 串行基线（默认流，主线程）——生产现状等价
    n_serial = args.iters * k_max
    npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_serial):
        fn()
    npu.synchronize()
    t_serial = time.perf_counter() - t0
    per_fwd = t_serial / n_serial
    print(f"RESULT serial        : {n_serial} forwards, {t_serial:.2f}s, "
          f"{per_fwd:.3f}s/fwd")

    # ④ 并发档位：K 线程 × 独立 Stream
    eff_by_k: dict[int, float] = {}
    total_err = 0
    for k in ks:
        wall, stats = run_concurrent(npu, fn, expect_n, k,
                                     args.iters, dev_index)
        ok = sum(s.ok for s in stats)
        err = sum(len(s.errors) for s in stats)
        total_err += err
        speedup = (k * args.iters * per_fwd) / wall if wall else 0.0
        eff = speedup / k
        print(f"RESULT concur K={k:<2}   : wall {wall:.2f}s → speedup "
              f"{speedup:.2f}x (efficiency {eff:.2f}) ok={ok} err={err}")
        for i, s in enumerate(stats):
            for e in s.errors[:2]:
                print(f"    [K={k} thread{i}] {e}")
        if err == 0:
            eff_by_k[k] = eff

    # ⑤ 结论（判读锚 = 最大并发档 K_max）
    print(f"\nmemory : reserved={npu.memory_reserved() / 2**20:.0f}MB "
          f"allocated={npu.memory_allocated() / 2**20:.0f}MB")
    if total_err > 0:
        fail("并发前向出现异常（见上方）——统一池方案需先解决错误")
    # Amdahl 拟合：S_K = 1/(s+(1-s)/K) → s_K = (1/S_K − 1/K)/(1 − 1/K)
    ests = [(1.0 / S - 1.0 / k) / (1.0 - 1.0 / k)
            for k, S in eff_by_k.items() if k > 1 and S > 1.0]
    if ests:
        print(f"amdahl : 全局串行占比估计 ≈ {min(ests):.0%}~{max(ests):.0%}"
              f"（各 K 点拟合；偏高 → 疑似 GIL launch / acl_op 全局锁）")
    eff = eff_by_k.get(k_max, 0.0)
    if eff >= 0.6:
        print(f"VERDICT: PASS — K={k_max} 效率 {eff:.2f} ≥ 0.60：多 Stream 并发"
              f"前向有效，单进程统一显存池方案可行 [{mode}]")
    elif eff >= 0.3:
        print(f"VERDICT: PARTIAL — K={k_max} 效率 {eff:.2f} ∈ [0.3, 0.6)："
              f"并发有收益但受限，统一池收益打折重估 [{mode}]")
    else:
        print(f"VERDICT: FAIL — K={k_max} 效率 {eff:.2f} < 0.30：近似串行，"
              f"统一池方案放弃，退多进程折中形态 [{mode}]")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("POC 执行异常（见堆栈）")

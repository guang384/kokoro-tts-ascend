#!/usr/bin/env python3
"""Kokoro TTS 单 worker 显存精确计算器（解析式，非经验估算）。

模型维度全部来自 hexgrad/Kokoro-82M-v1.1-zh 的 config.json
（hidden_dim=512、plBERT 768/12层/2048、istftnet rates=[10,6] hop=5
initial=512、max_dur=50、style=128、context=512），算子序列与存活集
合同 batched_model.py::forward_batch（inference_mode 下张量引用即释放，
峰值 = 任意时刻存活张量字节和的最大值）。

帧数链（全部可从实测推导，非估算）：
    帧率 = 24000 / (10×6×5) = 80 帧/秒（istftnet 上采样+hop 决定，恒定）
    实测校准：61 字 → 63 token → 13.8s 音频 → 1104 帧 → 17.5 帧/token @speed1.0
    T(max) = batch 内最长句的 token 数 × 17.5 ÷ speed_floor

显存构成（FunASR 已独立 asr 池，公式预算见 asr_worker_mb /
npu_ops.prime_asr_memory，随 ASR_MAX_BATCH_SIZE 缩放）：
    常驻：CANN 上下文 450 + KModel 权重 330 + 音色缓存 50 = 830MB
    激活峰值：按算子序列逐段算 shape×4B 取 max（见 peak_activation）
    形状缓存：cache_factor × 激活峰值（池内空闲块库存；因子按探顶校准，
        单一 shape ≈1.3，多 shape 混合 ≈1.5~2.2，默认 1.2）

校准锚点（npu-smi proc-mem 实测，FunASR 与 tts 共驻时代）：
    idle                     1030MB（= 旧常驻 830 + FunASR 200，精确）
    B=1 61字 @1.0            1473MB
    B=8×120字 @1.0           4621MB
    （三锚点均含 FunASR 权重 200 + 首跑缓存 250；--validate 预测时补回
     这 450MB 再比对，±20% 内通过——计算器自证。）

用法：
    python scripts/memcalc.py                       # 默认配置的预算表
    python scripts/memcalc.py --speed-floor 0.5 --max-batch 8 --workers 12
    python scripts/memcalc.py --validate            # 锚点校验
"""
import argparse
import math
import sys

# ---- 模型维度（config.json，逐项注明来源） ----
HIDDEN = 512              # config['hidden_dim']
PLBERT_H, PLBERT_LAYERS, PLBERT_HEADS, PLBERT_FFN = 768, 12, 12, 2048  # config['plbert']
STYLE = 128               # config['style_dim']
MAX_DUR = 50              # config['max_dur']
UPS_RATES = [10, 6]       # config['istftnet']['upsample_rates']
HOP = 5                   # config['istftnet']['gen_istft_hop_size']
UPS_INIT = 512            # config['istftnet']['upsample_initial_channel']
CONTEXT_LEN = 512         # plbert.max_position_embeddings
SR = 24000

FP32 = 4
MB = 2**20
GB = 2**30

# ---- 帧数链（实测校准） ----
FPS = SR / (math.prod(UPS_RATES) * HOP)     # = 80 帧/秒，恒定
CAL_FRAMES_PER_TOKEN = 17.5                  # 61字/13.8s 实测反推
CAL_TOKENS_PER_CHAR = 1.03                   # 61字 → 63 token 实测反推

# ---- 常驻分解（FunASR 已独立 asr 池； tts worker 常驻 = 830MB） ----
CONTEXT_MB = 450
WEIGHTS_TTS_MB = 330
VOICE_CACHE_MB = 50
RESIDENT_MB = CONTEXT_MB + WEIGHTS_TTS_MB + VOICE_CACHE_MB

# 历史锚点校准用：锚点实测于 FunASR 与 tts 共驻时代（权重 200 + 首跑缓存 250）
FUNASR_WEIGHTS_MB = 200
FUNASR_INFER_CACHE_MB = 250
LEGACY_FUNASR_ADD_MB = FUNASR_WEIGHTS_MB + FUNASR_INFER_CACHE_MB

CACHE_FACTOR_DEFAULT = 1.2    # 形状库存因子（探顶校准，见 --cache-factor）
ALLOC_SLACK = 1.10            # 分配器/工作区余量
PHYS_SAFETY = 1.02            # npu-smi 未归属到进程的驱动/系统开销余量
CARD_GB = 64

# ---- asr 池（FunASR MonotonicAligner，与 npu_ops._asr_auto_budget_mb 同公式） ----
# 模型维度取自 speech_timestamp_prediction-v1-16k-offline config.yaml：
# SANMEncoder 30 blocks × hidden 320 × heads 4 × ffn 1280，LFR n=6。
ASR_RESIDENT_MB = 860         # CANN 450（生产实测常量）+ 权重 160（model.pt 实测
                              # 151.1 MiB fp32 上卡）+ 首跑缓存 250（生产锚点）
ASR_LIVE_MB_PER_JOB = 90.0    # @T_max=500 帧（30s 组封顶）：SANM 块峰值 19.5MB×B
                              # （scores+softmax 2×B×4×T²×4B + FCN + qkv/proj，
                              # no_grad 分配器跨 30 blocks 复用取 2 blocks 在途 ×2）= 40，
                              # + CIF cnn_blstm 上采样 3T 序列 11.5 + 保留输出 8
                              # = 60MB×B → ×1.5 保守校准 = 90MB×B，按 T 线性缩放
ASR_MAX_GROUP_SEC = 30.0      # proc_common.ASR_MAX_GROUP_SEC 同源（组时长硬封顶：
                              # g2p._chunk_cap + funasr_ts._split_groups 两层保证）
# 帧率 100/6 ≈ 16.67 帧/秒（fbank 10ms shift ÷ LFR n=6）。注意必须用
# `sec * 100 / 6` 而不是 `sec * (100/6)`：后者 30s 会因浮点误差 ceil 成 501，
# 与 npu_ops._asr_auto_budget_mb 的 500 帧预算漂移。
ASR_CTX_HEADROOM_MB = 256     # npu_ops.prime_asr_memory 的 fraction 公式里
                              # 预占之外的对齐/舍入余量（不是物理 ctx 全量）
ASR_CTX_MB = 450              # asr 进程 CANN 上下文物理值（真机反推：
                              # 进程内存 2226 - 0.98×预算 1782 ≈ 444；取 450）


def asr_worker_mb(max_batch: int, max_group_sec: float = ASR_MAX_GROUP_SEC,
                  alloc_slack: float = ALLOC_SLACK) -> dict:
    """单 asr worker 总显存（MB）：常驻 860 + 激活×2.2（含形状库存 1.2×）。

    与 npu_ops._asr_auto_budget_mb 同 base 公式（双向镜像，改一处必改
    另一处）。最坏形状 = 批内全员顶格 30s 组（padded batch 按最长条对齐）。

    口径注意：ASR_RESIDENT_MB 里的 CANN 450 是预算帽里的保守预留；
    physical_mb 用 1.05×预算 + ASR_CTX_MB=450 算进程上限（再乘 1.02）。
    真机 B4 实测：进程内存 2226MB = 0.98×预算 1782 + 实测 ctx 444；
    所以旧的 0.98×预算+153 会低估约 260MB/worker。"""
    t_max = math.ceil(max_group_sec * 100 / 6)
    live = ASR_LIVE_MB_PER_JOB * (t_max / 500.0) * max(1, int(max_batch))
    total = ASR_RESIDENT_MB + live * 2.2      # live + 形状库存 1.2×
    return {"t_max": t_max, "live_mb": round(live, 1),
            "total_mb": round(total * alloc_slack, 1),
            "total_raw_mb": round(total, 1)}


def physical_mb(budget_mb: float, ctx_mb: float = CONTEXT_MB) -> float:
    """npu-smi 视角的单 worker 物理上限：1.05×budget + CANN 上下文，
    再乘 1.02 覆盖未归属到进程的驱动/系统开销。

    1.05 是 set_per_process_memory_fraction 的上限（进程可长到 1.05×预算），
    0.98 只是启动预占池——拿 0.98 做容量判断会低估（真机 5 tts+4 asr 实测
    62.4GB，旧式 0.98 口径仅 56.7GB）。asr 池传 ASR_CTX_MB（450），tts 池
    用默认 CONTEXT_MB（450）。"""
    return (1.05 * budget_mb + ctx_mb) * PHYS_SAFETY


def pipeline_budget_mb(chunk_chars: int, max_batch: int, speed_floor: float) -> int:
    """运行时实际预算（npu_ops._auto_budget_mb 同式）：live×1.15 非 gen
    段余量后 ×1.1。共卡核算必须用本式——运行时预占的是它，比
    worker_total_mb（解析 base）多 ~12%，低估会让物理核算失真。"""
    w = worker_total_mb(chunk_chars, max_batch, speed_floor, CACHE_FACTOR_DEFAULT)
    return math.ceil(1.1 * (RESIDENT_MB + w["live_mb"] * 1.15 * 2.2))


def mb(nbytes: float) -> float:
    return nbytes / MB


def peak_activation(B: int, L: int, F: int) -> float:
    """单次前向的激活峰值（MB）。B=批宽，L=token 数（含BOS/EOS），F=帧数。"""
    def t(c, t_):
        return B * c * t_ * FP32

    # ① plBERT：12 层顺序执行、层间释放 → 峰值 = 单层峰值 + 残差流。
    # QKV/proj 为 B×L×H，attention scores/softmax 为 B×heads×L×L，
    # FFN 为 B×L×FFN——所有项都必须带 L（旧式漏乘 L，只有默认 gen_s1
    # 主导时才碰巧不影响结果）。
    layer = (3 * PLBERT_H + PLBERT_H
             + 2 * PLBERT_HEADS * L
             + 2 * PLBERT_FFN) * B * L * FP32
    plbert = layer + B * L * PLBERT_H * FP32

    # ② 入口投影 / 文本编码器 / predictor LSTM（门控+双向）
    entry = t(HIDDEN, L)
    predictor = 8 * t(HIDDEN, L)          # LSTM 门控 ×4，双向峰值近似
    dur_head = t(MAX_DUR, L)              # duration_proj [B,L,50]
    aln = t(L, F)                         # 对齐矩阵 [B,L,F]
    en_asr = t(128, F) + t(HIDDEN, F)     # en / asr_t
    f0n = 2 * t(128, F // 2) + t(1, F)    # F0/N 双路 + 预测

    # ③ 解码器：cat[B,1090,F/2] + 块输出[B,1024,F/2] + resblock 内部 3×通道
    dec_cat = t(1090, F // 2)
    dec_blk = t(1024, F // 2)             # decode 块输出 = upsample_initial_channel
    dec_block_peak = dec_cat + 4 * dec_blk            # 3 个块顺序、峰值相同
    dec_4th = dec_cat + t(UPS_INIT, F) + 3 * t(UPS_INIT, F // 2)  # ×2 上采样回 F

    # ④ istftnet Generator（绝对大头）：两段上采样，resblock 存活 = 3×通道
    # （第4解码块输出 [B,512,F]，generator 输入即 F 帧）
    gen_s0 = 3 * t(UPS_INIT // 2, UPS_RATES[0] * F)
    gen_s1 = 3 * t(UPS_INIT // 4, math.prod(UPS_RATES) * F)
    gen_tail = t(22, math.prod(UPS_RATES) * F) + t(1, 300 * F)    # conv_post + 波形

    return mb(max(plbert, entry, predictor, aln + en_asr + f0n + dur_head,
                  dec_block_peak, dec_4th, gen_s0, gen_s1, gen_tail))


def worker_total_mb(chunk_chars: int, max_batch: int, speed_floor: float,
                    cache_factor: float) -> dict:
    """单 tts worker 总显存（MB）：常驻 830 + 激活峰值 + 形状库存。"""
    L = math.ceil(chunk_chars * CAL_TOKENS_PER_CHAR) + 2
    frames_per_sent = max(1, round(L * CAL_FRAMES_PER_TOKEN))       # @speed1.0
    F = math.ceil(frames_per_sent / speed_floor)                    # 最长句帧数（批内 max）
    live = peak_activation(max_batch, min(L, CONTEXT_LEN), F)
    shape_cache = cache_factor * live                                # 池内形状库存
    total = RESIDENT_MB + live + shape_cache
    return {"L": L, "F": F, "live_mb": round(live, 1),
            "total_mb": round(total * ALLOC_SLACK, 1),
            "total_raw_mb": round(total, 1)}


def budget_table(args) -> None:
    card_mb = (args.card_gb if args.card_gb else CARD_GB) * 1024
    w = worker_total_mb(args.chunk_chars, args.max_batch, args.speed_floor, args.cache_factor)
    budget_mb = math.ceil(w["total_mb"])                # total 已含 ALLOC_SLACK
    # 上限/建议按运行时预算（进程实际预占的是它，比 base 多 ~12%）——
    # 与下方共居核算、max_workers 同口径；base 只作为分解展示。
    rt_budget_mb = pipeline_budget_mb(args.chunk_chars, args.max_batch,
                                      args.speed_floor)
    max_workers = int(card_mb // rt_budget_mb)
    print(f"── 配置：chunk={args.chunk_chars}字  MAX_BATCH={args.max_batch}  "
          f"speed_floor={args.speed_floor}  cache_factor={args.cache_factor}  card={args.card_gb or CARD_GB}GB")
    print(f"   L={w['L']} token  F={w['F']} 帧（单批音频 ≈ {w['F'] / FPS:.1f}s）  "
          f"(帧率 {FPS:.0f}fps，{CAL_FRAMES_PER_TOKEN}帧/token@1.0 ÷ {args.speed_floor})")
    print(f"   激活峰值        : {w['live_mb']} MB")
    print(f"   worker 总显存   : {w['total_mb']} MB（常驻 {RESIDENT_MB} = CANN 450 + "
          f"KModel 330 + 音色缓存 50；FunASR 已独立 asr 池不占此预算；"
          f"含激活 + 形状库存 ×{args.cache_factor} 与分配器余量 ×{ALLOC_SLACK}）")
    print(f"   运行时预算（实际预占）= {rt_budget_mb} MB / {rt_budget_mb / 1024:.2f}GB"
          f"（与 npu_ops._auto_budget_mb 同式，含 1.15 非 gen 段余量）")
    print(f"   ➜ 建议显存预算 = {rt_budget_mb / 1024:.2f}GB（按实际预占口径，"
          f"base {budget_mb}MB 仅作分解参考）")
    # asr 池（同公式族，随 ASR_MAX_BATCH_SIZE 缩放）
    a = asr_worker_mb(args.asr_max_batch)
    asr_budget_mb = math.ceil(a["total_mb"])
    print(f"── asr 池：ASR_MAX_BATCH={args.asr_max_batch}（组≤{ASR_MAX_GROUP_SEC:.0f}s → "
          f"{a['t_max']} enc 帧）激活 {a['live_mb']} MB → 预算 {asr_budget_mb} MB"
          f"（npu-smi ≈ {physical_mb(asr_budget_mb, ASR_CTX_MB) / 1024:.2f}GB 含上下文）")
    if args.workers or args.asr_workers:
        tts_n = args.workers or 0
        asr_n = args.asr_workers or 0
        rt_budget = pipeline_budget_mb(args.chunk_chars, args.max_batch, args.speed_floor)
        need = tts_n * rt_budget + asr_n * asr_budget_mb
        phys = (tts_n * physical_mb(rt_budget)
                + asr_n * physical_mb(asr_budget_mb, ASR_CTX_MB))
        ok = "✓" if phys <= card_mb else "✗ 超出整卡！"
        print(f"   共卡核算（{tts_n} tts × {rt_budget}MB + {asr_n} asr × {asr_budget_mb}MB,"
              f" 运行时预算口径）: 预算 {need / 1024:.1f}GB / 物理(npu-smi) {phys / 1024:.1f}GB"
              f" / {card_mb // 1024}GB {ok}")
        print(f"   吞吐单位（workers×batch）: tts {tts_n * args.max_batch}"
              f" + asr {asr_n * args.asr_max_batch}")
    print(f"   本卡可容纳 tts worker 上限（该配置，不含 asr 池）: {max_workers}")
    print("   注：MAX_INPUT_CHARS 不影响显存峰值（切句后单句形状固定），"
          "只影响每请求句数（429/队列压力）。")


def profile_table(card_gb: float) -> None:
    """64GB 卡吞吐档配置表：两池共居核算（物理占用视角，预算随 B 公式缩放）。

    吞吐单位 = workers×batch（线性设计模型——上线以 bench 实测验证，
    见 README 调参段）。asr 按 ~0.4× tts 吞吐单位配置（SANM encoder
    算术强度远低于 istftnet 生成器，单条算量约为 tts 的 1/3；blstm
    补丁把混合长度批的 predictor 段逐条化，留 0.1 折损余量）。"""
    print(f"── 64GB 卡吞吐档参考（物理占用视角，含 asr 池）──")
    profiles = [
        ("旧基线（v6 前默认）", 6, 4, 3, 1),
        ("现行默认",            4, 8, 4, 8),
        ("均衡档",              6, 6, 4, 4),
        ("4×B12+2×B10（新核算超卡）", 4, 12, 2, 10),  # 48u；~67GB >64
        ("4×B12+2×B8（新核算超卡）", 4, 12, 2, 8),
    ]
    for name, tn, tb, an, ab in profiles:
        tb_mb = pipeline_budget_mb(120, tb, 0.5)          # 运行时预算口径（+12% 非 gen 余量）
        ab_mb = math.ceil(asr_worker_mb(ab)["total_mb"])  # asr 无额外 1.15 项，公式即运行时
        phys = (tn * physical_mb(tb_mb)
                + an * physical_mb(ab_mb, ASR_CTX_MB))
        ok = "✓" if phys <= card_gb * 1024 else "✗"
        print(f"   {name}: tts {tn}×B{tb} + asr {an}×B{ab}"
              f" → tts {tn * tb}u（asr {an * ab}u）"
              f" 物理 {phys / 1024:.1f}GB / {card_gb:.0f}GB {ok}")


def validate() -> int:
    """三锚点校验：计算 vs 实测，±20% 内通过。

    锚点实测于 FunASR 共驻时代——预测统一补回 LEGACY_FUNASR_ADD_MB 再比对。"""
    anchors = [
        ("idle（常驻分解）", 1030),
        ("B=1 61字 @1.0",    1473),
        ("B=8×120字 @1.0",   4621),
    ]
    ok = True
    for name, measured in anchors:
        if name.startswith("idle"):
            pred = RESIDENT_MB + FUNASR_WEIGHTS_MB  # idle 无 FunASR 首跑缓存
        elif name.startswith("B=1"):
            pred = worker_total_mb(61, 1, 1.0, 1.3)["total_raw_mb"] + LEGACY_FUNASR_ADD_MB
        else:
            pred = worker_total_mb(120, 8, 1.0, 1.2)["total_raw_mb"] + LEGACY_FUNASR_ADD_MB
        dev = (pred - measured) / measured * 100
        passed = abs(dev) <= 20
        ok = ok and passed
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: 计算 {pred:.0f}MB vs 实测 {measured}MB"
              f"（偏差 {dev:+.1f}%）")
    print("VALIDATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _int_min(minimum: int, name: str):
    def parse(s):
        try:
            v = int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} 必须是整数: {s!r}")
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{name} 不能小于 {minimum}: {v}")
        return v
    return parse


def _float_min(minimum: float, name: str):
    def parse(s):
        try:
            v = float(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} 必须是数值: {s!r}")
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{name} 不能小于 {minimum}: {v}")
        return v
    return parse


def main() -> int:
    ap = argparse.ArgumentParser(description="Kokoro 单 worker 显存精确计算器")
    ap.add_argument("--chunk-chars", type=_int_min(8, "chunk-chars"),
                    default=120, help="TTS_CHUNK_CHARS（>=8）")
    ap.add_argument("--max-batch", type=_int_min(1, "max-batch"), default=8,
                    help="TTS_MAX_BATCH_SIZE（与镜像默认一致，>=1）")
    ap.add_argument("--speed-floor", type=_float_min(0.1, "speed-floor"),
                    default=0.5, help="TTS_SPEED_FLOOR（>=0.1，避免除零）")
    ap.add_argument("--workers", type=_int_min(0, "workers"), default=0,
                    help="核算 N 个 tts worker 的总占用")
    ap.add_argument("--asr-max-batch", type=_int_min(1, "asr-max-batch"),
                    default=8, help="ASR_MAX_BATCH_SIZE（与镜像默认一致，>=1）")
    ap.add_argument("--asr-workers", type=_int_min(0, "asr-workers"), default=0,
                    help="核算 N 个 asr worker 的总占用")
    ap.add_argument("--cache-factor", type=_float_min(0.1, "cache-factor"),
                    default=CACHE_FACTOR_DEFAULT,
                    help="形状库存因子：单一 shape 1.3、多 shape 混合 1.5~2.2")
    ap.add_argument("--card-gb", type=_float_min(0.001, "card-gb"), default=CARD_GB)
    ap.add_argument("--profiles", action="store_true",
                    help="打印 64GB 卡吞吐档配置表（两池共居核算）")
    ap.add_argument("--validate", action="store_true", help="跑三个实测锚点校验")
    args = ap.parse_args()

    if args.validate:
        return validate()
    if args.profiles:
        profile_table(args.card_gb if args.card_gb else CARD_GB)
        return 0
    budget_table(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

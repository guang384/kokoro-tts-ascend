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

import proc_common as _pc
import protocol

log = logging.getLogger("kokoro.pipeline")
_last_budget = None   # (预算MB, L, 单批音频秒)：_prime_memory 写入，/metrics 的 npu_budget 读取
_last_asr_budget = None  # (预算MB, T_max 帧, 单条音频秒上限)：prime_asr_memory 成功时写入
_budget_error = None  # 预算/预占失败摘要（失败不阻塞服务，但必须可观测）
_asr_budget_error = None  # asr 池预算失败摘要（同上；两个池各自独立可观测）

# 每 worker 的 CANN 上下文物理占用估算（npu-smi 视角）。真机反推
# asr 进程内存 - 0.98×预算 ≈ 444MB，与 tts 同源，统一取 450MB。
# metrics 的 estimated_mb = torch reserved + 本值；
# planned_mb = 0.98×预算 + 本值（预占池）；limit_mb = 1.05×预算 + 本值
# （set_per_process_memory_fraction 允许的进程上限）。
TTS_CONTEXT_MB = 450
ASR_CONTEXT_MB = 450
CAPACITY_SAFETY = 1.02   # npu-smi 实际 HBM 还含驱动/未归属开销，容量判断留 2%
_DEVICE_TOTAL_MB = {}   # dev → 整卡显存 MB（进程内缓存：get_device_properties 不必每批查）

_model_id = os.getenv("KOKORO_MODEL_ID", "hexgrad/Kokoro-82M-v1.1-zh")
_default_voice = os.getenv("KOKORO_VOICE", "zf_001")

_init_lock = threading.Lock()
_g2p: "G2PEngine | None" = None
_compute: "TTSCompute | None" = None


_SMI_CARD_RE = re.compile(r"(?:Ascend|\d{3}[A-Za-z]\d?)")

# npu-smi 进程内存采样：npu-smi 是独立子进程查询，主线程调用；缓存避免
# 每个前向/每批都 fork。失败负缓存更长，防止驱动异常时反复超时。
_SMI_PROC_CACHE = {"ts": 0.0, "pid": None, "dev": None, "mb": None}
_SMI_PROC_TTL_S = 30.0
_SMI_PROC_NEG_TTL_S = 60.0
_SMI_PROC_TIMEOUT_S = 2.0


def _smi_card_ids() -> list:
    """从 npu-smi 解析容器内可见的 NPU 芯片 ID（驱动层视角）。

    vNPU 池化场景（cnos-vpool 等）：CANN Runtime 的设备枚举可能缺失或
    按物理槽位编号，torch.npu.device_count() 与真实可用标识不一致——
    而 npu-smi 走 davinci_manager 驱动接口，CANN 看不见卡时照样能报
    真实卡号（生产自证：torch probe 全失败时 npu-smi 显示 NPU 7 健康）。
    把驱动层卡号喂进探测候选，不猜。解析失败静默回退传统探测（增强，
    非依赖）；Windows 开发机无 npu-smi → 返回 []。"""
    try:
        import subprocess

        out = subprocess.run(["npu-smi", "info"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001  无二进制/超时/非 POSIX
        return []
    ids = []
    for line in out.splitlines():
        if "Process id" in line:
            break  # 第二表格（进程表）首列是 PID——停止解析防误读
        parts = [p.strip() for p in line.strip("| ").split("|")]
        if not parts or not parts[0]:
            continue
        # 首格含 "卡号 型号" 两个 token（真实输出 NPU+Name 同格）：
        # '7  910B2' → 7；进程表 '7  0' 的型号 token 不匹配 → 不误读
        head = parts[0].split()
        if (len(head) >= 2 and head[0].isdigit()
                and _SMI_CARD_RE.search(head[1])):
            ids.append(int(head[0]))
    return ids


def _pid_candidates() -> set:
    """当前进程在各 PID namespace 里的候选 PID（npu-smi 可能报宿主 PID）。

    /proc/self/status 的 NSpid 行：最左为挂载 procfs 的 namespace，向右为
    嵌套内层。容器内通常含容器 PID；hostPID/特权容器可能同时含宿主 PID。
    读不到（Windows/隐藏 proc）时退化为 os.getpid()。"""
    pids = {os.getpid()}
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("NSpid:"):
                    pids.update(int(tok) for tok in line.split()[1:]
                                if tok.isdigit())
                    break
    except OSError:
        pass
    return pids


def _parse_smi_process_mem(out: str, pid_candidates) -> float | None:
    """解析 `npu-smi info` 的进程表，返回本进程 Process memory(MB) 之和。

    表头可能含 `Process id` / `Process id in container` / `Process memory(MB)`：
    优先用容器 PID 列匹配 os.getpid()，否则用 Process id 匹配 PID 候选集
    （NSpid 宿主/容器 PID）。同 PID 多行（多芯片）求和；无匹配/无内存列
    返回 None——不能填 0（0 会把"查不到"伪装成"不占显存"）。"""
    if not out:
        return None
    pids = {int(p) for p in pid_candidates}
    lines = out.splitlines()
    header = None
    for line in lines:
        if "Process id" in line and "Process memory" in line:
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            break
    if header is None:
        return None

    def _idx(pred):
        for i, c in enumerate(header):
            if pred(c.lower()):
                return i
        return None

    i_container = _idx(lambda c: "process id in container" in c)
    i_pid = _idx(lambda c: "process id" in c and "container" not in c)
    i_mem = _idx(lambda c: "process memory" in c)
    if i_mem is None or (i_pid is None and i_container is None):
        return None

    total = 0.0
    matched = False
    seen = set()
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) <= i_mem:
            continue
        hit = False
        if (i_container is not None and i_container < len(cells)
                and cells[i_container].isdigit()):
            hit = int(cells[i_container]) in pids
        if (not hit and i_pid is not None and i_pid < len(cells)
                and cells[i_pid].isdigit()):
            hit = int(cells[i_pid]) in pids
        if not hit:
            continue
        # 同一进程可能在多张芯片各占一行；按 (npu/chip 首格, pid, 内存) 去重
        key = (cells[0] if cells else "",
               cells[i_pid] if i_pid is not None and i_pid < len(cells) else "",
               cells[i_mem])
        if key in seen:
            continue
        seen.add(key)
        m = re.search(r"(\d+(?:\.\d+)?)", cells[i_mem])
        if m:
            total += float(m.group(1))
            matched = True
    return round(total, 1) if matched else None


def _parse_smi_proc_mem_short(out: str, pid_candidates) -> float | None:
    """解析 `npu-smi info -t proc-mem` 的短格式：

        Process id:38329 Process name:... Process memory(MB):96

    作为完整 `npu-smi info` 进程表截断（每芯片最多 8 条）时的兜底。"""
    if not out:
        return None
    pids = {int(p) for p in pid_candidates}
    cur = None
    total = 0.0
    matched = False
    for line in out.splitlines():
        for m in re.finditer(r"Process id(?: in container)?\s*[:=]\s*(\d+)",
                             line):
            cur = int(m.group(1))
        m = re.search(r"Process memory(?:\(MB\))?\s*[:=]\s*([\d.]+)", line)
        if m:
            if cur in pids:
                total += float(m.group(1))
                matched = True
            cur = None
    return round(total, 1) if matched else None


def _run_smi(cmd: list) -> str:
    """执行 npu-smi；缺失/超时/nonzero/非 POSIX 一律返回空串（观测降级）。"""
    try:
        import subprocess

        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=_SMI_PROC_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001  无二进制/超时/权限
        log.debug("npu-smi call failed cmd=%s: %s", cmd, str(exc)[:120])
        return ""
    if r.returncode != 0:
        log.debug("npu-smi rc=%s cmd=%s stderr=%s", r.returncode, cmd,
                  (r.stderr or "")[:120])
        return ""
    return r.stdout or ""


def smi_process_mb(device: str = "npu:0") -> float | None:
    """当前 worker 的 npu-smi 实测进程内存（MB）；未知返回 None。

    先读完整进程表；本 PID 未出现（每芯片最多 8 条，5 tts + 4 asr 时可能
    有一个被截断）再试 `-t proc-mem -i <idx>`。30s 正缓存 / 60s 负缓存，
    避免每个前向/每批 fork npu-smi。仅在 worker 主线程调用（npu-smi 是
    子进程，但采样点跟随 torch 内存采样，保持单一调用方）。"""
    dev = str(device or "")
    if not dev.startswith("npu"):
        return None
    idx = int(dev.split(":")[-1]) if ":" in dev else 0
    pid = os.getpid()
    now = time.monotonic()
    c = _SMI_PROC_CACHE
    if c["pid"] == pid and c["dev"] == dev:
        ttl = _SMI_PROC_TTL_S if c["mb"] is not None else _SMI_PROC_NEG_TTL_S
        if now - c["ts"] < ttl:
            return c["mb"]

    pids = _pid_candidates()
    mb = _parse_smi_process_mem(_run_smi(["npu-smi", "info"]), pids)
    if mb is None:
        mb = _parse_smi_proc_mem_short(
            _run_smi(["npu-smi", "info", "-t", "proc-mem", "-i", str(idx)]),
            pids)
    c.update(ts=now, pid=pid, dev=dev, mb=mb)
    return mb


def _probe_npu(torch, idx: int) -> bool:
    """在 npu:{idx} 上做一次最小分配+同步：验证卡既存在又能真正出算子。"""
    try:
        dev = torch.device(f"npu:{idx}")
        torch.zeros(1, device=dev)
        torch.npu.synchronize(dev)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("npu:%d probe failed (%s), skipping", idx, str(exc)[:160])
        return False


_SAMPLE_RATE = _pc.env_int("KOKORO_SAMPLE_RATE", 24000, minimum=8000)  # 声明模型真实采样率（非可调旋钮：改它会连锁破坏 fade 窗口/预算换算/WAV 头）
if _SAMPLE_RATE != 24000:
    # batched_model/memcalc/ASR 时间戳都按 24k 编译；允许改 env 只会产出
    # 错误 WAV 头/时长/时间戳偏移。启动即拒绝，别把配置错误拖到首个请求。
    raise RuntimeError(
        f"KOKORO_SAMPLE_RATE={_SAMPLE_RATE} 不支持：Kokoro-82M-v1.1-zh 与 "
        "batched_model/memcalc 固定为 24000Hz；请删除该 env 或设为 24000")
_FRAMES_PER_SEC = _SAMPLE_RATE / (10 * 6 * 5)  # 80：istftnet 上采样 10×6 × hop5 → 每秒 80 帧


def _auto_budget_mb(max_batch: int, chunk_chars: int, speed_floor: float):
    """tts worker 显存预算自动推导。

    与 scripts/memcalc.py 同 base 公式（memcalc 逐算子解析激活；本处用
    gen_s1 主导项 ×1.15 覆盖其余段的简化式），本式略保守 ~12%（多一份
    非 gen 段余量）；锚点校准以 memcalc --validate 为准。

    L = ceil(chunk_chars×1.03)+2（token，含 BOS/EOS；1.03 为 61 字实测校准）
    F = ceil(L × 17.5 / speed_floor)（帧；17.5 帧/token 为 61 字实测校准，
        帧率恒定 80fps = 24000/(10×6×5)）
    live ≈ 1.15 × gen_s1：生成器 stage-1 resblock 峰值（3×B×128×60F×4B）为
        主导项，其余段（stage-0/aln/dec/plBERT）≤ 0.15×主导项
    常驻 830 = CANN 上下文 450 + KModel 权重 330 + 杂项 50
      （FunASR 已独立 asr 池，不再计入 tts 预算——asr 池走 prime_asr_memory）
    total = 常驻 830 + live×2.2（live + 形状库存 1.2×）
    预算 = total × 1.1（分配器/工作区余量）

    返回 (预算 MB, L token, 单批音频秒上限)——对外一律用秒展示，帧不直观。
    """
    L = math.ceil(chunk_chars * 1.03) + 2
    F = math.ceil(L * 17.5 / max(speed_floor, 0.1))
    live_gen_mb = 3 * max_batch * 128 * 60 * F * 4 / 2**20
    live_mb = live_gen_mb * 1.15
    total_mb = 830 + live_mb * 2.2
    return math.ceil(total_mb * 1.1), L, round(F / _FRAMES_PER_SEC, 1)


def _asr_auto_budget_mb(max_batch: int, max_group_sec: float = None):
    """asr worker 显存预算自动推导（FunASR MonotonicAligner，全程 fp32）。

    与 tts _auto_budget_mb 同方法论：常驻分解 + 激活×(1+形状库存1.2) ×
    分配器余量1.1。模型维度取自 speech_timestamp_prediction-v1-16k-
    offline 的 config.yaml：SANMEncoder 30 blocks × hidden 320 × heads 4 ×
    ffn 1280，WavFrontend frame_shift 10ms + LFR n=6 → encoder 帧率
    100/6 ≈ 16.67 帧/秒（16k 解码后）。

    帧数链：T_max = ceil(max_group_sec×100/6)——组由 max_group_sec 硬封顶
    （proc_common.ASR_MAX_GROUP_SEC=30 → 500 帧；封顶由 g2p._chunk_cap
    语速感知切句 + funasr_ts._split_groups 超长 part 硬切两层保证），即
    批内单条音频的 encoder 帧上限（padded batch 按最长条对齐，全员顶格
    即最坏）。
    常驻 860 = CANN 上下文 450（生产实测常量，与 tts 同源）
             + FunASR 权重 160（model.pt 实测 151.1 MiB fp32 原样上卡）
             + 首跑缓存/运行时 workspace 250（生产锚点，memcalc 同值）
    激活 live（fp32、no_grad：30 blocks 顺序执行、分配器跨块复用，取
    2 blocks 在途 ×2；CIF cnn_blstm 上采样 3T 序列 + 保留输出；合计
    60MB×B @T=500，×1.5 保守校准系数 = 90MB×B）：
        单块峰值 ≈ scores+softmax 2×B×4×T²×4B + FCN 3×B×T×1280×4B
                  + qkv/proj 6×B×T×320×4B ≈ 19.5MB×B @T=500 → ×2 = 40MB×B
    按 T 线性缩放（scores 项 ∝T²，短音频只省不多算——偏保守方向）。
    total = 常驻 860 + live×2.2（live + 形状库存 1.2×）
    预算 = total × 1.1

    返回 (预算 MB, T_max 帧, 单条音频秒上限)——对外一律用秒展示。
    """
    sec = float(max_group_sec if max_group_sec is not None
                else _pc.ASR_MAX_GROUP_SEC)
    t_max = math.ceil(sec * 100 / 6)
    live_mb = 90.0 * (t_max / 500.0) * max(1, int(max_batch))
    total_mb = 860 + live_mb * 2.2
    return math.ceil(total_mb * 1.1), t_max, round(sec, 1)


def budget_error() -> str | None:
    """最近一次 NPU 预算设置失败的摘要（无失败为 None）——心跳 stats 上报。"""
    return _budget_error


def asr_budget_error():
    """asr 池预算失败摘要（无失败为 None）。compute 心跳读取上报。"""
    return _asr_budget_error


def asr_budget_meta():
    """最近一次成功的 ASR 预算 (budget_mb, t_max, sec)；未设置过为 None。"""
    return _last_asr_budget


def _prime_memory(device: str) -> None:
    """tts worker：按公式自动推导显存预算：fraction 硬限 + 启动预占。

    预算随 MAX_BATCH / TTS_CHUNK_CHARS / TTS_SPEED_FLOOR 三个业务参数
    自动缩放（B=8/120字/0.5速 → 预算 ~9.5GB，npu-smi 进程上限 ~10.5GB
    含上下文；单卡 worker 数见 scripts/memcalc.py）。失败不阻塞服务，但记入
    _budget_error（心跳可见——共享卡上无预算运行可能挤占他人）。"""
    if not str(device).startswith("npu"):
        return
    global _last_budget, _budget_error
    try:
        import torch

        idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        total = torch.npu.get_device_properties(idx).total_memory
        # 与消费者（tts 认领队列/看门狗）同 minima：原裸 int() 在
        # TTS_MAX_BATCH_SIZE=0 时会算出 ~913MB 预算（实际批为 1）→ 首批真 OOM
        import proc_common as _pc

        max_batch = _pc.env_int("TTS_MAX_BATCH_SIZE", 8, minimum=1)
        chunk_chars = _pc.env_int("TTS_CHUNK_CHARS", 120, minimum=8)
        speed_floor = _pc.env_float("TTS_SPEED_FLOOR", 0.5, minimum=0.1)
        budget_mb, L, sec = _auto_budget_mb(max_batch, chunk_chars, speed_floor)
        budget = budget_mb * 2**20
        # fraction 必须高于预占值：预占恰好等于 allowed 时，torch 内部用浮点
        # 重算 allowed 会舍入出"少一丁点"，严格比较即假 OOM（生产事故：
        # 卡空 60GB 却报 5.11GB 分配失败）。硬上限本身仍生效——超出预算+5%
        # 才会真正拒绝，保护共享卡不被本进程挤占。
        frac = min(budget * 1.05 / total, 1.0)
        torch.npu.set_per_process_memory_fraction(frac, idx)
        # 预占（砍 2% 给分配器对齐/元数据开销）：临时表达式创建后立即无引用
        # 释放，整块进入缓存池供后续切分；npu-smi 从此恒定显示本预算
        torch.empty(int(budget * 0.98), dtype=torch.uint8,
                    device=torch.device(f"npu:{idx}"))
        _last_budget = (budget_mb, L, sec)
        _budget_error = None  # 重试成功必须复位：否则心跳永远假报预算失败
        log.info("NPU budget %.1fGB (auto: B=%d x %d chars/sent, speed>=%.1f,"
                 " L=%d tokens, batch audio<=%.1fs; %.1f%% of card) preallocated",
                 budget_mb / 1024, max_batch, chunk_chars, speed_floor,
                 L, sec, frac * 100)
    except Exception as exc:  # noqa: BLE001  预算/预占失败不阻塞服务
        _budget_error = str(exc)[:160]
        log.warning("NPU budget setup failed (non-blocking, reported via heartbeat): %s", exc)


def prime_asr_memory(device: str) -> None:
    """asr worker：按公式自动推导显存预算（批推峰值形状封顶）。

    预算随 ASR_MAX_BATCH_SIZE 自动缩放（B=8/30s 组 → 预算 ~2.69GB，
    npu-smi 进程上限 ~3.27GB 含上下文（真机反推 ctx ~450MB）；默认 4
    workers 共 ~10.8GB 预算。旧固定 1GB 在批推下余量不足，故公式化）。
    失败仅告警（与 tts 预算同策略：尽力而为，失败可观测）。

    调用时机注意：compute._load_asr 在模型已上卡后才调本函数，CANN
    上下文（npu-smi 进程内存反推 ~450MB）已在卡上——fraction 必须在
    预占值之外再给足对齐/舍入余量，否则预占假 OOM（tts 池生产事故同款：
    allowed 与预占恰好相减出负数时严格比较必失败）。256MB 是 fraction
    的对齐/舍入余量，不是物理 ctx 全量（物理口径见 ASR_CONTEXT_MB）。
    硬上限语义不变：超出预算+ctx 才真正拒绝，保护共享卡不被本进程挤占。"""
    if not str(device).startswith("npu"):
        return
    global _asr_budget_error, _last_asr_budget
    try:
        import torch

        idx = int(str(device).split(":")[-1]) if ":" in str(device) else 0
        total = torch.npu.get_device_properties(idx).total_memory
        # 与 asr_main 认领量同源同 minima：ASR_MAX_BATCH_SIZE=0 时裸 int()
        # 会算出 ~1.16GB 预算（实际批为 1）→ 热身/首批假象
        max_batch = _pc.env_int("ASR_MAX_BATCH_SIZE", 8, minimum=1)
        budget_mb, t_max, sec = _asr_auto_budget_mb(max_batch)
        budget = budget_mb * 2**20
        frac = min((budget * 0.98 + 256 * 2**20) / total, 1.0)
        torch.npu.set_per_process_memory_fraction(frac, idx)
        # 预占（砍 2% 给分配器对齐/元数据开销）：临时表达式创建后立即无
        # 引用释放，整块进入缓存池供后续切分；npu-smi 从此恒定显示本预算
        torch.empty(int(budget * 0.98), dtype=torch.uint8,
                    device=torch.device(f"npu:{idx}"))
        _last_asr_budget = (budget_mb, t_max, sec)
        _asr_budget_error = None  # 重试成功必须复位：否则心跳永远假报预算失败
        log.info("ASR NPU budget %.2fGB (auto: B=%d, group<=%.0fs → %d enc"
                 " frames; %.1f%% of card) preallocated (npu:%d)",
                 budget_mb / 1024, max_batch, sec, t_max, frac * 100, idx)
    except Exception as exc:  # noqa: BLE001  预算/预占失败不阻塞服务
        _asr_budget_error = str(exc)[:160]
        log.warning("ASR NPU budget setup failed (ignored): %s", exc)


def _npu_total_mb() -> float | None:
    """短命子进程查询卡总显存（MB）。

    为什么不在 supervisor 父进程直接 import torch_npu：supervisor 是容器
    PID 1，父进程一旦创建 CANN 上下文就会占用一份设备显存，挤占 worker
    预算；原实现只 `import torch` 不 import torch_npu，`torch.npu` 不存在，
    except 静默返回 None——自检形同虚设。子进程查询完即退出，父进程零
    上下文。失败必须留痕（warning），不能继续假装自检通过了。"""
    import subprocess
    import sys

    code = ("import torch, torch_npu\n"
            "print(torch.npu.get_device_properties(0).total_memory / 2**20)")
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=20)
    except Exception as exc:  # noqa: BLE001  超时/无解释器：自检降级
        log.warning("NPU capacity probe failed to start (ignored): %s", exc)
        return None
    if r.returncode != 0:
        log.warning("NPU capacity probe failed (rc=%s, ignored): %s",
                    r.returncode, (r.stderr or "").strip()[:200])
        return None
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        log.warning("NPU capacity probe unparsable output (ignored): %r",
                    (r.stdout or "")[:200])
        return None


def estimate_worker_mem(allocated_mb: float, reserved_mb: float,
                        total_mb: float, budget_mb: float | None,
                        context_mb: float,
                        process_mb: float | None = None) -> dict:
    """单 worker NPU 内存估算（纯函数，便于测试/复用）。

    - estimated_mb = torch reserved + CANN 上下文估算
    - planned_mb   = 0.98×budget + CANN 上下文（预占池口径）
    - limit_mb     = 1.05×budget + CANN 上下文（fraction 允许的进程上限；
      容量判断用这个，不是 0.98 预占池）
    - process_mb   = npu-smi 实测进程内存（可选；实际口径）
    - delta_mb     = process_mb - planned_mb（正=公式低估，负=公式高估）
    - context_actual_mb = max(0, process_mb - reserved_mb)（实测非 torch 开销）
    - budget_pct   = reserved / budget（预占池使用率）
    - card_pct     = estimated / 整卡（该 worker 占整卡比例）
    """
    estimated = max(0.0, float(reserved_mb)) + float(context_mb or 0.0)
    out = {
        "allocated_mb": round(float(allocated_mb), 1),
        "reserved_mb": round(float(reserved_mb), 1),
        "estimated_mb": round(estimated, 1),
        "context_mb": round(float(context_mb or 0.0), 1),
        "card_total_mb": round(float(total_mb), 1),
        "card_pct": (round(estimated / total_mb * 100, 1)
                     if total_mb and total_mb > 0 else None),
    }
    if process_mb is not None:
        pm = float(process_mb)
        out["process_mb"] = round(pm, 1)
        out["context_actual_mb"] = round(max(0.0, pm - float(reserved_mb)), 1)
    if budget_mb:
        planned = 0.98 * float(budget_mb) + float(context_mb or 0.0)
        limit = 1.05 * float(budget_mb) + float(context_mb or 0.0)
        out["budget_mb"] = round(float(budget_mb), 1)
        out["planned_mb"] = round(planned, 1)
        out["limit_mb"] = round(limit, 1)
        out["budget_pct"] = (round(float(reserved_mb) / float(budget_mb) * 100, 1)
                             if budget_mb > 0 else None)
        if process_mb is not None:
            out["delta_mb"] = round(float(process_mb) - planned, 1)
    return out


def npu_mem_estimate(device: str, budget_mb: float | None = None,
                     context_mb: float = 0.0) -> dict | None:
    """读取 torch 设备计数器 → estimate_worker_mem。

    仅 npu/cuda；cpu/API 失败返回 None。**必须在 worker 主线程调用**：
    心跳线程碰坏态 NPU 会 SIGBUS（生产事故 20261109）。"""
    dev = str(device or "")
    try:
        if dev.startswith("npu"):
            import torch

            idx = int(dev.split(":")[-1]) if ":" in dev else 0
            total_mb = _DEVICE_TOTAL_MB.get(dev)
            if total_mb is None:
                total_mb = (torch.npu.get_device_properties(idx).total_memory
                            / 2**20)
                _DEVICE_TOTAL_MB[dev] = total_mb
            allocated_mb = torch.npu.memory_allocated(idx) / 2**20
            reserved_mb = torch.npu.memory_reserved(idx) / 2**20
        elif dev.startswith("cuda"):
            import torch

            idx = int(dev.split(":")[-1]) if ":" in dev else 0
            total_mb = _DEVICE_TOTAL_MB.get(dev)
            if total_mb is None:
                total_mb = (torch.cuda.get_device_properties(idx).total_memory
                            / 2**20)
                _DEVICE_TOTAL_MB[dev] = total_mb
            allocated_mb = torch.cuda.memory_allocated(idx) / 2**20
            reserved_mb = torch.cuda.memory_reserved(idx) / 2**20
        else:
            return None
        process_mb = None
        if dev.startswith("npu"):
            try:
                process_mb = smi_process_mb(dev)
            except Exception:  # noqa: BLE001  观测失败不影响 torch 指标
                process_mb = None
        return estimate_worker_mem(allocated_mb, reserved_mb, total_mb,
                                   budget_mb, context_mb, process_mb)
    except Exception as exc:  # noqa: BLE001  观测失败不影响服务
        log.debug("NPU mem estimate failed device=%s: %s", dev, str(exc)[:120])
        return None


def capacity_check(tts_n: int, asr_n: int):
    """启动容量自检（best-effort，仅 npu）：返回 (卡容量MB, 预计物理MB, 是否>95%)。

    口径与 scripts/memcalc.py 一致：1.05×预算 + ctx（tts/asr 均 450），
    再乘 CAPACITY_SAFETY=1.02 覆盖 npu-smi 未归属开销。
    共享卡/多副本/ vNPU 场景真实可用显存由编排层决定——这里只告警不拒绝
    启动，避免把整卡容量当成进程可用上限而误杀合法部署。
    非 npu/驱动不可用/字段漂移返回 None（但会 warning 说明原因）。"""
    total_mb = _npu_total_mb()
    if total_mb is None:
        return None
    try:
        import proc_common as _pc

        tts_budget = _auto_budget_mb(
            _pc.env_int("TTS_MAX_BATCH_SIZE", 8, minimum=1),
            _pc.env_int("TTS_CHUNK_CHARS", 120, minimum=8),
            _pc.env_float("TTS_SPEED_FLOOR", 0.5, minimum=0.1))[0]
        asr_budget = _asr_auto_budget_mb(
            _pc.env_int("ASR_MAX_BATCH_SIZE", 8, minimum=1))[0]
        # 进程上限 = fraction 的 1.05×预算 + CANN 上下文；再留 2% 给
        # npu-smi 里未归属到进程的驱动/系统开销。0.98 只是预占池，
        # 用它做容量判断会低估（真机 5 tts+4 asr 实测 62.4GB，旧式 56.7GB）。
        phys_mb = (max(0, int(tts_n))
                   * (1.05 * tts_budget + TTS_CONTEXT_MB)
                   + max(0, int(asr_n))
                   * (1.05 * asr_budget + ASR_CONTEXT_MB))
        phys_mb *= CAPACITY_SAFETY
        return round(total_mb, 1), round(phys_mb, 1), phys_mb > 0.95 * total_mb
    except Exception as exc:  # noqa: BLE001
        log.warning("capacity check failed (ignored): %s", exc)
        return None


def _pick_npu_device() -> str:
    """自动发现容器内可用 NPU，兜住三类历史故障：

    - ASCEND_DEVICE_ID 填了宿主机物理卡号、但容器内只挂了一张卡
      （ERR00100: Invalid device ID，手动 --device 挂载在容器内从 0 重编号）
      → 告警后自动探测全部可见卡；
    - 指定卡存在但已被挂死（507014 aicore timeout）→ 实探失败自动换下一张，
      一张都不可用则回落 CPU，服务不再带着坏卡上线；
    - vNPU 切分（HAMi 等）后 CANN 枚举缺失：device_count() 抛异常或返回 0
      而驱动层看得见卡 → 候选是无条件多层并集，绝不在枚举缺失时提前放弃
      （生产实测：count=0 但 npu-smi 显示卡 7 健康）。
    """
    import torch

    try:
        count = int(torch.npu.device_count())
    except Exception as exc:  # noqa: BLE001  vNPU 切分常态：CANN 枚举整体缺失
        log.warning("torch.npu.device_count() failed (%s)——CANN 枚举缺失"
                    "（vNPU 切分常态），跳过枚举层按显式/驱动层候选实探",
                    str(exc)[:120])
        count = 0
    if count < 0:
        count = 0

    raw = os.getenv("ASCEND_DEVICE_ID", "").strip()
    order: List[int] = []
    seen = set()

    def _add(i: int) -> None:
        if 0 <= i < 256 and i not in seen:  # 上界防御：坏 env 不制造无限候选
            seen.add(i)
            order.append(i)

    # 1) 显式 ASCEND_DEVICE_ID：**不因越界丢弃**——vNPU 池化下 torch 枚举数
    #    与 CANN 设备标识常不一致（count=1 但真标识是物理卡 7），越界是常态
    if raw.isdigit():
        _add(int(raw))
    # 2) torch 枚举（传统路径；pid 取模打散起点——多卡容器内多 worker 自动
    #    分摊到不同卡）。探测通过（zeros+synchronize 真执行）即为真可用
    #    设备，先行不损任何场景。count<=0（vNPU 切分 CANN 枚举缺失）时本层
    #    缺席，但 3)/4) 层照常补位
    if count > 0:
        start = os.getpid() % count
        for k in range(count):
            _add((start + k) % count)
    # 3) 编排层注入的可见卡号：HAMi 注 ASCEND_RT_VISIBLE_DEVICES、
    #    ascend-docker-runtime 注 ASCEND_VISIBLE_DEVICES（逗号分隔卡号）——
    #    这是"分给本容器的精确集合"，枚举缺失时的权威真值
    for var in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        for tok in os.getenv(var, "").split(","):
            tok = tok.strip()
            if tok.isdigit():
                _add(int(tok))
    # 4) npu-smi 驱动层卡号兜底：vNPU 故障模式下 torch 枚举可能全失败
    #    （生产自证：count=1 但 probe 全败，npu-smi 显示 NPU 7 健康）——
    #    此时驱动层卡号是唯一真值来源。放在枚举之后：smi 报的是物理卡号
    #    全集，排前面会让多卡容器的 pid 打散失效（全部 worker 挤最低卡）
    for i in _smi_card_ids():
        _add(i)
    if raw.isdigit() and int(raw) >= count:
        log.info("ASCEND_DEVICE_ID=%s exceeds torch count(%d)——vNPU 标识体系"
                 "与枚举数不一致属常态，按显式值优先实探", raw, count)
    if count == 0 and order:
        log.warning("CANN 枚举缺失/为 0，改按驱动层候选实探: %s", order)

    for idx in order:
        if _probe_npu(torch, idx):
            return f"npu:{idx}"
    log.warning("NPU 自动发现失败：device_count=%d 显式=%r 候选=%s 全部实探"
                "失败——查 npu-smi info 与容器设备挂载", count, raw, order)
    return ""


def _resolve_device() -> str:
    """解析推理设备：DEVICE 显式指定 > 实探可用 NPU（严格）> 本地 CPU 调试。

    严格 fail-fast：torch_npu 可导入（生产镜像）但无可用卡 → 抛错，由
    worker 的重试链与编排层（K8s liveness → 容器重启）兜底——绝不静默
    降级 CPU 把"卡坏了"伪装成健康。本地开发机（无 torch_npu 模块）回落
    CPU 以便跑离线测试；显式 `DEVICE=cpu` 保留人工逃生口。
    """
    explicit = os.getenv("DEVICE", "").strip()
    if explicit:
        return explicit
    try:
        # import torch_npu 隐含校验 torch 可用并注册 torch.npu 后端
        # （importlib 形态：语义相同且静态检查不误报"导入了未用"）
        importlib.import_module("torch_npu")
    except ImportError as exc:
        log.warning("torch_npu not installed (%s), fall back to CPU (dev mode)", exc)
        return "cpu"
    dev = _pick_npu_device()
    if not dev:
        raise RuntimeError(
            "torch_npu 已加载但没有任何可用 NPU 卡——拒绝静默降级 CPU。"
            "检查 npu-smi / 驱动 / ASCEND_DEVICE_ID；确需 CPU 请显式 DEVICE=cpu")
    return dev


def _resolve_disable_complex(device: str) -> bool:
    """NPU 用纯 conv 版 STFT（CustomSTFT）规避复数算子；auto 档 NPU→开。"""
    mode = os.getenv("KOKORO_DISABLE_COMPLEX", "auto").strip().lower()
    if mode in ("1", "true"):
        return True
    if mode in ("0", "false"):
        return False
    return device.startswith("npu")


def _torch_import_ok() -> bool:
    """子进程探测 `import torch` 是否安全。spacy→thinc 会无条件 import
    torch（thinc/compat.py 的 try 拦不住段错误）——QEMU/坏 torch 环境
    进程内 import 直接 SIGSEGV，只能子进程探测。只在首个英文词时跑一次。"""
    import subprocess
    import sys

    try:
        r = subprocess.run([sys.executable, "-c", "import torch"],
                           capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:  # noqa: BLE001  超时/无法启动：按不可用处理
        return False


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
                if _torch_import_ok():
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

        _prime_memory(device)  # 预算硬限+预占先行：连加载期的分配也在预算内
        disable_complex = _resolve_disable_complex(device)
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
                _compute = TTSCompute(_resolve_device())
    return _compute

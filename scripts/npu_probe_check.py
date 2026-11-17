#!/usr/bin/env python3
"""上传镜像前的 NPU 环境预检（纯标准库，宿主机/容器内都可直接跑）。

只回答一个问题：这台机器上「发现逻辑 + 显存预算 + 基础算子」能不能
顺利走通。所有检查只读/极小分配，不改任何配置。

用法（无需虚拟环境、无需 torch）：
    python3 npu_probe_check.py                 # stdlib 档
    python3 npu_probe_check.py --deep          # 额外做 torch_npu 实探
                                               # （需在镜像/含 torch_npu 的环境内）
退出码：0 全通过（警告不算失败）；1 存在 ERROR（按提示处理后重跑）。
"""
import argparse
import os
import re
import subprocess
import sys

OK, WARN, ERR = "OK  ", "WARN", "ERR "

results = []


def add(level, name, detail=""):
    results.append((level, name, detail))
    print(f"[{level}] {name}" + (f"  | {detail}" if detail else ""))


def run(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return -1, "", "not found"
    except subprocess.TimeoutExpired:
        return -2, "", f"timeout {timeout}s"


# ── 1. 驱动层可见性 ──
def check_devices():
    for dev in ("/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
        present = os.path.exists(dev)
        add(OK if present else ERR, f"device {dev}",
            "present" if present else
            "missing: 容器挂载缺设备（--device /dev/... 或 ascr-docker-runtime）")
    devs = [d for d in os.listdir("/dev") if d.startswith("davinci")
            and d != "davinci_manager"] if os.path.isdir("/dev") else []
    add(OK if devs else WARN, "davinci 节点", ",".join(sorted(devs)) or "未发现")


# ── 2. npu-smi：驱动存活 + 卡号解析（与 npu_ops._smi_card_ids 同算法）──
_SMI_CARD_RE = re.compile(r"(?:Ascend|\d{3}[A-Za-z]\d?)")


def smi_card_ids(out):
    ids = []
    for line in out.splitlines():
        if "Process id" in line:
            break
        parts = [p.strip() for p in line.strip("| ").split("|")]
        if not parts or not parts[0]:
            continue
        head = parts[0].split()
        if len(head) >= 2 and head[0].isdigit() and _SMI_CARD_RE.search(head[1]):
            ids.append(int(head[0]))
    return ids


def check_smi():
    rc, out, err = run(["npu-smi", "info"])
    if rc != 0:
        add(ERR, "npu-smi info", f"rc={rc} {err[:120]}（驱动未就绪？）")
        return []
    ids = smi_card_ids(out)
    add(OK if ids else WARN, "npu-smi 卡号解析", f"解析到芯片 {ids}"
        if ids else "未解析到（表格格式差异？贴输出给开发）")
    for ln in out.splitlines():
        if "Version" in ln or "Driver" in ln:
            m = re.search(r"Version[:\s]+([\d.]+)", ln)
            if m:
                add(OK, "npu-smi 版本", m.group(1))
            break
    # 健康状态：OK / Alarm 等
    bad = [ln.strip() for ln in out.splitlines()
           if re.search(r"\b(Alarm|Critical|Warning)\b", ln)]
    add(WARN if bad else OK, "芯片健康", bad[0][:100] if bad else "无异常状态行")
    return ids


# ── 3. 候选生成顺序（与 npu_ops._pick_npu_device 一致）──
def candidate_order(count, raw_env, smi_ids):
    order, seen = [], set()

    def add_i(i):
        if 0 <= i < 256 and i not in seen:
            seen.add(i)
            order.append(i)

    if raw_env.isdigit():
        add_i(int(raw_env))
    start = os.getpid() % max(count, 1)
    for k in range(max(count, 0)):
        add_i((start + k) % count)
    # 编排层注入的可见卡号（HAMi 注 ASCEND_RT_VISIBLE_DEVICES、
    # ascend-docker-runtime 注 ASCEND_VISIBLE_DEVICES）：驱动层真值，
    # 与 npu_ops._pick_npu_device 第 3 层同算法
    for var in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES"):
        for tok in os.getenv(var, "").split(","):
            tok = tok.strip()
            if tok.isdigit():
                add_i(int(tok))
    for i in smi_ids:
        add_i(i)
    return order


def check_order(smi_ids, deep):
    raw = os.getenv("ASCEND_DEVICE_ID", "").strip()
    count = None
    if deep:
        try:
            import torch
            __import__("torch_npu")
            count = int(torch.npu.device_count())
        except Exception as exc:  # noqa: BLE001
            add(WARN, "torch.npu.device_count", f"不可用: {type(exc).__name__}")
    if count is None:
        count = max(len(smi_ids), 1)
        add(WARN, "device_count（估算）", f"未做 torch 实探，按 smi 数量 {count} 估算")
    order = candidate_order(count, raw, smi_ids)
    add(OK, "ASCEND_DEVICE_ID", raw or "<未设置，自动发现>")
    add(OK, "探测候选顺序",
        f"显式 → torch 枚举(pid 打散) → env 可见卡 → smi 兜底: {order}")
    if raw.isdigit() and int(raw) >= count:
        add(WARN, "显式卡号越界", f"ASCEND_DEVICE_ID={raw} >= device_count({count})"
            "——vNPU 下属常态（会按显式值优先实探）")


# ── 4. 显存与共享内存 ──
def check_memory():
    try:
        st = os.statvfs("/dev/shm")
        gb = st.f_frsize * st.f_blocks / 2**30
        add(OK if gb >= 1 else ERR, "/dev/shm",
            f"{gb:.2f}GB" + ("" if gb >= 1 else " < 1GB：长文本 /tts 会 SIGBUS，"
                             "docker 加 --shm-size 2g / k8s emptyDir Memory 2Gi"))
    except (OSError, AttributeError):
        add(WARN, "/dev/shm", "不可读（非 Linux？）")
    rc, out, _ = run(["npu-smi", "info", "-m"])
    if rc == 0 and out:
        add(OK, "显存快照", out.splitlines()[0][:100])
    # 预算对拍（与 npu_ops._auto_budget_mb 同公式，默认参数）
    # L = ceil(120*1.03)+2 = 126；F = ceil(126*17.5/0.5) = 4410
    F, batch = 4410, 4
    live = 3 * batch * 128 * 60 * F * 4 / 2**20 * 1.15
    budget = (830 + live * 2.2) * 1.1
    add(OK, "默认配置预算", f"{budget:.0f}MB/worker（B=4,120字,0.5速）——"
        f"workers×预算 必须 ≤ 单卡可用显存")


# ── 5. deep：torch_npu 实探（与生产探测同款 zeros+synchronize）──
def check_deep(smi_ids):
    try:
        import torch
        __import__("torch_npu")
    except Exception as exc:  # noqa: BLE001
        add(WARN, "torch_npu 导入", f"不可用: {type(exc).__name__}: {exc}")
        return
    try:
        count = int(torch.npu.device_count())
        add(OK, "torch.npu.device_count", str(count))
    except Exception as exc:  # noqa: BLE001
        # vNPU 切分（HAMi 等）CANN 枚举可能整体缺失——不放弃，降级 count=0
        # 继续实探显式/env/驱动层候选（与 npu_ops._pick_npu_device 对齐）
        count = 0
        add(WARN, "torch.npu.device_count",
            f"不可用: {type(exc).__name__}: {str(exc)[:80]}"
            "——按 count=0 继续实探驱动层候选")
    raw = os.getenv("ASCEND_DEVICE_ID", "").strip()
    cands = candidate_order(count, raw, smi_ids)
    good = []
    for idx in cands:
        try:
            dev = torch.device(f"npu:{idx}")
            torch.zeros(1, device=dev)
            torch.npu.synchronize(dev)
            good.append(idx)
        except Exception as exc:  # noqa: BLE001
            add(WARN, f"probe npu:{idx}", f"{type(exc).__name__}: {str(exc)[:90]}")
    add(OK if good else ERR, "可用卡实探", f"可用={good}" if good else
        "候选全部失败——torch_npu 起着但卡不可用，生产会 fail-fast")
    if good:
        try:
            dev = torch.device(f"npu:{good[0]}")
            a = torch.ones(1024, device=dev)
            b = (a * 2).sum().item()
            add(OK, "基础算子", f"mul+sum={b:.0f} (期望 2048)")
        except Exception as exc:  # noqa: BLE001
            add(ERR, "基础算子", f"{type(exc).__name__}: {str(exc)[:100]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true",
                    help="额外做 torch_npu 实探（需在含 torch_npu 的环境跑）")
    args = ap.parse_args()
    check_devices()
    smi = check_smi()
    check_order(smi, args.deep)
    check_memory()
    if args.deep:
        check_deep(smi)
    else:
        add(WARN, "torch 实探", "未开启(--deep)；建议在镜像内再跑一次")
    errs = [r for r in results if r[0] == ERR]
    print("\n" + "=" * 60)
    print(f"总结：OK {sum(r[0]==OK for r in results)} / "
          f"WARN {sum(r[0]==WARN for r in results)} / ERR {len(errs)}")
    if errs:
        print("存在 ERROR —— 按上方提示处理后重跑。")
        return 1
    print("未发现阻塞项。镜像上传后建议在容器内再跑："
          "python3 scripts/npu_probe_check.py --deep")
    return 0


if __name__ == "__main__":
    sys.exit(main())

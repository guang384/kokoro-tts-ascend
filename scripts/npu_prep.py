#!/usr/bin/env python3
"""NPU/ARM 发布前静态与 arm64 镜像预检（QEMU 可执行部分）。

所有 CPU E2E 的最终目的都是 arm64 NPU 设备；本脚本把 QEMU 下可验证的
arm64 风险单独拉出来：
  - 生产 Dockerfile 的 arm64 镜像构建（无 torch import，QEMU 安全）
  - 镜像 Config.Env 全量 TUNABLE 默认值 + 底座遗留业务 env 检查
  - /app 模块清单与仓库根模块一致（拆分后新文件不能漏 COPY）
  - compileall / 全部非 torch 模块 import（清单见 IMPORT_MODULES；batched_model
    模块级 import torch，QEMU 下可能段错误，故排除）
  - arm64 镜像内 pytest（排除需 in-process torch / 真实 redis 多进程的用例）
  - 依赖版本（requirements.txt 精确钉住的 RELEASE_PINS）与 pip check
  - 模型资产已烘入镜像（Kokoro voice、FunASR model.pt）
  - NPU 预算公式锚点 / DEVICE=cpu 解析 / KOKORO_SAMPLE_RATE 启动守卫
  - torch 子进程探测：记录 QEMU 是否可 import torch（不可则明确标注，
    真机 Ascend 上再跑完整训练/推理）

用法（WSL，仓库根目录）：
  python3 scripts/npu_prep.py                 # 构建 + 检查
  python3 scripts/npu_prep.py --skip-build    # 复用已有 arm64 镜像
  python3 scripts/npu_prep.py --image kokoro-tts-ascend:npuprep
注意：不导出/保存镜像 tar（按用户规则需单独确认）。
"""
import argparse
import glob
import os
import re
import sys
from datetime import datetime

from cpu_e2e import E2E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import TUNABLE_PARAMS  # noqa: E402  默认值单一来源

DEFAULT_IMAGE = "kokoro-tts-ascend:npuprep-" + datetime.now().strftime("%Y%m%d")
PLATFORM = "linux/arm64"

# 业务 env 命名空间：镜像里出现这些前缀、却既不在 Dockerfile 也不在
# TUNABLE_PARAMS 的键，说明底座镜像带了历史遗留配置（如旧 KOKORO_LANG_CODE）。
BUSINESS_ENV_PREFIXES = ("KOKORO_", "TTS_", "ASR_", "G2P_", "GATEWAY_",
                         "FUNASR_", "WAIT_", "MAX_", "WORKER_")
# 部署/内部注入变量：不属于 TUNABLE_PARAMS，但允许出现在镜像 env 中。
KNOWN_EXTRA_ENV = {"FUNASR_MODEL_DIR", "KOKORO_HEALTH_DIR",
                   "WORKER_TAG", "WORKER_IDX", "IMAGE_VERSION"}

# arm64 import 冒烟清单：除 batched_model（模块级 import torch，QEMU 下可能
# 段错误）外的全部运行时模块。test_packaging_consistency 保证它与仓库根
# 模块清单一致——新增模块漏加时测试失败，不必等发布预检才发现。
IMPORT_MODULES = ("gateway", "gateway_state", "gateway_agg", "gateway_obs",
                  "audio_enc", "pipeline", "npu_ops", "funasr_ts",
                  "supervisor", "g2p", "compute", "worker_common",
                  "tts_worker", "asr_worker", "proc_common", "config",
                  "redis_transport", "heartbeat", "protocol")

# 发布必须精确钉住的依赖：版本从 requirements.txt 解析，避免预检脚本与
# 安装清单各写一份字面量后漂移（改 requirements 而忘改预检 = 门禁失真）。
RELEASE_PINS = ("funasr", "kokoro", "pypinyin")


def _pinned_versions(text: str) -> dict:
    """从 requirements.txt 文本解析 `name==version` 精确钉版（忽略注释/额外索引）。

    只认裸包名：`misaki[zh]==0.9.4` 这类 extras 形式不参与发布门禁。
    """
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def _docker_env_map(text: str) -> dict:
    """解析 Dockerfile 的 ENV 指令（含续行），忽略注释里的 `KEY=value`。

    不能用全局正则扫全文：Dockerfile 注释里有 `GATEWAY_WORKERS=N` 这类
    示例，会被误当成真实默认值。"""
    env = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ENV "):
            buf = line[4:]
            while buf.endswith("\\") and i + 1 < len(lines):
                i += 1
                buf = buf[:-1] + " " + lines[i].strip()
            for m in re.finditer(r"([A-Z][A-Z0-9_]*)=(\"[^\"]*\"|\S*)", buf):
                env[m.group(1)] = m.group(2).strip('"')
        i += 1
    return env


def _env_drift(image_env: dict, docker_env: dict, tunables: dict):
    """返回 (值不一致, Dockerfile 有而镜像缺失, 遗留业务 env)。

    覆盖三类历史漂移：
    - 增量 FROM 旧镜像，底座 ENV 覆盖 Dockerfile/代码默认（6 tts/3 asr/B4）；
    - Dockerfile 未写、由代码默认兜底的 TUNABLE 键，被底座设成旧值；
    - 底座残留已废弃的业务 env（旧 KOKORO_LANG_CODE）。"""
    mismatched, missing, legacy = {}, {}, {}
    docker_business = {k: v for k, v in docker_env.items() if k in tunables}
    for key, want in docker_business.items():
        got = image_env.get(key)
        if got is None:
            missing[key] = want
        elif got != want:
            mismatched[key] = f"image={got!r} Dockerfile={want!r}"
    # Dockerfile 里还有非 TUNABLE 的部署常量（FUNASR_MODEL_DIR / PYTHONPATH /
    # HF_HOME ...）：底座镜像若带旧值同样会覆盖 Dockerfile，必须一起校验。
    # 含 ${...} 的是 build-arg 模板（PIP_INDEX_URL/HF_ENDPOINT/IMAGE_VERSION），
    # 镜像里是展开后的值，不能按字面比较。
    for key, want in docker_env.items():
        if key in tunables or "${" in want:
            continue
        got = image_env.get(key)
        if got is None:
            missing[key] = want
        elif got != want:
            mismatched[key] = f"image={got!r} Dockerfile={want!r}"
    for key, default in tunables.items():
        if key in docker_business:
            continue
        got = image_env.get(key)
        if got is not None and got != default:
            mismatched[key] = (f"image={got!r} code-default={default!r}"
                               " (Dockerfile 未设置)")
    for key, val in image_env.items():
        if key in tunables or key in docker_env or key in KNOWN_EXTRA_ENV:
            continue
        if key.startswith(BUSINESS_ENV_PREFIXES):
            legacy[key] = val
    return mismatched, missing, legacy


class NpuPrep(E2E):
    def __init__(self, args):
        self.args = args
        self.image = args.image
        self.assets = ""
        self.port = 5174
        self.args.keep = False   # E2E.cleanup 语义
        self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = os.path.abspath(os.path.join("artifacts", f"npu-prep-{self.stamp}"))
        os.makedirs(self.out, exist_ok=True)
        self.checks = []
        self.containers = []
        self.current_phase = "npu-prep"

    def armrun(self, *cmd, env=None, timeout=1800):
        full = ["run", "--rm", "--platform", PLATFORM]
        for k, v in (env or {}).items():
            full += ["-e", f"{k}={v}"]
        full += [self.image, *cmd]
        return self.docker(*full, timeout=timeout)

    def phase_build(self):
        self.current_phase = "build"
        if self.args.skip_build:
            rc, out = self.docker("image", "inspect", self.image)
            return self.check("arm64 镜像已存在（--skip-build）", rc == 0, out[-200:])
        self.note(f"docker build --platform {PLATFORM} -f Dockerfile -t {self.image} ...")
        rc, out = self.docker(
            "build", "--platform", PLATFORM, "-f", "Dockerfile",
            "--build-arg", f"IMAGE_VERSION={self.args.version}",
            "-t", self.image, ".", timeout=7200)
        self.save("arm64-build.log", out)
        self.check("生产 arm64 镜像构建成功", rc == 0, out[-600:])
        return rc == 0

    def run_checks(self):
        # 1) 架构 / 入口 / 健康检查元数据
        rc, out = self.docker("image", "inspect", self.image, "--format",
                              "{{.Architecture}}|{{.Os}}|{{.Config.Cmd}}|"
                              "{{.Config.ExposedPorts}}|{{index .Config.Labels "
                              "\"org.opencontainers.image.version\"}}")
        self.save("image-inspect.txt", out)
        parts = out.strip().split("|")
        arch = parts[0] if parts else ""
        self.check("镜像 tag 带 8 位日期", bool(re.search(r"20\d{6}", self.image)),
                   self.image)
        self.check("IMAGE_VERSION label 带日期",
                   bool(re.search(r"20\d{6}",
                                  parts[-1] if parts else "")),
                   parts[-1] if parts else "")
        self.check("镜像架构 arm64/linux", arch == "arm64", out.strip()[:300])
        self.check("入口为 supervisor.py",
                   "supervisor.py" in (parts[2] if len(parts) > 2 else ""),
                   parts[2] if len(parts) > 2 else "")
        self.check("EXPOSE 5174", "5174" in (parts[3] if len(parts) > 3 else ""),
                   parts[3] if len(parts) > 3 else "")

        # 1b) 镜像 Config.Env 必须重放当前 Dockerfile 的业务默认值；Dockerfile
        # 未显式设置、由代码默认兜底的 TUNABLE 键，若被底座镜像设过旧值也必须
        # 拦住。增量 overlay 从旧镜像 FROM 时最容易漏：旧 TTS_WORKERS=6 /
        # ASR_WORKERS=3 / TTS_MAX_BATCH_SIZE=4 会覆盖代码默认（实测 6/3/B4）。
        rc, env_out = self.docker(
            "image", "inspect", self.image, "--format",
            "{{range .Config.Env}}{{println .}}{{end}}")
        env = dict(ln.split("=", 1) for ln in env_out.splitlines() if "=" in ln)
        df_text = open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8").read()
        df_env = _docker_env_map(df_text)

        df_conflicts = {k: (v, TUNABLE_PARAMS[k]) for k, v in df_env.items()
                        if k in TUNABLE_PARAMS and v != TUNABLE_PARAMS[k]}
        self.check("Dockerfile ENV 与 TUNABLE_PARAMS 默认一致",
                   not df_conflicts, str(df_conflicts))
        mismatched, missing, legacy = _env_drift(env, df_env, TUNABLE_PARAMS)
        self.save("image-env.txt", "\n".join(f"{k}={env.get(k)}"
                                              for k in sorted(env)))
        self.check("镜像 ENV 业务默认与 Dockerfile/代码默认一致",
                   not mismatched and not missing,
                   f"mismatch={mismatched} missing={missing}")
        self.check("镜像无遗留旧业务 ENV", not legacy, str(legacy))

        # 2) /app 模块清单与仓库根模块一致（拆分后新文件漏 COPY 会在这里暴露）
        host_mods = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*.py"))
                           if os.path.basename(p) != "conftest.py")
        rc, out = self.armrun("sh", "-c", "ls /app/*.py | xargs -n1 basename | sort")
        image_mods = [x for x in out.split()
                      if x.endswith(".py") and x != "conftest.py"]
        self.check("/app 模块清单与仓库一致", image_mods == host_mods,
                   f"missing={sorted(set(host_mods) - set(image_mods))} "
                   f"extra={sorted(set(image_mods) - set(host_mods))}")
        self.check("/app tests/scripts 已随镜像分发",
                   self.armrun("sh", "-c",
                               "test -d /app/tests && test -d /app/scripts && "
                               "test -f /opt/benchmark.py && echo ok")[1].strip() == "ok",
                   "")

        # 3) 编译（不 import，QEMU 安全）
        rc, out = self.armrun("python", "-m", "compileall", "-q", "/app", timeout=900)
        self.check("arm64 镜像 python compileall", rc == 0, out[-300:])

        # 4) 模块 import（清单单一来源；batched_model 模块级 import torch 被排除）
        mods = [m for m in IMPORT_MODULES if f"{m}.py" in host_mods]
        rc, out = self.armrun("python", "-c",
                              "import " + ", ".join(mods) + "; print('imports ok')",
                              timeout=900)
        self.save("arm64-import-smoke.txt", out)
        self.check(f"arm64 {len(mods)} 模块 import（不含 torch 模块）",
                   rc == 0 and "imports ok" in out, out.strip()[-300:])

        # 5) 依赖版本 / pip check（期望值来自 requirements.txt，不写死）
        rc, out = self.armrun(
            "python", "-c",
            "from importlib.metadata import version as v;"
            "print('funasr',v('funasr'));print('kokoro',v('kokoro'));"
            "print('pypinyin',v('pypinyin'));print('redis',v('redis'))", timeout=300)
        self.save("arm64-versions.txt", out)
        req_text = open(os.path.join(ROOT, "requirements.txt"),
                        encoding="utf-8").read()
        pins = _pinned_versions(req_text)
        missing_pins = [n for n in RELEASE_PINS if n not in pins]
        self.check("requirements.txt 精确钉住发布依赖", not missing_pins,
                   f"missing={missing_pins}")
        self.check("关键依赖版本与 requirements.txt 一致",
                   not missing_pins
                   and all(f"{n} {pins[n]}" in out for n in RELEASE_PINS),
                   out.strip())
        rc, out = self.armrun("pip", "check", timeout=600)
        self.save("arm64-pip-check.txt", out)
        # 华为基底自带 mindstudio/ms-service-profiler 等工具链的缺失依赖与
        # numpy<=1.26.4 约束；本服务不使用它们，只门控项目自身依赖。
        project_conflicts = [
            ln for ln in out.splitlines()
            if ln.lower().startswith(("funasr", "kokoro", "misaki", "pypinyin",
                                      "fastapi", "uvicorn", "soundfile",
                                      "huggingface", "prometheus", "redis"))
        ]
        self.check("项目依赖无 pip 冲突", not project_conflicts,
                   str(project_conflicts[:5]))
        self.note(f"pip check 基底工具告警（不计入门控）: "
                  f"{len(out.strip().splitlines()) - len(project_conflicts)} 行")

        # 6) 系统工具
        for tool, token, flag in (("redis-server", "Redis", "--version"),
                                  ("ffmpeg", "ffmpeg", "-version"),
                                  ("espeak-ng", "eSpeak", "--version")):
            rc, out = self.armrun(tool, flag, timeout=180)
            self.check(f"arm64 镜像含 {tool}",
                       rc == 0 and token.lower() in out.lower(),
                       f"rc={rc} {out.strip().splitlines()[0] if out.strip() else ''}")

        # 7) 模型资产已烘入镜像
        rc, out = self.armrun(
            "sh", "-c",
            "test -f /opt/funasr-models/timestamp-v1/model.pt && "
            "stat -c%s /opt/funasr-models/timestamp-v1/model.pt && "
            "ls /opt/hf-cache/hub/models--hexgrad--Kokoro-82M-v1.1-zh/"
            "snapshots/*/voices/zf_001.pt", timeout=300)
        lines = [x for x in out.strip().splitlines() if x]
        size = int(lines[0]) if lines and lines[0].isdigit() else 0
        self.check("FunASR 模型已烘入（>150MB）", size > 150000000, f"{size}")
        self.check("Kokoro voice 已烘入", len(lines) > 1 and lines[1].endswith("zf_001.pt"),
                   str(lines[1:]))

        # 8) NPU 预算公式锚点（不 import torch）
        rc, out = self.armrun(
            "python", "-c",
            "import npu_ops;print(npu_ops._auto_budget_mb(8,120,0.5));"
            "print(npu_ops._asr_auto_budget_mb(8,30.0));"
            "print(npu_ops.capacity_check.__name__)", timeout=300)
        self.save("arm64-budget-anchors.txt", out)
        self.check("arm64 预算锚点 (9543,126,55.1)/(2689,500,30.0)",
                   "(9543, 126, 55.1)" in out and "(2689, 500, 30.0)" in out,
                   out.strip())
        rc, out = self.armrun("python", "-c",
                              "import npu_ops;print(npu_ops._resolve_disable_complex('npu:0'));"
                              "print(npu_ops._resolve_disable_complex('cpu'))",
                              env={"KOKORO_DISABLE_COMPLEX": "auto"}, timeout=120)
        self.check("arm64 auto disable_complex: npu=True/cpu=False",
                   out.split() == ["True", "False"], out.strip())
        rc, out = self.armrun("python", "-c",
                              "import npu_ops;print(npu_ops._resolve_device())",
                              env={"DEVICE": "cpu"}, timeout=120)
        self.check("DEVICE=cpu 解析不触碰 NPU", out.strip() == "cpu", out.strip())

        # 9) 采样率启动守卫（arm64 上也必须 fail-fast）
        rc, out = self.armrun("python", "-c", "import pipeline",
                              env={"KOKORO_SAMPLE_RATE": "44100"}, timeout=120)
        self.check("KOKORO_SAMPLE_RATE!=24000 启动即拒绝",
                   rc != 0 and "不支持" in out, out.strip()[-200:])

        # 10) torch 子进程探测（QEMU 常见 xbyak 段错误；不作为失败）
        # QEMU 下 torch import 会卡在 oneDNN/xbyak MIDR 读；容器内 timeout
        # 保证进程退出（否则只杀 docker CLI 会留下僵尸容器）。
        rc, out = self.armrun("timeout", "60", "python", "-c",
                              "import torch;print('torch',torch.__version__)",
                              timeout=180)
        torch_ok = rc == 0
        self.save("arm64-torch-probe.txt", out)
        self.note(f"QEMU torch probe rc={rc}: {out.strip()[:200]}")
        self.check("QEMU torch probe 结果已记录（不门控）", True,
                   f"rc={rc} ok={torch_ok}")

        # 11) arm64 镜像内 pytest：排除 in-process torch 与真实 redis 多进程
        ignores = ["tests/test_funasr_math.py", "tests/test_real_forward.py",
                   "tests/test_tts_api.py", "tests/test_claim_integration.py",
                   "tests/test_ipc_crashsafe.py"]
        args = ["python", "-m", "pytest", "tests/", "-q", "-rs"]
        for path in ignores:
            args += ["--ignore", path]
        rc, out = self.armrun(*args, timeout=3600)
        self.save("arm64-pytest.txt", out)
        last = out.strip().splitlines()[-1] if out.strip() else ""
        self.check("arm64 镜像 pytest（QEMU 安全子集）", rc == 0, last)
        self.check("arm64 pytest 无意外 skip（仅允许 torch 探测类）",
                   "skipped" not in last or "passed" in last, last)

        return torch_ok

    def run(self):
        os.chdir(ROOT)
        try:
            if not self.phase_build():
                return self.report()
            torch_ok = self.run_checks()
            self.note(f"QEMU 下 torch 可导入: {torch_ok}；"
                      "真机 Ascend 需再跑 supervisor+模型验证")
        finally:
            self.cleanup()
        return self.report()


def main():
    ap = argparse.ArgumentParser(description="NPU/ARM 发布前预检")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--version",
                    default="npuprep-" + datetime.now().strftime("%Y%m%d"))
    args = ap.parse_args()
    prep = NpuPrep(args)
    code = 0 if prep.run() else 1
    print(f"[npu_prep] exit={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

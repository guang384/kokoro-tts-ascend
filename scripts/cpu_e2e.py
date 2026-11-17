#!/usr/bin/env python3
"""CPU 端到端验证驱动（x86 Docker，原生运行，无 QEMU）。

阶段：
  build    构建 Dockerfile.cpu 并做 import 冒烟
  pytest   镜像内全量 pytest（校验 skip 集合，防“假绿”）
  baseline 真实模型起 supervisor：/health 就绪、双端点、格式、多批、时间戳质量
  numeric  validate_batch / validate_asr_batch / test_real_forward 三个数值门
  fault    parity_smoke、SO_REUSEPORT、kill redis、ASR 503、SHM 清扫、
           429×3/504/499、优雅退出
  report   汇总 summary.json / summary.md

用法（WSL，仓库根目录）：
  python3 scripts/cpu_e2e.py --assets cpu-e2e-assets
  python3 scripts/cpu_e2e.py --assets cpu-e2e-assets --phases baseline,numeric
  python3 scripts/cpu_e2e.py --skip-build --keep --phases fault

产物：artifacts/cpu-e2e-<timestamp>/（日志、health/metrics 快照、pytest.xml、
tts.json、各数值门输出、summary.*）。任何 gate 失败退出码 1。

依赖：宿主仅需 python3 标准库 + docker CLI；模型资产由
scripts/extract_cpu_assets.sh 提前准备并只读挂载。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_IMAGE = "kokoro-tts-cpu:local"
DEFAULT_ASSETS = "cpu-e2e-assets"
DEFAULT_PORT = 5174

# 容器内公共环境（模型路径与离线属性；设备显式 CPU）
COMMON_ENV = {
    "DEVICE": "cpu",
    "FUNASR_DEVICE": "cpu",
    "HF_HOME": "/opt/hf-cache",
    "FUNASR_MODEL_DIR": "/opt/funasr-models/timestamp-v1",
    "KOKORO_HEALTH_DIR": "/tmp/kokoro-health",
    "PORT": str(DEFAULT_PORT),
    "LOG_LEVEL": "INFO",
}

ALLOWED_IMAGE_SKIP = ("tests.test_tts_api", "tests.test_real_forward")


class E2E:
    def __init__(self, args):
        self.args = args
        self.image = args.image
        self.assets = os.path.abspath(args.assets)
        self.port = args.port
        self.stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out = os.path.abspath(os.path.join("artifacts", f"cpu-e2e-{self.stamp}"))
        os.makedirs(self.out, exist_ok=True)
        self.checks = []
        self.containers = []
        self.current_phase = ""
        self.fatal = False

    # ── 记录与输出 ────────────────────────────────────────────────
    def check(self, name, ok, detail=""):
        ok = bool(ok)
        self.checks.append({"phase": self.current_phase, "name": name,
                            "ok": ok, "detail": str(detail)[:400]})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  {detail}" if detail else ""), flush=True)
        return ok

    def note(self, msg):
        print(f"  {msg}", flush=True)

    def save(self, name, data):
        path = os.path.join(self.out, name)
        mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
        kw = {} if mode == "wb" else {"encoding": "utf-8"}
        with open(path, mode, **kw) as f:
            f.write(data)
        return path

    # ── 进程 ──────────────────────────────────────────────────────
    def sh(self, cmd, timeout=600, check=False):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            out = (exc.stdout or "") + (exc.stderr or "")
            if check:
                raise RuntimeError(f"timeout({timeout}s): {' '.join(cmd)}\n{out[-2000:]}")
            return 124, out
        out = (p.stdout or "") + (p.stderr or "")
        if check and p.returncode != 0:
            raise RuntimeError(f"rc={p.returncode}: {' '.join(cmd)}\n{out[-2000:]}")
        return p.returncode, out

    def docker(self, *args, timeout=600, check=False):
        return self.sh(["docker", *args], timeout=timeout, check=check)

    def dexec(self, container, *args, timeout=1800, check=False):
        return self.docker("exec", container, *args, timeout=timeout, check=check)

    def logs(self, container):
        rc, out = self.docker("logs", container)
        self.save(f"{container}.log", out)
        return out

    def remove(self, container):
        self.docker("rm", "-f", container, timeout=60)

    def cleanup(self):
        if self.args.keep:
            self.note(f"--keep: 保留容器 {self.containers}")
            return
        for c in self.containers:
            self.remove(c)
        self.containers.clear()

    # ── 容器编排 ──────────────────────────────────────────────────
    def start(self, name, env, shm="2g", host_port=None):
        self.remove(name)
        port = host_port or self.port
        cmd = ["run", "-d", "--name", name,
               "-p", f"127.0.0.1:{port}:{DEFAULT_PORT}",
               "--shm-size", shm,
               "-v", f"{self.assets}/hf-cache:/opt/hf-cache:ro",
               "-v", f"{self.assets}/funasr-models:/opt/funasr-models:ro",
               "-v", f"{self.out}:/artifacts"]
        merged = dict(COMMON_ENV)
        merged.update({k: str(v) for k, v in env.items()})
        for k, v in merged.items():
            cmd += ["-e", f"{k}={v}"]
        cmd.append(self.image)
        rc, out = self.docker(*cmd, timeout=120)
        if rc != 0:
            raise RuntimeError(f"docker run 失败: {out[-2000:]}")
        cid = out.strip().splitlines()[-1]
        self.containers.append(name)
        self.note(f"容器 {name} 启动 cid={cid[:12]} host_port={port}")
        return name

    def wait_ready(self, base, timeout):
        t0 = time.time()
        seen = []
        while time.time() - t0 < timeout:
            st, _h, body = self.http(base, "/health", timeout=10)
            if st == 200:
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = {}
                if payload.get("ready"):
                    self.note(f"就绪耗时 {time.time() - t0:.0f}s（历史 {seen[-3:]}）")
                    return payload
            elif st is not None:
                seen.append(st)
            time.sleep(2)
        raise RuntimeError(f"/health 在 {timeout}s 内未 ready（观测 {seen[-5:]}）")

    # ── HTTP ──────────────────────────────────────────────────────
    def http(self, base, path, payload=None, timeout=600):
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()
        except Exception as exc:  # noqa: BLE001  连接拒绝/超时/断连
            return None, {}, repr(exc).encode()

    def metrics(self, base):
        st, _h, body = self.http(base, "/metrics", timeout=30)
        return body.decode("utf-8", "replace") if st == 200 else ""

    # ── 阶段：构建 ────────────────────────────────────────────────
    def phase_build(self):
        self.current_phase = "build"
        if not self.args.skip_build:
            self.note(f"docker build -f Dockerfile.cpu -t {self.image} ...")
            rc, out = self.docker("build", "-f", "Dockerfile.cpu",
                                  "-t", self.image,
                                  "--build-arg",
                                  f"TORCH_VERSION={self.args.torch_version}",
                                  ".", timeout=3600)
            self.save("docker-build.log", out)
            if not self.check("CPU 镜像构建成功", rc == 0, out[-600:]):
                self.fatal = True
                return False
        else:
            rc, out = self.docker("image", "inspect", self.image)
            if not self.check("镜像已存在（--skip-build）", rc == 0, out[-300:]):
                self.fatal = True
                return False
        rc, out = self.docker("run", "--rm", self.image, "python", "-c",
                              "import torch, kokoro, funasr;"
                              "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())",
                              timeout=600)
        self.save("image-import-smoke.txt", out)
        self.check("镜像 import 冒烟（torch/kokoro/funasr）", rc == 0, out.strip()[-300:])
        return rc == 0

    # ── 阶段：镜像内 pytest ──────────────────────────────────────
    def phase_pytest(self):
        self.current_phase = "pytest"
        junit = os.path.join(self.out, "pytest-image.xml")
        rc, out = self.docker(
            "run", "--rm", "-v", f"{self.out}:/artifacts", self.image,
            "python", "-m", "pytest", "tests/", "-q", "-rs",
            "--junitxml=/artifacts/pytest-image.xml", timeout=3600)
        self.save("pytest-image.txt", out)
        self.check("镜像内 pytest 退出码 0", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")
        bad_skip, redis_ran, skipped_n = [], False, 0
        if os.path.isfile(junit):
            root = ET.parse(junit).getroot()
            for case in root.iter("testcase"):
                cls = case.get("classname") or ""
                skipped = case.find("skipped") is not None
                if skipped:
                    skipped_n += 1
                if skipped and not cls.startswith(ALLOWED_IMAGE_SKIP):
                    bad_skip.append(f"{cls}::{case.get('name')}")
                if cls.startswith(("tests.test_claim_integration",
                                   "tests.test_ipc_crashsafe")) and not skipped:
                    redis_ran = True
            totals = root if root.tag == "testsuite" else next(root.iter("testsuite"), None)
            fails = int(totals.get("failures", 0)) + int(totals.get("errors", 0)) \
                if totals is not None else -1
            self.check("镜像内 pytest 无失败/错误", fails == 0, f"failures+errors={fails}")
        else:
            self.check("pytest JUnit 产物存在", False, junit)
        self.check("skip 仅有环境门控（live smoke/real forward）", not bad_skip,
                   str(bad_skip[:8]))
        self.check("skip 数量符合预期（7）", skipped_n == 7, f"skipped={skipped_n}")
        self.check("真实 redis 集成测试已执行（非 skip）", redis_ran,
                   "test_claim_integration/test_ipc_crashsafe")
        return rc == 0 and not bad_skip and redis_ran

    # ── 阶段：基线功能/契约 ──────────────────────────────────────
    def phase_baseline(self):
        self.current_phase = "baseline"
        name = "ktts-e2e-baseline"
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 1, "G2P_WORKERS": 1,
                "TTS_WORKERS": 1, "ASR_WORKERS": 1,
                "TTS_MAX_BATCH_SIZE": 1, "ASR_MAX_BATCH_SIZE": 1,
                "TTS_CHUNK_CHARS": 60, "TTS_MAX_INPUT_CHARS": 500,
                "SPAWN_READY_TIMEOUT_S": 0, "STUCK_KILL_S": 1800,
                "HEARTBEAT_STALE_S": 120, "WAIT_TIMEOUT_S": 300,
                "WEDGE_RESTART_S": 0, "KOKORO_EN_G2P": "auto",
            })
            base = f"http://127.0.0.1:{self.port}"
            health = self.wait_ready(base, self.args.startup_timeout)
            self.save("baseline-health.json", json.dumps(health, ensure_ascii=False, indent=2))
            log = self.logs(name)
            self.check("config dump 显示 DEVICE=cpu", "DEVICE=cpu" in log, "")
            self.check("config dump 显示 FUNASR_DEVICE=cpu", "FUNASR_DEVICE=cpu" in log, "")
            self.check("worker 日志显示 device=cpu", "device=cpu" in log, "")

            mt = self.metrics(base)
            self.save("baseline-metrics.txt", mt)
            self.check("/health 顶层恰好 5 键",
                       set(health) == {"status", "ready", "version", "shm", "recent_failures"},
                       str(sorted(health)))
            for role in ("gateway", "g2p", "tts", "asr"):
                vals = re.findall(
                    rf'^kokoro_worker_ready\{{[^}}]*role="{role}"[^}}]*\}}\s+(\S+)$',
                    mt, re.M)
                self.check(f"/metrics {role} worker ready=1", vals and all(v == "1.0" for v in vals),
                           str(vals))
            npu_samples = re.search(r"^kokoro_worker_npu_mem_[a-z_]*\{", mt, re.M)
            self.check("CPU 无 NPU 内存指标样本", not npu_samples,
                       (npu_samples.group(0) if npu_samples else ""))

            st, _h, body = self.http(base, "/v1/models")
            self.check("/v1/models", st == 200 and b"kokoro-v1.1-zh" in body, f"{st}")
            st, _h, body = self.http(base, "/v1/voices")
            voices = json.loads(body).get("voices", []) if st == 200 else []
            self.check("/v1/voices > 10", len(voices) > 10, f"{len(voices)}")

            # /v1 格式与响应头
            for fmt, magic in (("wav", b"RIFF"), ("pcm", None), ("flac", b"fLaC")):
                st, h, body = self.http(base, "/v1/audio/speech",
                                        {"input": "格式验证。", "voice": "zf_001",
                                         "response_format": fmt})
                ok = st == 200 and len(body) > 1000 and (magic is None or body.startswith(magic))
                self.check(f"/v1 {fmt}", ok, f"{st} {len(body)}B")
                if fmt == "wav":
                    self.check("/v1 wav 响应头",
                               h.get("x-audio-duration", "0") > "0"
                               and float(h.get("x-audio-duration", 0)) > 0
                               and float(h.get("x-synth-time", 0)) > 0
                               and int(h.get("x-batch-size", 0)) >= 1
                               and re.fullmatch(r"[0-9a-f]{12}", h.get("x-request-id", "")),
                               str({k: h.get(k) for k in
                                    ("x-audio-duration", "x-synth-time",
                                     "x-batch-size", "x-request-id")}))
            st, _h, body = self.http(base, "/v1/audio/speech",
                                     {"input": "mp3 格式验证。", "voice": "zf_001",
                                      "response_format": "mp3"}, timeout=600)
            self.check("/v1 mp3（ffmpeg）", st == 200 and len(body) > 500, f"{st} {len(body)}B")

            # 多批聚合：chunk=60，长文本应产生多个 synth_part
            long_text = "语音合成批量验证句子。" * 12
            st, h, body = self.http(base, "/v1/audio/speech",
                                    {"input": long_text, "voice": "zf_001",
                                     "response_format": "pcm"}, timeout=900)
            batch = int(h.get("x-batch-size", 0) or 0)
            self.check("/v1 多批聚合 X-Batch-Size>1", st == 200 and batch > 1,
                       f"{st} batch={batch} chars={len(long_text)}")

            # /tts 字级时间戳 + 严格对齐门
            text = "你好欢迎使用语音合成服务"
            st, h, body = self.http(base, "/tts",
                                    {"text": text, "speed": 1.0}, timeout=900)
            ok, detail = False, f"{st}"
            if st == 200:
                d = json.loads(body)
                self.save("tts.json", json.dumps(d, ensure_ascii=False))
                ts = d.get("timestamp") or []
                n_align = sum(1 for c in text
                              if (c.isascii() and c.isalnum())
                              or "\u4e00" <= c <= "\u9fff")
                ok = (set(d) == {"audio_base64", "code", "duration", "message",
                                 "text", "timestamp"}
                      and d.get("code") == 0 and d.get("message") == "success"
                      and d.get("text") == text and d.get("duration", 0) > 0
                      and len(ts) == n_align
                      and all(e >= s for s, e in ts)
                      and all(ts[i][0] >= ts[i - 1][0] for i in range(1, len(ts)))
                      and ts and ts[-1][1] <= d["duration"] * 1000 + 200)
                detail = f"{st} keys={sorted(d) if hasattr(d,'keys') else d} ts={len(ts)}/{n_align}"
            self.check("/tts 6 键契约+时间戳段数/单调/时长", ok, detail)
            if st == 200:
                rc, out = self.dexec(name, "python", "scripts/check_align.py",
                                     "--strict", "/artifacts/tts.json")
                self.save("check-align.txt", out)
                self.check("check_align --strict", rc == 0, out.strip().splitlines()[-1] if out.strip() else "")

            # 现有冒烟脚本
            rc, out = self.dexec(name, "python", "-m", "pytest",
                                 "tests/test_tts_api.py", "-v",
                                 env_extra={"RUN_LIVE_SMOKE": "1"})
            self.save("test-tts-api.txt", out)
            self.check("tests/test_tts_api.py（live smoke）", rc == 0,
                       out.strip().splitlines()[-1] if out.strip() else "")

            rc, out = self.dexec(name, "python", "scripts/verify_boundaries.py",
                                 f"http://127.0.0.1:{DEFAULT_PORT}", "--max-chars=200",
                                 timeout=1800)
            self.save("verify-boundaries.txt", out)
            self.check("scripts/verify_boundaries.py", rc == 0,
                       out.strip().splitlines()[-1] if out.strip() else "")

            # 业务覆盖与错误路径
            st, _h, _b = self.http(base, "/tts",
                                   {"text": "欢迎长乐平台",
                                    "g2p_overrides": {"长乐": "zhang3 le4"}},
                                   timeout=600)
            self.check("/tts g2p_overrides", st == 200, f"{st}")
            for label, payload, allowed in (
                    ("/tts 空文本", {"text": ""}, (400, 422)),
                    ("/tts 非法 dtype", {"text": "你好", "dtype": "fp8"}, (400,)),
                    ("/tts 未知音色", {"text": "你好", "voice": "no_such_voice"}, (400,)),
                    ("/v1 非法格式", {"input": "你好", "response_format": "ogg"}, (422,))):
                st, _h, _b = self.http(base, "/tts" if label.startswith("/tts") else "/v1/audio/speech",
                                       payload, timeout=60)
                self.check(label, st in allowed, f"{st}")
            return True
        except Exception as exc:  # noqa: BLE001
            self.check("baseline 阶段异常", False, str(exc))
            self.logs(name)
            return False

    # ── 阶段：数值门 ──────────────────────────────────────────────
    def phase_numeric(self):
        self.current_phase = "numeric"
        name = "ktts-e2e-baseline"
        if name not in self.containers:
            self.check("数值门需要 baseline 容器仍运行", False, "容器不存在")
            return False
        ok = True
        for label, args, timeout in (
                ("validate_batch", ["python", "tests/validate_batch.py"], 1800),
                ("validate_asr_batch", ["python", "tests/validate_asr_batch.py"], 3600),
                ("test_real_forward",
                 ["python", "-m", "pytest", "tests/test_real_forward.py", "-v"], 1800)):
            extra = {"RUN_REAL_FORWARD": "1"} if label == "test_real_forward" else {}
            rc, out = self.dexec(name, *args, timeout=timeout, env_extra=extra)
            self.save(f"{label}.txt", out)
            self.check(f"数值门 {label}", rc == 0,
                       out.strip().splitlines()[-1] if out.strip() else "")
            ok = ok and rc == 0
        return ok

    # ── 进程/心跳辅助 ────────────────────────────────────────────
    def role_pid(self, container, role):
        code = (
            "import glob,json\n"
            "for f in glob.glob('/tmp/kokoro-health/worker-*.json'):\n"
            "    try:\n"
            "        d=json.load(open(f))\n"
            "    except Exception:\n"
            "        continue\n"
            f"    if d.get('role') == '{role}':\n"
            "        print(d.get('pid')); break\n")
        rc, out = self.dexec(container, "python3", "-c", code)
        m = re.search(r"\d+", out)
        return int(m.group(0)) if m else None

    def signal_pid(self, container, pid, sig):
        self.dexec(container, "python3", "-c",
                   f"import os,signal; os.kill({int(pid)}, signal.{sig})")

    def kill_redis(self, container):
        lines = [
            "import glob,os,signal",
            "for p in glob.glob('/proc/[0-9]*'):",
            "    try:",
            "        cmd=open(p+'/cmdline','rb').read()",
            "    except Exception:",
            "        continue",
            "    if b'redis-server' in cmd:",
            "        os.kill(int(p.rsplit('/',1)[-1]), signal.SIGKILL)",
        ]
        return self.dexec(container, "python3", "-c", "\n".join(lines))

    def heartbeat_summary(self, container):
        code = (
            "import glob,json\n"
            "out=[]\n"
            "for f in glob.glob('/tmp/kokoro-health/worker-*.json'):\n"
            "    try:\n"
            "        out.append(json.load(open(f)))\n"
            "    except Exception:\n"
            "        pass\n"
            "print(json.dumps(out, ensure_ascii=False))\n")
        rc, out = self.dexec(container, "python3", "-c", code)
        try:
            return json.loads(out)
        except ValueError:
            return []

    # ── 阶段：故障注入 ────────────────────────────────────────────
    def phase_fault(self):
        self.current_phase = "fault"
        ok = True
        name = "ktts-e2e-fault"
        base = f"http://127.0.0.1:{self.port + 2}"
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 2, "G2P_WORKERS": 1,
                "TTS_WORKERS": 2, "ASR_WORKERS": 1,
                "TTS_MAX_BATCH_SIZE": 1, "ASR_MAX_BATCH_SIZE": 1,
                "TTS_CHUNK_CHARS": 60, "TTS_MAX_INPUT_CHARS": 500,
                "WAIT_TIMEOUT_S": 120, "HEARTBEAT_STALE_S": 30,
                "STUCK_KILL_S": 1800, "SPAWN_READY_TIMEOUT_S": 0,
                "WEDGE_RESTART_S": 0,
            }, host_port=self.port + 2)
            self.wait_ready(base, self.args.startup_timeout)

            # 1) parity_smoke：双网关/断连/杀 tts/崩溃计数/重生 ready
            rc, out = self.dexec(name, "python", "scripts/parity_smoke.py",
                                 timeout=1800,
                                 env_extra={"PORT": str(DEFAULT_PORT),
                                            "PARITY_CONCURRENCY": "4",
                                            "PARITY_ABANDON_REPEAT": "4"})
            self.save("parity-smoke.txt", out)
            self.check("parity_smoke（双网关/断连/杀 tts）", rc == 0,
                       out.strip().splitlines()[-1] if out.strip() else "")
            ok = ok and rc == 0

            # 2) SO_REUSEPORT：两个 gateway 绑同一端口，且都分到请求
            rc, ss = self.dexec(name, "sh", "-c", "ss -ltnp | grep ':5174' || true")
            pids = set(re.findall(r"pid=(\d+)", ss))
            self.check("SO_REUSEPORT 两个 gateway 绑同一端口", len(pids) >= 2,
                       f"pids={sorted(pids)}")
            for i in range(20):
                self.http(base, "/v1/audio/speech",
                          {"input": f"分流验证{i}", "voice": "zf_001",
                           "response_format": "pcm"}, timeout=120)
            gws = [w for w in self.heartbeat_summary(name)
                   if w.get("role") == "gateway"]
            reqs = [((w.get("stats") or {}).get("requests") or 0) for w in gws]
            self.check("双网关均分到请求", len(reqs) >= 2 and all(r > 0 for r in reqs),
                       f"gateway requests={reqs}")
            ok = ok and len(pids) >= 2 and len(reqs) >= 2 and all(r > 0 for r in reqs)

            # 3) kill 内嵌 redis -> 重启 -> 恢复服务
            self.kill_redis(name)
            st, _h, _b = self.http(base, "/v1/audio/speech",
                                   {"input": "redis 故障验证。", "voice": "zf_001"},
                                   timeout=10)
            self.check("kill redis 期间请求失败（5xx/超时）",
                       st is None or st >= 500, f"{st}")
            recovered = False
            for _ in range(40):
                rc, out = self.dexec(name, "redis-cli", "-s",
                                     "/tmp/kokoro-redis.sock", "ping")
                if "PONG" in out:
                    recovered = True
                    break
                time.sleep(0.5)
            self.check("supervisor 重启 redis", recovered, "")
            time.sleep(4)
            st, _h, _b = self.http(base, "/v1/audio/speech",
                                   {"input": "redis 恢复验证。", "voice": "zf_001"},
                                   timeout=120)
            self.check("redis 恢复后请求 200", st == 200, f"{st}")
            ok = ok and recovered and st == 200

            # 4) ASR 未就绪 -> /tts 503 -> 心跳重建 -> 200
            asr_pid = self.role_pid(name, "asr")
            if asr_pid:
                # 先 SIGSTOP 再删心跳：否则 asr 心跳线程会在 2.5s 窗口内
                # 重写文件，/tts 又看到 ready（实测复现）。
                self.signal_pid(name, asr_pid, "SIGSTOP")
                self.dexec(name, "sh", "-c",
                           f"rm -f /tmp/kokoro-health/worker-{asr_pid}.json")
                time.sleep(2.5)
                st, _h, body = self.http(base, "/tts",
                                         {"text": "时间戳服务未就绪验证"}, timeout=30)
                ok503 = st == 503 and b"error" in body
                self.check("/tts ASR 未就绪 503", ok503, f"{st} {body[:120]}")
                self.signal_pid(name, asr_pid, "SIGCONT")
                st = None
                for _ in range(20):
                    time.sleep(1)
                    st, _h, _b = self.http(base, "/tts",
                                           {"text": "时间戳服务恢复验证"},
                                           timeout=600)
                    if st == 200:
                        break
                self.check("/tts ASR 心跳恢复后 200", st == 200, f"{st}")
                ok = ok and ok503 and st == 200
            else:
                self.check("/tts ASR 未就绪测试（找不到 asr pid）", False, "")

            # 5) SHM 孤儿清扫（mtime 10 分钟前，心跳清扫周期 5s）
            self.dexec(name, "sh", "-c",
                       "touch /dev/shm/kokora_e2e_orphan && "
                       "touch -d '10 minutes ago' /dev/shm/kokora_e2e_orphan")
            time.sleep(7)
            rc, out = self.dexec(name, "sh", "-c",
                                 "test ! -e /dev/shm/kokora_e2e_orphan && echo gone")
            self.check("SHM 孤儿被周期清扫", "gone" in out, out.strip())
            ok = ok and "gone" in out

            # 6) 优雅退出
            self.docker("stop", "-t", "20", name, timeout=60)
            rc, out = self.docker("inspect", "-f", "{{.State.ExitCode}}", name)
            self.check("docker stop 优雅退出（ExitCode=0）", out.strip() == "0", out.strip())
            ok = ok and out.strip() == "0"
            self.logs(name)
            self.containers.remove(name)
        except Exception as exc:  # noqa: BLE001
            self.check("fault 阶段异常", False, str(exc))
            self.logs(name)
            ok = False

        # 7) 背压/超时/断连（独立小容器）
        ok2 = self._phase_backpressure()
        return ok and ok2

    def _phase_backpressure(self):
        name = "ktts-e2e-bp"
        base = f"http://127.0.0.1:{self.port + 3}"
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 1, "G2P_WORKERS": 1,
                "TTS_WORKERS": 1, "ASR_WORKERS": 1,
                "TTS_MAX_BATCH_SIZE": 1, "ASR_MAX_BATCH_SIZE": 1,
                "TTS_CHUNK_CHARS": 60, "TTS_MAX_INPUT_CHARS": 500,
                # CPU 单次前向 ~10s：预算需大于正常请求耗时，超时门仍有意义
                "WAIT_TIMEOUT_S": 20, "MAX_PENDING_REQUESTS": 1,
                "MAX_INFLIGHT_SENTENCES": 2, "WORKER_MAILBOX_SIZE": 8,
                "HEARTBEAT_STALE_S": 120, "STUCK_KILL_S": 1800,
                "SPAWN_READY_TIMEOUT_S": 0, "WEDGE_RESTART_S": 0,
            }, shm="256m", host_port=self.port + 3)
            self.wait_ready(base, self.args.startup_timeout)
            g2p = self.role_pid(name, "g2p")
            tts = self.role_pid(name, "tts")
            if not self.check("背压容器找到 g2p/tts pid", bool(g2p and tts),
                              f"{g2p} {tts}"):
                return False
            ok = True

            # queue_full：暂停 g2p，队列 cap=1
            self.signal_pid(name, g2p, "SIGSTOP")
            self.http(base, "/v1/audio/speech",
                      {"input": "队列满验证一。", "voice": "zf_001",
                       "response_format": "pcm"}, timeout=30)
            time.sleep(0.5)
            st, h, body = self.http(base, "/v1/audio/speech",
                                    {"input": "队列满验证二。", "voice": "zf_001",
                                     "response_format": "pcm"}, timeout=10)
            qf = st == 429 and h.get("x-reject-reason") == "queue_full" \
                and h.get("retry-after") == "1"
            self.check("429 queue_full + Retry-After/X-Reject-Reason", qf,
                       f"{st} {h.get('x-reject-reason')} {body[:80]}")
            ok = ok and qf
            self.signal_pid(name, g2p, "SIGCONT")
            time.sleep(3)

            # wait_budget：暂停 tts，堆 depth + 注入高 EMA
            self.signal_pid(name, tts, "SIGSTOP")
            self.http(base, "/v1/audio/speech",
                      {"input": "等待预算验证占位。", "voice": "zf_001",
                       "response_format": "pcm"}, timeout=30)
            time.sleep(1.5)
            self.dexec(name, "redis-cli", "-s", "/tmp/kokoro-redis.sock",
                       "SET", "kokoro:q:1:tts:avg", "60000")
            st, h, body = self.http(base, "/v1/audio/speech",
                                    {"input": "等待预算验证拒绝。", "voice": "zf_001",
                                     "response_format": "pcm"}, timeout=10)
            wb = st == 429 and h.get("x-reject-reason") == "wait_budget"
            self.check("429 wait_budget（EMA 预算拒绝）", wb,
                       f"{st} {h.get('x-reject-reason')} {body[:80]}")
            ok = ok and wb
            self.signal_pid(name, tts, "SIGCONT")
            time.sleep(4)

            # shm_pressure：256m tmpfs 填到 >90%
            self.dexec(name, "sh", "-c",
                       "dd if=/dev/zero of=/dev/shm/e2e_fill bs=1M count=240 "
                       "status=none")
            st, h, body = self.http(base, "/v1/audio/speech",
                                    {"input": "共享内存水位验证。", "voice": "zf_001",
                                     "response_format": "pcm"}, timeout=10)
            sp = st == 429 and h.get("x-reject-reason") == "shm_pressure"
            self.check("429 shm_pressure（/dev/shm ≥90%）", sp,
                       f"{st} {h.get('x-reject-reason')} {body[:80]}")
            ok = ok and sp
            self.dexec(name, "sh", "-c", "rm -f /dev/shm/e2e_fill")
            time.sleep(1)

            # timeout 504：暂停 tts
            before = self._metric_delta(base, "kokoro_e2e_timeouts_total")
            self.signal_pid(name, tts, "SIGSTOP")
            st, _h, _b = self.http(base, "/v1/audio/speech",
                                   {"input": "超时验证请求。", "voice": "zf_001",
                                    "response_format": "pcm"}, timeout=30)
            self.check("等待预算超时 504", st == 504, f"{st}")
            self.signal_pid(name, tts, "SIGCONT")
            time.sleep(6)
            after = self._metric_delta(base, "kokoro_e2e_timeouts_total")
            self.check("timeouts 指标递增", after >= before + 1, f"{before}->{after}")
            st = self.wait_200(base, {"input": "超时恢复验证。", "voice": "zf_001",
                                      "response_format": "pcm"})
            self.check("超时后恢复 200", st == 200, f"{st}")
            ok = ok and st == 200 and after >= before + 1

            # 499：客户端 1s 断开，cancel 指标递增
            before = self._metric_delta(base, "kokoro_e2e_cancelled_total")
            self.http(base, "/v1/audio/speech",
                      {"input": "客户端断开验证。" * 10, "voice": "zf_001",
                       "response_format": "pcm"}, timeout=1)
            time.sleep(7)
            after = self._metric_delta(base, "kokoro_e2e_cancelled_total")
            self.check("499 断连 cancel 指标递增", after >= before + 1, f"{before}->{after}")
            st = self.wait_200(base, {"input": "断连恢复验证。", "voice": "zf_001",
                                      "response_format": "pcm"})
            self.check("断连后恢复 200", st == 200, f"{st}")
            ok = ok and after >= before + 1 and st == 200

            self.docker("stop", "-t", "20", name, timeout=60)
            self.logs(name)
            self.containers.remove(name)
            return ok
        except Exception as exc:  # noqa: BLE001
            self.check("背压阶段异常", False, str(exc))
            self.logs(name)
            return False

    def wait_200(self, base, payload, timeout_s=120, request_timeout=180):
        """轮询到 200（负载排空/EMA 衰减需要时间）；返回最后一次状态码。"""
        deadline = time.time() + timeout_s
        st = None
        while time.time() < deadline:
            st, _h, _b = self.http(base, "/v1/audio/speech", payload,
                                   timeout=request_timeout)
            if st == 200:
                return st
            time.sleep(2)
        return st

    def _metric_delta(self, base, name):
        mt = self.metrics(base)
        vals = [float(v) for v in re.findall(rf"^{re.escape(name)}\s+(\S+)$", mt, re.M)]
        return max(vals) if vals else 0.0

    # ── 报告 ──────────────────────────────────────────────────────
    def report(self):
        total = len(self.checks)
        failed = [c for c in self.checks if not c["ok"]]
        summary = {
            "timestamp": self.stamp,
            "image": self.image,
            "assets": self.assets,
            "artifacts": self.out,
            "total": total,
            "passed": total - len(failed),
            "failed": len(failed),
            "checks": self.checks,
        }
        self.save("summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        lines = [f"# CPU E2E {self.stamp}", "",
                 f"- image: `{self.image}`",
                 f"- assets: `{self.assets}`",
                 f"- checks: **{total - len(failed)}/{total} passed**", ""]
        for c in self.checks:
            lines.append(f"- [{'x' if c['ok'] else ' '}] `{c['phase']}` {c['name']}"
                         + (f" — {c['detail']}" if c["detail"] and not c["ok"] else ""))
        self.save("summary.md", "\n".join(lines) + "\n")
        print(f"\n==== CPU E2E: {total - len(failed)}/{total} passed ====")
        print(f"artifacts: {self.out}")
        if failed:
            print("FAILED:", ", ".join(f"{c['phase']}/{c['name']}" for c in failed))
        return not failed

    # ── 入口 ──────────────────────────────────────────────────────
    def run(self):
        os.chdir(ROOT)
        phases = [p.strip() for p in self.args.phases.split(",") if p.strip()]
        try:
            if not self.phase_build():
                self.note("构建/镜像检查失败，后续阶段跳过；报告仍会生成")
                return self.report()
            if "pytest" in phases and not self.phase_pytest():
                self.note("镜像 pytest 失败，继续跑功能阶段以收集更多证据")
            baseline_ok = False
            if "baseline" in phases:
                baseline_ok = self.phase_baseline()
            if "numeric" in phases and baseline_ok:
                self.phase_numeric()
            if "ktts-e2e-baseline" in self.containers:
                self.docker("stop", "-t", "20", "ktts-e2e-baseline", timeout=60)
                self.logs("ktts-e2e-baseline")
                self.containers.remove("ktts-e2e-baseline")
            if "fault" in phases:
                self.phase_fault()
        finally:
            self.cleanup()
        return self.report()


# docker exec 需要额外 -e：包一层 monkey-patch 风格的辅助
def _dexec_with_env(self, container, *args, env_extra=None, timeout=1800,
                    check=False):
    env_extra = env_extra or {}
    cmd = ["docker", "exec"]
    for k, v in env_extra.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += [container, *args]
    return self.sh(cmd, timeout=timeout, check=check)


E2E.dexec = _dexec_with_env


def main():
    ap = argparse.ArgumentParser(description="CPU end-to-end verification")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help="baseline 宿主端口；fault 用 port+2、背压用 port+3")
    ap.add_argument("--phases", default="pytest,baseline,numeric,fault",
                    help="逗号分隔：pytest,baseline,numeric,fault")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--keep", action="store_true", help="失败时保留容器")
    ap.add_argument("--torch-version", default="2.15.0")
    ap.add_argument("--startup-timeout", type=int, default=1800,
                    help="CPU 模型加载 + FunASR 热身等待上限（秒）")
    args = ap.parse_args()

    e2e = E2E(args)
    if not os.path.isdir(os.path.join(e2e.assets, "hf-cache")):
        print(f"缺少模型资产 {e2e.assets}/hf-cache；先运行 "
              f"bash scripts/extract_cpu_assets.sh --out {e2e.assets}",
              file=sys.stderr)
        return 2
    if not os.path.isdir(os.path.join(e2e.assets, "funasr-models")):
        print(f"缺少模型资产 {e2e.assets}/funasr-models", file=sys.stderr)
        return 2
    return 0 if e2e.run() else 1


if __name__ == "__main__":
    _code = main()
    print(f"[cpu_e2e] exit={_code}", flush=True)
    raise SystemExit(_code)

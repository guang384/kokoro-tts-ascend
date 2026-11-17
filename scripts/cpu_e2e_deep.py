#!/usr/bin/env python3
"""CPU 全功能深度验证（在 cpu_e2e.py 基础上做行为级/故障级/配置矩阵扩展）。

阶段：
  contract   编码格式全量(ffprobe)/音色/语速/覆盖/精度效果、中英混合、
             真实批处理、长文本对齐、CORS/keep-alive/404/405、指标不变量、
             ab_quality_test + benchmark
  supervisor 杀 gateway/asr/g2p/tts、stale 心跳击杀、queue bad 注入、
             wedge 真实退出
  config     METRICS_ENABLED=0、EN_G2P=off/espeak、TTS_SPEED_FLOOR、
             MAX_BODY_BYTES、KOKORO_G2P_DICT、IPC_REDIS_URL 外部 Redis

沿用 cpu_e2e.E2E 的容器/http/check/report 基础设施：
  python3 scripts/cpu_e2e_deep.py --phases contract,load
  python3 scripts/cpu_e2e_deep.py --phases supervisor
  python3 scripts/cpu_e2e_deep.py --phases config
  python3 scripts/cpu_e2e_deep.py --phases contract,load,supervisor,config
"""
import argparse
import hashlib
import http.client
import io
import json
import math
import struct
import os
import re
import sys
import time
import urllib.error
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor

from cpu_e2e import COMMON_ENV, DEFAULT_PORT, E2E

DEFAULT_IMAGE = "kokoro-tts-cpu:local"
DEFAULT_ASSETS = "cpu-e2e-assets"


class DeepE2E(E2E):
    # ── 容器扩展：自定义网络（外部 redis 模式） ──────────────────
    def start_net(self, name, env, network, host_port=None, shm="2g"):
        self.remove(name)
        port = host_port or self.port
        cmd = ["run", "-d", "--name", name, "--network", network,
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
            raise RuntimeError(f"docker run(net) 失败: {out[-2000:]}")
        self.containers.append(name)
        self.note(f"容器 {name} 启动（network={network} host_port={port}）")
        return name

    # ── 通用请求 ─────────────────────────────────────────────────
    def post_json(self, base, path, payload):
        return self.http(base, path, payload)

    def post_raw(self, base, path, data, headers, timeout=60):
        req = urllib.request.Request(base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
        except urllib.error.HTTPError as exc:
            return exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read()
        except Exception as exc:  # noqa: BLE001
            return None, {}, repr(exc).encode()

    def synth(self, base, text, fmt="wav", voice=None, speed=1.0,
              dtype=None, overrides=None):
        payload = {"input": text if fmt != "tts" else None,
                   "voice": voice or "zf_001", "speed": speed,
                   "response_format": fmt if fmt != "tts" else "wav"}
        if dtype is not None:
            payload["dtype"] = dtype
        if overrides is not None:
            payload["g2p_overrides"] = overrides
        if fmt == "tts":
            payload.pop("input", None)
            payload.pop("response_format", None)
            payload["text"] = text
        path = "/tts" if fmt == "tts" else "/v1/audio/speech"
        return self.post_json(base, path, payload)

    def probe(self, container, path):
        rc, out = self.dexec(
            container, "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration",
            "-of", "json", path)
        try:
            return json.loads(out)
        except ValueError:
            return {}

    @staticmethod
    def digest(data):
        return hashlib.sha256(data).hexdigest()[:16]

    def metric_line(self, text, name, labels=""):
        pat = rf"^{re.escape(name)}{re.escape(labels)}\s+(\S+)$"
        m = re.search(pat, text, re.M)
        return float(m.group(1)) if m else None

    def role_entries(self, container, role):
        return [w for w in self.heartbeat_summary(container)
                if w.get("role") == role]

    def wait_metric(self, base, needle, predicate, timeout=60):
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            last = self.metrics(base)
            if needle in last and predicate(last):
                return True, last
            time.sleep(1)
        return False, last

    @staticmethod
    def pcm(body):
        with wave.open(io.BytesIO(body)) as w:
            return struct.unpack("<%dh" % w.getnframes(),
                                 w.readframes(w.getnframes()))

    def corr(self, a, b):
        pa, pb = self.pcm(a), self.pcm(b)
        n = min(len(pa), len(pb))
        if n == 0:
            return 0.0
        pa, pb = pa[:n], pb[:n]
        ma, mb = sum(pa) / n, sum(pb) / n
        cov = sum((x - ma) * (y - mb) for x, y in zip(pa, pb))
        va = sum((x - ma) ** 2 for x in pa)
        vb = sum((y - mb) ** 2 for y in pb)
        return cov / math.sqrt(va * vb) if va and vb else 0.0

    # ── Phase contract ───────────────────────────────────────────
    def phase_contract(self):
        self.current_phase = "contract"
        name = "ktts-e2e-deep"
        base = f"http://127.0.0.1:{self.port + 4}"
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 1, "G2P_WORKERS": 1,
                "TTS_WORKERS": 1, "ASR_WORKERS": 1,
                "TTS_MAX_BATCH_SIZE": 4, "ASR_MAX_BATCH_SIZE": 4,
                "TTS_CHUNK_CHARS": 30, "TTS_MAX_INPUT_CHARS": 150,
                "WAIT_TIMEOUT_S": 600, "SPAWN_READY_TIMEOUT_S": 0,
                "STUCK_KILL_S": 1800, "HEARTBEAT_STALE_S": 120,
                "WEDGE_RESTART_S": 0, "KEEP_ALIVE_TIMEOUT_S": 5,
                # 生产 NPU 的 auto 档等价于纯 conv STFT；CPU 默认复数 STFT
                # 不支持 ComplexHalf，显式开启后才能验证 fp16 路径。
                "KOKORO_DISABLE_COMPLEX": 1,
                "KOKORO_EN_G2P": "auto",
            }, host_port=self.port + 4)
            self.wait_ready(base, self.args.startup_timeout)
            ok = True
            ok &= self._contract_formats(name, base)
            ok &= self._contract_effects(base)
            ok &= self._contract_text(base)
            ok &= self._contract_batch(name, base)
            ok &= self._contract_http(base)
            ok &= self._contract_metrics(name, base)
            ok &= self._contract_tools(name, base)
            return ok
        except Exception as exc:  # noqa: BLE001
            self.check("contract 阶段异常", False, str(exc))
            self.logs(name)
            return False

    def _contract_formats(self, name, base):
        ok = True
        text = "编码格式验证。"
        for fmt, ext, codec in (("wav", "wav", "pcm_s16le"),
                                 ("flac", "flac", "flac"),
                                 ("mp3", "mp3", "mp3"),
                                 ("opus", "opus", "opus"),
                                 ("aac", "aac", "aac")):
            st, h, body = self.synth(base, text, fmt=fmt)
            path = self.save(f"contract-fmt.{ext}", body)
            st_ok = st == 200 and len(body) > 200
            self.check(f"/v1 {fmt} 200/非空", st_ok, f"{st} {len(body)}B")
            ok &= st_ok
            if st_ok:
                info = self.probe(name, f"/artifacts/{os.path.basename(path)}")
                streams = info.get("streams") or [{}]
                got = streams[0].get("codec_name")
                dur = float((info.get("format") or {}).get("duration") or 0)
                self.check(f"/v1 {fmt} ffprobe codec={codec}",
                           got == codec and dur > 0, f"{got} dur={dur}")
                ok &= got == codec and dur > 0
                if fmt == "wav":
                    import struct
                    sr = struct.unpack("<I", body[24:28])[0]
                    ch = struct.unpack("<H", body[22:24])[0]
                    bits = struct.unpack("<H", body[34:36])[0]
                    self.check("/v1 wav 24k/mono/16bit",
                               (sr, ch, bits) == (24000, 1, 16),
                               f"{sr}/{ch}/{bits}")
                    ok &= (sr, ch, bits) == (24000, 1, 16)
                    hdr_dur = float(h.get("x-audio-duration") or 0)
                    # 44B 头 + PCM16 mono
                    body_dur = (len(body) - 44) / 2 / 24000
                    self.check("X-Audio-Duration 与 WAV 数据一致",
                               abs(hdr_dur - body_dur) < 0.15,
                               f"{hdr_dur} vs {body_dur:.2f}")
                    ok &= abs(hdr_dur - body_dur) < 0.15
        st, h, body = self.synth(base, text, fmt="pcm")
        hdr = float(h.get("x-audio-duration") or 0)
        pcm_ok = st == 200 and abs(len(body) - hdr * 24000 * 2) < 2400
        self.check("/v1 pcm 长度=时长×24k×2", pcm_ok,
                   f"{st} {len(body)} vs {hdr * 24000 * 2:.0f}")
        ok &= pcm_ok
        return ok

    def _contract_effects(self, base):
        ok = True
        text = "欢迎使用语音合成服务。"
        st, _h, body = self.synth(base, text)
        self.check("/v1 默认合成 200", st == 200, f"{st}")
        ok &= st == 200
        # 同请求重复：CPU torch 卷积/FFT 非逐位确定，按波形相关性判定
        st2, _h, body2 = self.synth(base, text)
        c = self.corr(body, body2) if st2 == 200 else 0.0
        same = st2 == 200 and c >= 0.98
        self.check("同请求重复输出高度一致（corr≥0.98）", same,
                   f"{st2} corr={c:.4f}")
        ok &= same
        # 音色
        st, _h, voices_body = self.http(base, "/v1/voices")
        voices = json.loads(voices_body).get("voices", []) if st == 200 else []
        other = next((v for v in voices if v != "zf_001"), None)
        st, _h, body_v = self.synth(base, text, voice=other)
        c = self.corr(body, body_v) if st == 200 else 1.0
        diff_voice = st == 200 and c < 0.7
        self.check("不同音色输出显著不同（corr<0.7）", diff_voice,
                   f"voice={other} {st} corr={c:.3f}")
        ok &= diff_voice
        # 语速
        durs = {}
        for sp in (0.5, 1.0, 2.0, 4.0):
            st, h, _b = self.synth(base, text, speed=sp)
            durs[sp] = float(h.get("x-audio-duration") or 0)
            self.check(f"speed={sp} 200", st == 200, f"{st}")
            ok &= st == 200
        ratio = durs[0.5] / durs[2.0] if durs[2.0] else 0
        self.check("语速 0.5 时长约为 2.0 的 4 倍", ratio > 2.5, f"{ratio:.2f}")
        ok &= ratio > 2.5
        st, h, _b = self.synth(base, text, speed=0.1)
        clamped = abs(float(h.get("x-audio-duration") or 0) - durs[0.5]) < 0.15
        self.check("speed=0.1 钳制到 0.5", st == 200 and clamped,
                   f"{st} dur={h.get('x-audio-duration')}")
        ok &= st == 200 and clamped
        # per-request 多音字覆盖
        st, _h, body_o = self.synth(base, "欢迎长乐平台",
                                    overrides={"长乐": "zhang3 le4"})
        st0, _h, body_d = self.synth(base, "欢迎长乐平台")
        self.save("contract-g2p-default.wav", body_d)
        self.save("contract-g2p-override.wav", body_o)
        c = self.corr(body_o, body_d) if st == 200 and st0 == 200 else 1.0
        override_diff = st == 200 and st0 == 200 and c < 0.8
        self.check("g2p_overrides 显著改变发音（corr<0.8）", override_diff,
                   f"{st}/{st0} corr={c:.3f}")
        ok &= override_diff
        # dtype fp32/fp16
        st32, h32, b32 = self.synth(base, text, dtype="fp32")
        st16, h16, b16 = self.synth(base, text, dtype="fp16")
        d32 = float(h32.get("x-audio-duration") or 0)
        d16 = float(h16.get("x-audio-duration") or 0)
        self.check("dtype=fp32 200", st32 == 200 and d32 > 0, f"{st32} {d32}")
        ok &= st32 == 200 and d32 > 0
        if st16 == 200:
            c = self.corr(b32, b16)
            dtype_ok = abs(d32 - d16) <= max(0.2, d32 * 0.15) and c >= 0.9
            self.check("dtype=fp16 200 且与 fp32 高度接近", dtype_ok,
                       f"{st16} d={d32}/{d16} corr={c:.3f}")
            ok &= dtype_ok
            found, mt = self.wait_metric(
                base, "kokoro_worker_fp16_batches_total",
                lambda t: (self.metric_line(
                    t, "kokoro_worker_fp16_batches_total",
                    '{worker="tts-0"}') or 0) >= 1, 20)
            self.check("fp16 批计数指标", found, "")
            ok &= found
        else:
            # CPU torch 无 ComplexHalf 算子：要求优雅失败且不拖垮服务
            self.check("CPU fp16 优雅失败（非 200 且无崩溃）",
                       st16 in (400, 422, 500, 502), f"{st16}")
            st_after, _h, _b = self.synth(base, text, dtype="fp32")
            healthy = st_after == 200 and json.loads(
                self.http(base, "/health")[2]).get("ready") is True
            self.check("fp16 失败后 fp32/健康不受影响", healthy,
                       f"fp32={st_after}")
            ok &= st16 in (400, 422, 500, 502) and healthy
        return ok

    def _contract_text(self, base):
        ok = True
        # 输入边界
        for label, payload, allowed in (
                ("/tts 空文本", {"text": ""}, (400, 422)),
                ("/tts 空白文本", {"text": "   "}, (400,)),
                ("/tts 纯标点", {"text": "。。。！？"}, (200, 400)),
                ("/tts 超限 151 字", {"text": "长" * 151}, (400,)),
                ("/v1 空输入", {"input": ""}, (422,))):
            path = "/v1/audio/speech" if label.startswith("/v1") else "/tts"
            st, _h, _b = self.post_json(base, path, payload)
            self.check(label, st in allowed, f"{st}")
            ok &= st in allowed
        # 中英混合（spacy 路径）
        mixed = "Hello 世界 123 Kokoro 语音合成。"
        st, _h, body = self.http(base, "/tts", {"text": mixed, "speed": 1.0},
                                 timeout=600)
        if st == 200:
            d = json.loads(body)
            n_align = sum(1 for c in mixed
                          if (c.isascii() and c.isalnum())
                          or "\u4e00" <= c <= "\u9fff")
            ts = d.get("timestamp") or []
            self.check("中英混合时间戳段数=可对齐字符数",
                       len(ts) == n_align, f"ts={len(ts)} align={n_align}")
            ok &= len(ts) == n_align
        else:
            self.check("中英混合 /tts 200", False, f"{st}")
            ok = False
        # 长文本：120 字 / chunk=30 → 多块，ASR 多批
        long_text = "语音合成多批次对齐验证。" * 10
        st, h, body = self.http(base, "/v1/audio/speech",
                                {"input": long_text, "voice": "zf_001",
                                 "response_format": "pcm"}, timeout=900)
        batch = int(h.get("x-batch-size") or 0)
        self.check("长文本 /v1 200 + X-Batch-Size≥1",
                   st == 200 and batch >= 1, f"{st} batch={batch}")
        ok &= st == 200 and batch >= 1
        st, _h, body = self.http(base, "/tts",
                                 {"text": long_text, "speed": 1.0}, timeout=900)
        good = False
        if st == 200:
            d = json.loads(body)
            ts = d.get("timestamp") or []
            n = sum(1 for c in long_text
                    if (c.isascii() and c.isalnum())
                    or "\u4e00" <= c <= "\u9fff")
            good = (set(d) == {"audio_base64", "code", "duration", "message",
                               "text", "timestamp"}
                    and d.get("code") == 0 and len(ts) == n
                    and all(e >= s for s, e in ts)
                    and ts and ts[-1][1] <= d["duration"] * 1000 + 200)
            self.save("contract-long-tts.json",
                      json.dumps(d, ensure_ascii=False))
        self.check("长文本 /tts 时间戳契约", good, f"{st}")
        ok &= good
        return ok

    def _contract_batch(self, name, base):
        ok = True
        # 并发 /v1：真实 TTS 批宽 >1（日志 n=）
        def one(i):
            return self.synth(base, f"并发批处理验证第{i}句。")[0]

        with ThreadPoolExecutor(max_workers=8) as ex:
            codes = list(ex.map(one, range(8)))
        self.check("并发 8×/v1 全部 200", all(c == 200 for c in codes), str(codes))
        ok &= all(c == 200 for c in codes)
        log = self.logs(name)
        widths = [int(x) for x in re.findall(r"\bbid=\S+ n=(\d+)", log)]
        self.check("TTS 实际批宽 >1（日志 n=）", any(w >= 2 for w in widths),
                   f"batch widths={sorted(set(widths))}")

        # 并发 /tts：ASR 批宽
        def one_tts(i):
            st, _h, _b = self.http(base, "/tts",
                                   {"text": f"对齐批处理验证第{i}句。"},
                                   timeout=900)
            return st

        with ThreadPoolExecutor(max_workers=4) as ex:
            codes = list(ex.map(one_tts, range(4)))
        self.check("并发 4×/tts 全部 200", all(c == 200 for c in codes), str(codes))
        ok &= all(c == 200 for c in codes)
        log = self.logs(name)
        asr_widths = [int(x) for x in re.findall(r"asr-\d+.*?n=(\d+)", log)]
        self.check("ASR 实际批宽 >1（日志 n=）",
                   any(w >= 2 for w in asr_widths),
                   f"asr widths={sorted(set(asr_widths))}")
        found, mt = self.wait_metric(
            base, 'kokoro_worker_batch_effective{role="asr"',
            lambda t: "4.0" in t.split('kokoro_worker_batch_effective{role="asr"')[-1][:40],
            20)
        self.check("ASR batch_effective=4", found, "")
        ok &= found
        return ok

    def _contract_http(self, base):
        ok = True
        # CORS 预检
        req = urllib.request.Request(
            base + "/v1/audio/speech",
            headers={"Origin": "http://example.test",
                     "Access-Control-Request-Method": "POST"})
        req.method = "OPTIONS"
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                cors = (r.status == 200
                        and r.headers.get("access-control-allow-origin") in ("*", "http://example.test"))
        except urllib.error.HTTPError as exc:
            cors = exc.code == 200
        self.check("CORS 预检通过", cors, "")
        ok &= cors
        # 405 / 404
        st, _h, _b = self.http(base, "/v1/audio/speech")   # GET
        self.check("GET /v1/audio/speech → 405", st == 405, f"{st}")
        ok &= st == 405
        st, _h, _b = self.http(base, "/no-such-route")
        self.check("未知路由 → 404", st == 404, f"{st}")
        ok &= st == 404
        # urlencoded Content-Type 归一（JSON body 仍可解析）
        body = json.dumps({"text": "内容类型归一验证。"}).encode()
        st, _h, _b = self.post_raw(
            base, "/tts", body,
            {"Content-Type": "application/x-www-form-urlencoded"})
        self.check("urlencoded Content-Type 归一", st == 200, f"{st}")
        ok &= st == 200
        # keep-alive：同一连接连续两个请求
        host = "127.0.0.1"
        conn = http.client.HTTPConnection(host, self.port + 4, timeout=120)
        try:
            keep_ok = True
            for i in range(2):
                payload = json.dumps({"input": f"keepalive {i}。"}).encode()
                conn.request("POST", "/v1/audio/speech", body=payload,
                             headers={"Content-Type": "application/json",
                                      "Connection": "keep-alive"})
                resp = conn.getresponse()
                resp.read()
                keep_ok &= resp.status == 200
        except Exception as exc:  # noqa: BLE001
            keep_ok = False
            self.note(f"keep-alive 异常: {exc}")
        finally:
            conn.close()
        self.check("HTTP keep-alive 同连接两次 200", keep_ok, "")
        ok &= keep_ok
        return ok

    def _contract_metrics(self, name, base):
        ok = True
        time.sleep(6)   # 心跳落盘
        mt = self.metrics(base)
        self.save("contract-metrics.txt", mt)
        self.check("/health status=ok", json.loads(
            self.http(base, "/health")[2]).get("status") == "ok", "")
        for fam in ("kokoro_e2e_requests_total", "kokoro_e2e_successes_total",
                    "kokoro_queue_depth", "kokoro_worker_done_total",
                    "kokoro_request_duration_seconds_count",
                    "kokoro_worker_batch_effective", "kokoro_build_info"):
            self.check(f"/metrics 含 {fam}", fam in mt, "")
            ok &= fam in mt
        succ = self.metric_line(mt, "kokoro_e2e_successes_total") or 0
        hist = self.metric_line(mt, "kokoro_request_duration_seconds_count") or 0
        self.check("直方图 count == e2e successes",
                   abs(succ - hist) <= 1, f"{succ} vs {hist}")
        ok &= abs(succ - hist) <= 1
        depths = re.findall(r'^kokoro_queue_depth\{stage="([^"]+)"\}\s+(\S+)$',
                            mt, re.M)
        self.check("空闲队列 depth 全 0", depths and all(float(v) == 0 for _s, v in depths),
                   str(depths))
        ok &= depths and all(float(v) == 0 for _s, v in depths)
        shm_pct = self.metric_line(mt, "kokoro_shm_pct")
        self.check("SHM 水位 <80%", shm_pct is not None and shm_pct < 80,
                   f"{shm_pct}")
        ok &= shm_pct is not None and shm_pct < 80
        rc, out = self.dexec(name, "sh", "-c",
                             "ls /dev/shm 2>/dev/null | grep -c '^kokora_' || true")
        self.check("空闲无残留音频 SHM 块", out.strip() in ("0", ""), out.strip())
        ok &= out.strip() in ("0", "")
        return ok

    def _contract_tools(self, name, base):
        ok = True
        internal = f"http://127.0.0.1:{DEFAULT_PORT}"   # dexec 在容器内执行
        rc, out = self.dexec(
            name, "python", "scripts/benchmark.py", "--url", internal,
            "--endpoint", "speech", "-c", "2", "-n", "4",
            "--random", "4", "--format", "wav", timeout=1800)
        self.save("contract-benchmark.txt", out)
        self.check("benchmark.py 运行完成", rc == 0,
                   out.strip().splitlines()[-1] if out.strip() else "")
        ok &= rc == 0
        rc, out = self.dexec(
            name, "python", "scripts/ab_quality_test.py", "--base", internal,
            "--out", "/artifacts/ab-quality.html", "--wait-s", "600",
            timeout=2400)
        self.save("contract-ab-quality.txt", out)
        exists = os.path.isfile(os.path.join(self.out, "ab-quality.html"))
        # CPU 无 ComplexHalf：脚本按设计 exit 2 放弃 fp16；NPU 上应 exit 0
        cpu_fp16_limited = rc == 2 and "fp16" in out.lower()
        self.check("ab_quality_test.py（CPU fp16 不支持时按设计放弃）",
                   (rc == 0 and exists) or cpu_fp16_limited,
                   out.strip().splitlines()[-1] if out.strip() else "")
        ok &= (rc == 0 and exists) or cpu_fp16_limited
        return ok

    # ── Phase supervisor ─────────────────────────────────────────
    def phase_supervisor(self):
        self.current_phase = "supervisor"
        name = "ktts-e2e-sup"
        base = f"http://127.0.0.1:{self.port + 5}"
        ok = True
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 2, "G2P_WORKERS": 1,
                "TTS_WORKERS": 2, "ASR_WORKERS": 2,
                "TTS_MAX_BATCH_SIZE": 1, "ASR_MAX_BATCH_SIZE": 1,
                "TTS_CHUNK_CHARS": 60, "TTS_MAX_INPUT_CHARS": 200,
                "WAIT_TIMEOUT_S": 120, "HEARTBEAT_STALE_S": 10,
                "STUCK_KILL_S": 1800, "SPAWN_READY_TIMEOUT_S": 0,
                "WEDGE_RESTART_S": 0,
            }, host_port=self.port + 5)
            self.wait_ready(base, self.args.startup_timeout)
            ok &= self._sup_kill_gateway(name, base)
            ok &= self._sup_kill_asr(name, base)
            ok &= self._sup_kill_g2p(name, base)
            ok &= self._sup_stale_tts(name, base)
            ok &= self._sup_queue_bad(name, base)
            self.docker("stop", "-t", "20", name, timeout=60)
            self.logs(name)
            if not self.args.keep:
                self.remove(name)
            self.containers.remove(name)
        except Exception as exc:  # noqa: BLE001
            self.check("supervisor 阶段异常", False, str(exc))
            self.logs(name)
            ok = False
        ok &= self._sup_wedge()
        return ok

    def _crash_total(self, base):
        mt = self.metrics(base)
        vals = [float(v) for v in re.findall(
            r"^kokoro_worker_crashes_total\{[^}]*\}\s+(\S+)$", mt, re.M)]
        return sum(vals)

    def _sup_kill_gateway(self, name, base):
        gws = self.role_entries(name, "gateway")
        if not gws:
            return self.check("supervisor: 找到 gateway 心跳", False, "")
        before = self._crash_total(base)
        victim = gws[0]
        self.signal_pid(name, victim["pid"], "SIGKILL")
        time.sleep(2)
        st, _h, _b = self.http(base, "/v1/audio/speech",
                               {"input": "网关重生验证。", "voice": "zf_001",
                                "response_format": "pcm"}, timeout=120)
        alive = st == 200
        recovered = False
        for _ in range(40):
            if len(self.role_entries(name, "gateway")) >= 2:
                recovered = True
                break
            time.sleep(1)
        time.sleep(6)
        after = self._crash_total(base)
        self.check("杀 gateway：另一入口继续服务", alive, f"{st}")
        self.check("杀 gateway：supervisor 重生", recovered,
                   f"gateways={len(self.role_entries(name, 'gateway'))}")
        self.check("杀 gateway：崩溃计数递增", after > before, f"{before}->{after}")
        return alive and recovered and after > before

    def _sup_kill_asr(self, name, base):
        asrs = self.role_entries(name, "asr")
        if not asrs:
            return self.check("supervisor: 找到 asr 心跳", False, "")
        before = self._crash_total(base)
        self.signal_pid(name, asrs[0]["pid"], "SIGKILL")
        st_v, _h, _b = self.http(base, "/v1/audio/speech",
                                 {"input": "ASR 故障期间 /v1。", "voice": "zf_001",
                                  "response_format": "pcm"}, timeout=120)
        recovered = False
        tts_ok = False
        for _ in range(60):
            if len(self.role_entries(name, "asr")) >= 2:
                recovered = True
            if recovered:
                st, _h, _b = self.http(base, "/tts", {"text": "ASR 重生验证。"},
                                       timeout=300)
                if st == 200:
                    tts_ok = True
                    break
            time.sleep(2)
        time.sleep(6)
        after = self._crash_total(base)
        self.check("杀 asr：/v1 不中断", st_v == 200, f"{st_v}")
        self.check("杀 asr：池重生", recovered, "")
        self.check("杀 asr：/tts 恢复 200", tts_ok, "")
        self.check("杀 asr：崩溃计数递增", after > before, f"{before}->{after}")
        return st_v == 200 and recovered and tts_ok and after > before

    def _sup_kill_g2p(self, name, base):
        g2ps = self.role_entries(name, "g2p")
        if not g2ps:
            return self.check("supervisor: 找到 g2p 心跳", False, "")
        before = self._crash_total(base)
        self.signal_pid(name, g2ps[0]["pid"], "SIGKILL")
        recovered = False
        for _ in range(60):
            if len(self.role_entries(name, "g2p")) >= 1:
                st, _h, _b = self.http(base, "/v1/audio/speech",
                                       {"input": "G2P 重生验证。", "voice": "zf_001",
                                        "response_format": "pcm"}, timeout=180)
                if st == 200:
                    recovered = True
                    break
            time.sleep(2)
        time.sleep(6)
        after = self._crash_total(base)
        self.check("杀 g2p：池重生且 /v1 恢复 200", recovered, "")
        self.check("杀 g2p：崩溃计数递增", after > before, f"{before}->{after}")
        return recovered and after > before

    def _sup_stale_tts(self, name, base):
        ttss = self.role_entries(name, "tts")
        if not ttss:
            return self.check("supervisor: 找到 tts 心跳", False, "")
        victim = ttss[0]
        before = self._crash_total(base)
        self.signal_pid(name, victim["pid"], "SIGSTOP")   # 心跳停更 → stale 击杀
        dead = False
        for _ in range(40):
            now = self.role_entries(name, "tts")
            pids = {w.get("pid") for w in now}
            if victim["pid"] not in pids or len(now) < 2:
                dead = True
                break
            time.sleep(1)
        time.sleep(6)
        after = self._crash_total(base)
        st, _h, _b = self.http(base, "/v1/audio/speech",
                               {"input": "stale worker 验证。", "voice": "zf_001",
                                "response_format": "pcm"}, timeout=180)
        self.check("SIGSTOP tts → 心跳 stale 被杀/重生", dead, "")
        self.check("stale 击杀崩溃计数递增", after > before, f"{before}->{after}")
        self.check("stale 后服务仍 200", st == 200, f"{st}")
        return dead and after > before and st == 200

    def _sup_queue_bad(self, name, base):
        before_mt = self.metrics(base)
        before = self.metric_line(
            before_mt, "kokoro_queue_bad_total", '{stage="tts"}') or 0
        self.dexec(name, "redis-cli", "-s", "/tmp/kokoro-redis.sock",
                   "ZADD", "kokoro:q:1:tts", "1", "shortbad")
        found = False
        for _ in range(20):
            time.sleep(1)
            mt = self.metrics(base)
            val = self.metric_line(mt, "kokoro_queue_bad_total",
                                   '{stage="tts"}') or 0
            if val > before:
                found = True
                break
        st, _h, _b = self.http(base, "/v1/audio/speech",
                               {"input": "毒块后服务验证。", "voice": "zf_001",
                                "response_format": "pcm"}, timeout=180)
        self.check("queue bad 注入被丢弃并计数", found, f"{before}->")
        self.check("毒块后服务 200", st == 200, f"{st}")
        return found and st == 200

    def _sup_wedge(self):
        name = "ktts-e2e-wedge"
        base = f"http://127.0.0.1:{self.port + 6}"
        try:
            self.start(name, {
                "GATEWAY_WORKERS": 1, "G2P_WORKERS": 1,
                "TTS_WORKERS": 1, "ASR_WORKERS": 1,
                "TTS_MAX_BATCH_SIZE": 1, "ASR_MAX_BATCH_SIZE": 1,
                "WAIT_TIMEOUT_S": 3, "WEDGE_RESTART_S": 10,
                "HEARTBEAT_STALE_S": 120, "STUCK_KILL_S": 1800,
                "SPAWN_READY_TIMEOUT_S": 0,
            }, host_port=self.port + 6)
            self.wait_ready(base, self.args.startup_timeout)
            tts = self.role_pid(name, "tts")
            self.signal_pid(name, tts, "SIGSTOP")
            for _ in range(3):
                self.http(base, "/v1/audio/speech",
                          {"input": "wedge 验证。", "voice": "zf_001",
                           "response_format": "pcm"}, timeout=10)
            exited = False
            code = None
            for _ in range(40):
                rc, out = self.docker("inspect", "-f",
                                      "{{.State.Running}} {{.State.ExitCode}}",
                                      name)
                if out.strip().startswith("false"):
                    exited = True
                    code = out.strip().split()[-1]
                    break
                time.sleep(1)
            log = self.logs(name)
            self.check("wedge 真实退出", exited and code == "1",
                       f"running_exit={out.strip()}")
            self.check("wedge 日志留痕", "wedge" in log.lower(), "")
            if not self.args.keep:
                self.remove(name)
            if name in self.containers:
                self.containers.remove(name)
            return exited and code == "1" and "wedge" in log.lower()
        except Exception as exc:  # noqa: BLE001
            self.check("wedge 阶段异常", False, str(exc))
            self.logs(name)
            return False

    # ── Phase config ─────────────────────────────────────────────
    def phase_config(self):
        self.current_phase = "config"
        ok = True
        ok &= self._cfg_metrics_off()
        ok &= self._cfg_en_off()
        ok &= self._cfg_en_espeak()
        ok &= self._cfg_speed_floor()
        ok &= self._cfg_body_limit()
        ok &= self._cfg_g2p_dict()
        ok &= self._cfg_external_redis()
        return ok

    def _one_container(self, name, env, host_off, fn, shm="2g"):
        base = f"http://127.0.0.1:{self.port + host_off}"
        try:
            self.start(name, env, shm=shm, host_port=self.port + host_off)
            self.wait_ready(base, self.args.startup_timeout)
            return fn(name, base)
        except Exception as exc:  # noqa: BLE001
            self.check(f"{name} 异常", False, str(exc))
            self.logs(name)
            return False
        finally:
            self.docker("stop", "-t", "20", name, timeout=60)
            self.logs(name)
            if not self.args.keep and name in self.containers:
                self.remove(name)
            if name in self.containers:
                self.containers.remove(name)

    def _cfg_metrics_off(self):
        def fn(name, base):
            st_m, _h, _b = self.http(base, "/metrics")
            st_h, _h, _b = self.http(base, "/health")
            st_v, _h, _b = self.http(base, "/v1/audio/speech",
                                     {"input": "metrics 关闭验证。",
                                      "voice": "zf_001",
                                      "response_format": "pcm"}, timeout=180)
            self.check("METRICS_ENABLED=0 → /metrics 404", st_m == 404, f"{st_m}")
            self.check("metrics 关闭时 /health 正常", st_h == 200, f"{st_h}")
            self.check("metrics 关闭时 /v1 正常", st_v == 200, f"{st_v}")
            return st_m == 404 and st_h == 200 and st_v == 200
        return self._one_container("ktts-e2e-cfg-metrics", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "METRICS_ENABLED": 0,
        }, 7, fn)

    def _cfg_en_off(self):
        def fn(name, base):
            st_c, _h, _b = self.http(base, "/tts", {"text": "中文正常。"},
                                     timeout=300)
            st_e, _h, _b = self.http(base, "/tts", {"text": "Hello world"},
                                     timeout=60)
            st_m, _h, _b = self.http(base, "/tts",
                                     {"text": "Hello 世界"}, timeout=300)
            self.check("EN_G2P=off：中文 200", st_c == 200, f"{st_c}")
            self.check("EN_G2P=off：纯英文不 5xx", st_e is not None and st_e < 500,
                       f"{st_e}")
            self.check("EN_G2P=off：混合不 5xx", st_m is not None and st_m < 500,
                       f"{st_m}")
            return st_c == 200 and st_e is not None and st_e < 500 \
                and st_m is not None and st_m < 500
        return self._one_container("ktts-e2e-cfg-enoff", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "KOKORO_EN_G2P": "off",
        }, 8, fn)

    def _cfg_en_espeak(self):
        def fn(name, base):
            st, _h, _b = self.http(base, "/tts",
                                   {"text": "Hello 世界 Kokoro"}, timeout=300)
            self.check("EN_G2P=espeak：混合 200", st == 200, f"{st}")
            return st == 200
        return self._one_container("ktts-e2e-cfg-espeak", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "KOKORO_EN_G2P": "espeak",
        }, 9, fn)

    def _cfg_speed_floor(self):
        def fn(name, base):
            st1, h1, _b = self.synth(base, "语速下限验证。", speed=0.5)
            st2, h2, _b = self.synth(base, "语速下限验证。", speed=1.0)
            d1 = float(h1.get("x-audio-duration") or 0)
            d2 = float(h2.get("x-audio-duration") or 0)
            ratio = d1 / d2 if d2 else 0
            good = st1 == 200 and st2 == 200 and 0.85 <= ratio <= 1.15
            self.check("TTS_SPEED_FLOOR=1.0 时 speed=0.5 被钳制", good,
                       f"{st1}/{st2} ratio={ratio:.2f}")
            return good
        return self._one_container("ktts-e2e-cfg-floor", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "TTS_SPEED_FLOOR": 1.0,
        }, 10, fn)

    def _cfg_body_limit(self):
        def fn(name, base):
            ok_pad = "x" * (60 * 1024)
            big_pad = "x" * (70 * 1024)
            st_ok, _h, _b = self.post_json(
                base, "/v1/audio/speech",
                {"input": "body 上限内。", "pad": ok_pad})
            st_big, _h, _b = self.post_json(
                base, "/v1/audio/speech",
                {"input": "body 超限。", "pad": big_pad})
            self.check("MAX_BODY_BYTES=64KiB：60KiB body 通过", st_ok == 200,
                       f"{st_ok}")
            self.check("MAX_BODY_BYTES=64KiB：70KiB body 413", st_big == 413,
                       f"{st_big}")
            return st_ok == 200 and st_big == 413
        return self._one_container("ktts-e2e-cfg-body", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "MAX_BODY_BYTES": 65536,
        }, 11, fn)

    def _cfg_g2p_dict(self):
        text = "欢迎长乐平台"
        mapping = {"长乐": "zhang3 le4"}

        def fn(name, base):
            # 全局词典生效 ⇔ 默认请求与同映射的请求级覆盖发音一致
            st0, _h, body0 = self.synth(base, text)
            st1, _h, body1 = self.synth(base, text, overrides=mapping)
            c = self.corr(body0, body1) if st0 == 200 and st1 == 200 else 0.0
            good = st0 == 200 and st1 == 200 and c >= 0.95
            self.save("config-g2p-dict-default.wav", body0)
            self.check("KOKORO_G2P_DICT 全局词典生效（与同映射覆盖一致）",
                       good, f"{st0}/{st1} corr={c:.3f}")
            return good
        return self._one_container("ktts-e2e-cfg-dict", {
            "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
            "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
            "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
            "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
            "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
            "KOKORO_DISABLE_COMPLEX": 1,
            "KOKORO_G2P_DICT": "长乐 zhang3 le4",
        }, 12, fn)

    def _cfg_external_redis(self):
        name = "ktts-e2e-cfg-extredis"
        redis_name = "ktts-e2e-redis"
        net = "ktts-e2e-net"
        base = f"http://127.0.0.1:{self.port + 13}"
        try:
            self.docker("network", "rm", net, timeout=60)
            self.docker("network", "create", net, check=True)
            self.docker("rm", "-f", redis_name)
            rc, out = self.docker(
                "run", "-d", "--name", redis_name, "--network", net,
                self.image, "redis-server", "--bind", "0.0.0.0",
                "--port", "6379", "--protected-mode", "no",
                "--save", "", "--appendonly", "no")
            self.check("外部 redis 启动", rc == 0, out[-200:])
            self.start_net(name, {
                "GATEWAY_WORKERS": 1, "TTS_WORKERS": 1, "ASR_WORKERS": 1,
                "G2P_WORKERS": 1, "TTS_MAX_BATCH_SIZE": 1,
                "ASR_MAX_BATCH_SIZE": 1, "TTS_MAX_INPUT_CHARS": 100,
                "WAIT_TIMEOUT_S": 300, "SPAWN_READY_TIMEOUT_S": 0,
                "STUCK_KILL_S": 1800, "WEDGE_RESTART_S": 0,
                "IPC_REDIS_URL": f"tcp://{redis_name}:6379",
            }, net, host_port=self.port + 13)
            self.wait_ready(base, self.args.startup_timeout)
            st, _h, _b = self.http(base, "/v1/audio/speech",
                                   {"input": "外部 redis 验证。", "voice": "zf_001",
                                    "response_format": "pcm"}, timeout=180)
            self.check("IPC_REDIS_URL 模式服务正常", st == 200, f"{st}")
            rc, top = self.docker("top", name, "-eo", "args")
            no_embedded = "redis-server" not in top or redis_name in top
            self.check("外部模式不拉起内嵌 redis", no_embedded, "")
            self.docker("rm", "-f", redis_name)
            time.sleep(1)
            st_fail, _h, _b = self.http(base, "/v1/audio/speech",
                                        {"input": "redis 停机验证。",
                                         "voice": "zf_001",
                                         "response_format": "pcm"}, timeout=10)
            self.check("外部 redis 停机 → 请求失败",
                       st_fail is None or st_fail >= 500, f"{st_fail}")
            time.sleep(4)
            rc, out = self.docker("run", "-d", "--name", redis_name,
                                  "--network", net, self.image,
                                  "redis-server", "--bind", "0.0.0.0",
                                  "--port", "6379", "--protected-mode", "no",
                                  "--save", "", "--appendonly", "no")
            recovered = False
            for _ in range(30):
                st, _h, _b = self.http(base, "/v1/audio/speech",
                                       {"input": "外部 redis 恢复验证。",
                                        "voice": "zf_001",
                                        "response_format": "pcm"}, timeout=120)
                if st == 200:
                    recovered = True
                    break
                time.sleep(2)
            self.check("外部 redis 恢复后服务 200", recovered, "")
            return st == 200 and no_embedded and recovered
        except Exception as exc:  # noqa: BLE001
            self.check("外部 redis 阶段异常", False, str(exc))
            self.logs(name)
            return False
        finally:
            self.docker("stop", "-t", "20", name, timeout=60)
            self.logs(name)
            if name in self.containers:
                if not self.args.keep:
                    self.remove(name)
                self.containers.remove(name)
            self.docker("rm", "-f", redis_name)
            self.docker("network", "rm", net)

    # ── 入口 ─────────────────────────────────────────────────────
    def run_deep(self, phases):
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            if not self.phase_build():
                return self.report()
            if "contract" in phases:
                self.phase_contract()
                if not self.args.keep:
                    self.remove("ktts-e2e-deep")
                if "ktts-e2e-deep" in self.containers:
                    self.containers.remove("ktts-e2e-deep")
            if "supervisor" in phases:
                self.phase_supervisor()
            if "config" in phases:
                self.phase_config()
        finally:
            self.cleanup()
        return self.report()


def main():
    ap = argparse.ArgumentParser(description="Deep CPU functional verification")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--assets", default=DEFAULT_ASSETS)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--phases", default="contract,supervisor,config",
                    help="逗号分隔：contract,supervisor,config")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--torch-version", default="2.15.0")
    ap.add_argument("--startup-timeout", type=int, default=1800)
    args = ap.parse_args()
    e2e = DeepE2E(args)
    if not os.path.isdir(os.path.join(e2e.assets, "hf-cache")):
        print(f"缺少模型资产 {e2e.assets}/hf-cache", file=sys.stderr)
        return 2
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    code = 0 if e2e.run_deep(phases) else 1
    print(f"[cpu_e2e_deep] exit={code}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

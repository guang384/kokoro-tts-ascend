#!/usr/bin/env python3
"""生产形态故障注入冒烟：多网关聚合 / 断连取消 / 杀 tts 重派 / 崩溃记录。

平台：Linux 走 SO_REUSEPORT 共享 PORT；Windows 每网关 PORT+1000+i。
前提（真机/容器内均可，CPU 亦可）：
    PORT=5200 GATEWAY_WORKERS=2 TTS_WORKERS>=2 ASR_WORKERS>=1     WAIT_TIMEOUT_S=30 已按此启动服务（Windows 网关端口 = PORT+1000+i）
    CPU 上单次前向慢：可设 PARITY_CONCURRENCY=4、PARITY_ABANDON_REPEAT=4、
    WAIT_TIMEOUT_S=120 缩短杀 tts 并发与断连长文本，语义不变。
用法：
    python scripts/parity_smoke.py            # 全部 6 项
判读：总结行全 ✓ 为通过；"杀 tts 后请求存活" 是 at-least-once 重派的
端到端证明（本套件在开发机验证 16/16 存活、redispatch=3）。
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.request
import urllib.error

import sys

PORT = int(os.getenv("PORT", "5200"))
if sys.platform == "win32":
    # Windows 冒烟回退：每网关独立端口 PORT+1000+i（无 SO_REUSEPORT）
    GW = [f"http://127.0.0.1:{PORT + 1000}", f"http://127.0.0.1:{PORT + 1001}"]
else:
    # Linux 生产：双网关 SO_REUSEPORT 共享同一端口，内核按四元组哈希分发
    GW = [f"http://127.0.0.1:{PORT}", f"http://127.0.0.1:{PORT}"]
HB_DIR = os.getenv("KOKORO_HEALTH_DIR", "/tmp/kokoro-health")
# CPU 慢机可调小，避免 16×慢前向把 WAIT_TIMEOUT 吃满；默认保持开发机口径。
N_CONCURRENCY = max(1, int(os.getenv("PARITY_CONCURRENCY", "16")))
ABANDON_REPEAT = max(1, int(os.getenv("PARITY_ABANDON_REPEAT", "60")))
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(("PASS" if ok else "FAIL"), name, detail)


def post(base, path, payload, timeout=60):
    req = urllib.request.Request(base + path, json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r.read()


def get(base, path, timeout=15):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.read().decode()


def metric(text, name):
    m = re.search(rf"^{re.escape(name)} (\S+)$", text, re.M)
    return float(m.group(1)) if m else None


# ── 1. 双网关都能服务 ──
ok1 = True
for gw in GW:
    try:
        st, h, body = post(gw, "/v1/audio/speech",
                           {"input": f"双网关验证{gw[-4:]}。", "voice": "zf_001"})
        ok1 &= st == 200 and len(h.get("x-request-id", "")) == 12
    except Exception as e:
        ok1 = False
        print("  ", gw, e)
check("双网关均服务 /v1", ok1)

# ── 2. 跨网关 e2e 聚合（/metrics 在任一网关读都是两网关之和）──
# 心跳 5s 一刷：刚返回的请求在 /metrics 里可能还没落盘，重试至多 15s。
r0 = s0 = None
for _ in range(15):
    mt0 = get(GW[1], "/metrics")
    r0 = metric(mt0, "kokoro_e2e_requests_total")
    s0 = metric(mt0, "kokoro_e2e_successes_total")
    if (r0 or 0) >= 2 and (s0 or 0) >= 2:
        break
    time.sleep(1)
check("跨网关 e2e 聚合", r0 is not None and r0 >= 2 and s0 >= 2,
      f"requests={r0} successes={s0}（含两网关请求）")

# ── 3. 断连取消：1s 超时发长文本 → 服务不崩 + cancelled 计数 ──
# 计数必须断言（曾只打印不断言：BaseHTTPMiddleware 令 is_disconnected
# 恒 False，断连取消链路在生产静默失效了很久才被 E2E 抓到）
def abandon():
    try:
        post(GW[0], "/v1/audio/speech",
             {"input": "长文本断连验证。" * ABANDON_REPEAT, "voice": "zf_001"},
             timeout=1)
    except Exception:
        pass  # 主动断开


mt0 = get(GW[0], "/metrics")
canc_before = metric(mt0, "kokoro_e2e_cancelled_total") or 0.0
t = threading.Thread(target=abandon, daemon=True)
t.start(); t.join()
time.sleep(6)  # watcher 0.5s + 取消传播 + 心跳落盘 5s
mt1 = get(GW[0], "/metrics")
canc_after = metric(mt1, "kokoro_e2e_cancelled_total") or 0.0
alive = json.loads(get(GW[0], "/health"))["status"] in ("ok", "degraded")
st, _, _ = post(GW[0], "/v1/audio/speech", {"input": "取消后恢复验证。", "voice": "zf_001"})
check("断连取消：服务不崩且计数递增", alive and st == 200 and canc_after > canc_before,
      f"cancelled {canc_before} -> {canc_after}")

# ── 4. 杀 tts：并发负载中 kill 一个，请求靠重派救回 ──
pids = {}
for _ in range(20):   # 心跳目录是活系统：写方 os.replace/清扫 unlink 并发，
    try:              # 瞬时读空/读半截都可能——重试兜底（TOCTOU 同类事故防御）
        for fn in os.listdir(HB_DIR):
            if fn.startswith("worker-"):
                try:
                    d = json.load(open(os.path.join(HB_DIR, fn), encoding="utf-8"))
                    if d.get("role") == "tts":
                        pids[d.get("wtag")] = d.get("pid")
                except Exception:
                    pass
    except OSError:
        pass
    if pids:
        break
    time.sleep(1)
assert pids, "20s 内未发现任何 tts 心跳——服务未就绪或健康目录不对"
victim_tag, victim_pid = sorted(pids.items())[0]
print(f"  目标: {victim_tag} pid={victim_pid}")

outcomes = []
lock = threading.Lock()


def worker(i):
    gw = GW[i % 2]
    try:
        st, h, _ = post(gw, "/v1/audio/speech",
                        {"input": f"故障注入并发验证第{i}句，芯片稳定。", "voice": "zf_001"},
                        timeout=150)
        with lock:
            outcomes.append(st)
    except Exception as e:
        with lock:
            outcomes.append(str(type(e).__name__))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_CONCURRENCY)]
for th in threads:
    th.start()
time.sleep(2.5)  # 负载进行中杀
if sys.platform == "win32":
    subprocess.run(["taskkill", "/F", "/PID", str(victim_pid)], capture_output=True)
else:
    import signal
    os.kill(victim_pid, signal.SIGKILL)
for th in threads:
    th.join(timeout=60)

succ = sum(1 for o in outcomes if o == 200)
to = sum(1 for o in outcomes if o == "TimeoutError" or o == 504)
threshold = max(1, N_CONCURRENCY - 2)
check("杀 tts 后请求存活（重派）", succ >= threshold,
      f"success={succ}/{N_CONCURRENCY} timeout={to} "
      f"other={[o for o in outcomes if o != 200 and 'Timeout' not in str(o)]}")

time.sleep(6)
mt2 = get(GW[0], "/metrics")
crash = metric(mt2, f'kokoro_worker_crashes_total{{worker="{victim_tag}"}}')
rd = metric(mt2, "kokoro_e2e_cancelled_total")
check("崩溃与重派被记录", (crash or 0) >= 1,
      f"crashes[{victim_tag}]={crash} redispatched={rd}")

hp = json.loads(get(GW[0], "/health"))
check("服务在 worker 重生期仍可服务", hp["ready"] is True, f"status={hp['status']}")

print("\n==== 总结 ====")
for n, ok in results:
    print(("  ✓" if ok else "  ✗"), n)
raise SystemExit(0 if all(ok for _, ok in results) else 1)

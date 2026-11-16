"""TTS 全功能 + 边界验证（多长度段落 / 边界值 / 错误路径）。

对服务发请求逐项断言：10 档长度段落（1字~2000字，验音频+时间戳单调）、
speed 钳制/拒绝、超长/空/纯标点/中英混合、非法 voice/format、坏 JSON、
413 限体、/v1 三格式、/health /metrics /v1/voices。

用法：
    python scripts/verify_boundaries.py [http://127.0.0.1:5174]
"""
import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:5174"
MAX_CHARS = 10**9
for a in sys.argv[1:]:
    if a.startswith("--max-chars="):
        MAX_CHARS = int(a.split("=", 1)[1])
    else:
        BASE = a
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, e.read(), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return -1, repr(e).encode(), time.time() - t0


def get(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, r.read()


# ══ 多长度段落实测（/tts）══
PARAGRAPHS = [
    ("p01-单字", "好", 1),
    ("p02-双字", "你好", 1),
    ("p03-五字", "今天天气好", 1),
    ("p04-短句", "这是一个短句测试。", 1),
    ("p05-两句", "第一句。第二句。", 1),
    ("p06-中句50字", "语音合成服务的质量验证需要覆盖不同长度的文本输入，"
                    "从单个字符到数百字的段落都应该稳定输出。", 1),
    ("p07-200字", "这是一段大约两百字的中等长度测试文本。" * 5, 1),
    ("p08-500字", "这段文字用于验证中长文本的合成稳定性与时间戳对齐精度。" * 12, 1),
    ("p09-跨chunk1200", "昇腾算力平台的中文语音合成服务在高并发场景下需要"
                        "稳定的批处理能力。" * 18, 1),
    ("p10-上限2000字", "边界验证文本，覆盖最大长度限制。" * 42, 1),
]

print("══ 多长度段落（/tts：音频 + 时间戳）══")
for name, text, expect_chunks in PARAGRAPHS:
    if len(text) > MAX_CHARS:
        print(f"  SKIP  {name}（{len(text)}字 > --max-chars）")
        continue
    st, body, dt = post("/tts", {"text": text, "speed": 1.0})
    if st != 200:
        check(name, False, f"HTTP {st} {body[:120]}")
        continue
    d = json.loads(body)
    ts = d.get("timestamp") or []
    audio = base64.b64decode(d["audio_base64"])
    dur = d.get("duration") or 0
    ok = (d.get("code") == 0 and len(audio) > 44
          and dur > 0 and len(ts) > 0
          and all(e >= s for s, e in ts)
          and all(ts[i][0] >= ts[i - 1][0] for i in range(1, len(ts))))
    check(f"{name}（{len(text)}字 ts={len(ts)}条 dur={dur:.1f}s {dt:.1f}s）",
          ok, f"code={d.get('code')} ts前3={ts[:3]}")
    if ok and name in ("p01-单字", "p06-中句50字"):
        # 时间戳合理性抽查：字级时间戳应在音频时长内
        inside = ts[-1][1] <= dur * 1000 + 200
        check(f"  {name} 时间戳在音频时长内", inside,
              f"ts尾={ts[-1]} dur={dur*1000:.0f}ms")

print("\n══ 边界值（/tts 与 /v1）══")
# speed 钳制/拒绝
st, body, _ = post("/tts", {"text": "语速边界。", "speed": 0.1})
check("speed=0.1 钳制到 0.5（200）", st == 200, f"{st}")
st, body, _ = post("/tts", {"text": "语速边界。", "speed": 9.0})
check("speed=9.0 拒绝（422）", st == 422, f"{st}")
st, body, _ = post("/tts", {"text": "语速上限。", "speed": 4.0})
check("speed=4.0 合法（200）", st == 200, f"{st}")

# 超长文本拒绝
st, body, _ = post("/tts", {"text": "长" * 2001, "speed": 1.0})
check("2001 字拒绝（400/422）", st in (400, 422), f"{st}")
# 空文本
st, body, _ = post("/tts", {"text": "", "speed": 1.0})
check("空 text 拒绝（400/422）", st in (400, 422), f"{st}")
# 纯标点
st, body, _ = post("/tts", {"text": "！！！……，。", "speed": 1.0})
check("纯标点 → 400（不可发音）或 200（空结果）", st in (200, 400), f"{st}")
# 英文/中英混合
st, body, _ = post("/tts", {"text": "Hello world, 你好世界!", "speed": 1.0})
check("中英混合（200）", st == 200, f"{st}")
# voice 边界
st, body, _ = post("/tts", {"text": "音色测试。", "voice": "no_such_voice"})
check("非法 voice 拒绝（400/422）", st in (400, 422), f"{st}")
# 非法 JSON
req = urllib.request.Request(BASE + "/tts", b'{"text": "broken',
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=10)
    check("坏 JSON 拒绝（4xx）", False, "200?")
except urllib.error.HTTPError as e:
    check("坏 JSON 拒绝（4xx）", 400 <= e.code < 500, f"{e.code}")
# 超大 body（413）
big = json.dumps({"text": "x" * (5 * 2**20)}).encode()
req = urllib.request.Request(BASE + "/tts", big,
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=10)
    check("5MB body 拒绝（413）", False, "200?")
except urllib.error.HTTPError as e:
    check("5MB body 拒绝（413）", e.code == 413, f"{e.code}")

print("\n══ /v1 端点与响应头 ══")
st, body, dt = post("/v1/audio/speech",
                    {"model": "kokoro-v1.1-zh", "input": "OpenAI 兼容接口测试。",
                     "voice": "zf_001", "response_format": "wav", "speed": 1.0})
check("/v1 wav（200 + RIFF 头）", st == 200 and body[:4] == b"RIFF", f"{st}")
st, body, _ = post("/v1/audio/speech",
                   {"model": "kokoro-v1.1-zh", "input": "pcm 测试。",
                    "voice": "zf_001", "response_format": "pcm", "speed": 1.0})
check("/v1 pcm（200 + 非空）", st == 200 and len(body) > 1000, f"{st} {len(body)}")
st, body, _ = post("/v1/audio/speech",
                   {"model": "kokoro-v1.1-zh", "input": "mp3 测试。",
                    "voice": "zf_001", "response_format": "mp3", "speed": 1.0})
check("/v1 mp3（200 + 非空）", st == 200 and len(body) > 500, f"{st} {len(body)}")
st, body, _ = post("/v1/audio/speech",
                   {"model": "kokoro-v1.1-zh", "input": "坏格式。",
                    "voice": "zf_001", "response_format": "ogg", "speed": 1.0})
check("response_format 非法（4xx）", 400 <= st < 500, f"{st}")

print("\n══ /health /metrics 与 /v1/voices ══")
st, body = get("/health")
h = json.loads(body)
check("health 就绪（200 + ok + ready）",
      st == 200 and h["status"] == "ok" and h["ready"] is True, str(h.get("status")))
check("health 五键齐", set(h) >= {"status", "ready", "version", "shm", "recent_failures"})
st, body = get("/metrics")
m = body.decode()
check("metrics 含 p50/p90/p99 与 e2e", all(k in m for k in
      ("kokoro_request_duration_p50_seconds", "kokoro_request_duration_p99_seconds",
       "kokoro_e2e_requests_total")))
check("metrics 含 chunk 流水的 asr jobs 指标", "kokoro_worker_done_total" in m)
st, body = get("/v1/voices")
check("voices 非空列表", st == 200 and len(json.loads(body)["voices"]) > 10)

print(f"\n══ 结果：PASS={PASS} FAIL={FAIL} ══")
sys.exit(1 if FAIL else 0)

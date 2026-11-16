"""/tts 与 /v1/audio/speech 全链路冒烟（需真实服务已起且模型权重就位）。

默认跳过（收集期不打服务）；跑真机端到端时设 RUN_LIVE_SMOKE=1 并用 PORT
指定服务端口（与 Dockerfile/README 的 5174 对齐）：
    RUN_LIVE_SMOKE=1 PORT=5174 python -m pytest tests/test_tts_api.py -v
"""
import base64
import io
import json
import os
import re
import wave

import httpx
import pytest

PORT = os.getenv("PORT", "5174")
BASE = f"http://127.0.0.1:{PORT}"

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_SMOKE", "0") != "1",
    reason="真机冒烟需完整镜像/模型；设 RUN_LIVE_SMOKE=1 开启")


def test_health_ready():
    """/health 契约：固定五键 {status, ready, version, shm, recent_failures}；
    未就绪 503 / 就绪 200（degraded 也 200）。"""
    r = httpx.get(f"{BASE}/health", timeout=60)
    d = r.json()
    print("health:", r.status_code, d["status"], "ready:", d["ready"])
    assert r.status_code in (200, 503)
    assert (r.status_code == 200) == d["ready"]   # 门闸一致性
    assert d["status"] in ("ok", "starting", "degraded")
    assert isinstance(d["ready"], bool)
    assert set(d) == {"status", "ready", "version", "shm", "recent_failures"}


def test_tts_business_contract():
    """/tts：code=0、text 字段（契约）、wav 时长与 duration 一致、
    timestamp 条目 [起ms, 止ms] 且单调非负。"""
    r = httpx.post(f"{BASE}/tts",
                   json={"text": "你好，欢迎使用昇腾算力。", "speed": 1.0},
                   timeout=300)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["code"] == 0 and d["message"] == "success"
    # 对外 6 键契约：恰好 6 键，g2p 兼容别名已移除
    assert set(d) == {"audio_base64", "code", "duration", "message",
                      "text", "timestamp"}
    assert "text" in d, "契约字段 text 必须存在"
    assert d["text"].startswith("你好")

    wav = base64.b64decode(d["audio_base64"])
    w = wave.open(io.BytesIO(wav))
    dur_wav = w.getnframes() / w.getframerate()
    assert abs(dur_wav - d["duration"]) < 0.1
    print(f"audio: {dur_wav:.2f}s")

    ts = d.get("timestamp") or []
    assert ts, "timestamp 应非空"
    assert all(isinstance(p, list) and len(p) == 2 and 0 <= p[0] <= p[1]
               for p in ts)
    starts = [p[0] for p in ts]
    assert starts == sorted(starts), "字级时间戳应按文本顺序单调"


def test_curl_no_content_type_contract():
    """不带 Content-Type 的 curl 请求复刻（--data-raw 默认
    x-www-form-urlencoded）→ 200；响应恰好 6 键；
    timestamp 段数 == 可对齐字符数。"""
    text = "你好，欢迎使用昇腾算力平台的语音合成服务。"
    r = httpx.post(
        f"{BASE}/tts",
        content=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=300)
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) == {"audio_base64", "code", "duration", "message",
                      "text", "timestamp"}
    assert d["code"] == 0 and d["message"] == "success" and d["text"] == text
    alignable = sum(1 for c in text
                    if re.match(r"[\u4e00-\u9fff0-9a-zA-Z]", c))
    assert len(d["timestamp"]) == alignable, \
        f"段数 {len(d['timestamp'])} != 可对齐字符 {alignable}"


def test_v1_audio_speech_no_alignment_needed():
    """/v1：OpenAI 兼容路径不应为纯 TTS 付出对齐成本，返回合法 wav。"""
    r = httpx.post(f"{BASE}/v1/audio/speech",
                   json={"input": "这是 OpenAI 兼容接口。", "voice": "zf_001",
                         "response_format": "wav"},
                   timeout=300)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    assert float(r.headers["X-Audio-Duration"]) > 0.2
    w = wave.open(io.BytesIO(r.content))
    assert w.getnframes() > 0


def test_error_mapping():
    """400 空 input / 未知音色 / 429 响应头语义存在性。"""
    r = httpx.post(f"{BASE}/v1/audio/speech", json={"input": "  "}, timeout=60)
    assert r.status_code == 400
    r = httpx.post(f"{BASE}/v1/audio/speech",
                   json={"input": "你好", "voice": "not_a_voice"}, timeout=60)
    assert r.status_code == 400 and "未知音色" in r.text

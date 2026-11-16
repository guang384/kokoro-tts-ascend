#!/usr/bin/env python3
"""两个 /tts 响应 JSON 的逐项对比工具（键集 / code / message / text /
duration / timestamp / audio）。

用法：
    python compare_responses.py <参考响应.json> [--url http://127.0.0.1:5174/tts] [--text 文本]

无 --text 时用内置中性样例文本向 --url 发一次请求取本地响应；也可先用
其它方式取得本地响应后自行对比（工具只做纯对比，无网络依赖的逻辑在
load_file 分支外）。

对比口径：
- 结构：键集、code/message/text、duration（秒）
- timestamp：段数与逐段窗口（同音频则应逐毫秒一致；跨合成路径仅比段数）
- audio：sha256；不同时给样本级差异统计（差异率/最大幅差/RMS 差，
  满幅 32767）——跨 CPU/NPU 浮点路径波形天然不同，统计值即"内容等价性"
  度量。
请求不带 Content-Type 发出（复刻 curl --data-raw 默认行为，可顺带验证
网关的 Content-Type 归一）。"""
import base64
import hashlib
import json
import struct
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:5174/tts"
DEFAULT_TEXT = "你好，欢迎使用昇腾算力平台的语音合成服务。"


def call_tts(url, text):
    # 不带 Content-Type：复刻 curl --data-raw 默认行为
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode("utf-8"),
                                 method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def load_file(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def audio_fingerprint(b64):
    b = base64.b64decode(b64)
    n = (len(b) - 44) // 2
    samples = struct.unpack("<%dh" % n, b[44:n * 2 + 44])
    return {
        "bytes": len(b),
        "sha256": hashlib.sha256(b).hexdigest(),
        "samples": samples,
    }


def main(argv):
    url, text, ref_path, ours_path = DEFAULT_URL, DEFAULT_TEXT, None, None
    it = iter(argv)
    for a in it:
        if a == "--url":
            url = next(it)
        elif a == "--text":
            text = next(it)
        elif a == "--ours":
            ours_path = next(it)     # 本地侧也用文件（纯离线对比）
        else:
            ref_path = a
    if not ref_path:
        print(__doc__)
        return 1

    ref = load_file(ref_path)
    ours = load_file(ours_path) if ours_path else call_tts(url, text)

    print("== 键集 ==")
    print("  参考:", sorted(ref.keys()))
    print("  本地:", sorted(ours.keys()))
    print("  一致:", sorted(ref.keys()) == sorted(ours.keys()))

    print("== 标量字段 ==")
    print("  code:", ref["code"], "vs", ours["code"])
    print("  message:", repr(ref["message"]), "vs", repr(ours["message"]))
    print("  text 一致:", ref["text"] == ours["text"])
    print("  duration: 参考", ref["duration"], " 本地", ours["duration"],
          " 差", round(abs(ref["duration"] - ours["duration"]), 4), "s")

    print("== timestamp ==")
    ta, tb = ref["timestamp"], ours["timestamp"]
    print("  段数: 参考", len(ta), " 本地", len(tb))
    print("  完全一致:", ta == tb)
    if ta != tb and len(ta) == len(tb):
        deltas = [max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p, q in zip(ta, tb)]
        print("  逐段最大偏差 ms:", max(deltas), " 平均:",
              round(sum(deltas) / len(deltas), 1))

    print("== audio ==")
    fa, fb = audio_fingerprint(ref["audio_base64"]), audio_fingerprint(ours["audio_base64"])
    print("  bytes: 参考", fa["bytes"], " 本地", fb["bytes"])
    print("  sha256: 参考", fa["sha256"][:16], " 本地", fb["sha256"][:16])
    if fa["sha256"] == fb["sha256"]:
        print("  逐字节一致 ✓✓")
    else:
        n = min(len(fa["samples"]), len(fb["samples"]))
        sa, sb = fa["samples"][:n], fb["samples"][:n]
        diff = sum(1 for x, y in zip(sa, sb) if x != y)
        mx = max(abs(x - y) for x, y in zip(sa, sb))
        rms = (sum((x - y) ** 2 for x, y in zip(sa, sb)) / n) ** 0.5
        print(f"  样本差异 {diff}/{n} ({100.0 * diff / n:.2f}%)  最大幅差 {mx}  "
              f"RMS差 {rms:.1f}（满幅 32767）")
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

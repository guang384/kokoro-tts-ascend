#!/usr/bin/env python3
"""本地 CPU 全链路复现 /tts 管线（g2p → TTS 前向 → FunASR 对齐 → 绑定
→ 6 键响应），用于离线验证端到端行为。

路径与生产同构：g2p.split_text + phonemize → phonemes_to_ids +
voice_pack[len(ps)-1] + batched_model.forward_batch（fp32 CPU）→
funasr_ts.timestamp（MonotonicAligner CPU）→ bind_char_timestamps。

用法：
    python local_e2e_compare.py [可选响应.json]
  无参数：合成内置样例文本并自检（6 键 / 段数==可对齐字符数 / 单调）；
  给参考响应.json：额外与参考逐项对比（口径同 compare_responses.py）。

运行前置（权重不入库，见 copy_models.sh 从镜像现取）：
  scripts/repro/ 下需有 kokoro_cfg.json / kokoro.pth / zf_001.pt /
  model.pt / config.yaml / tokens.json / seg_dict / am.mvn /
  configuration.json。

音频为 CPU fp32 前向，与 NPU 部署存在浮点末位差（sha256 必不同，样本级
差异率与最大幅差即"内容等价性"度量）。"""
import base64
import hashlib
import io
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("FUNASR_DEVICE", "cpu")
os.environ.setdefault("FUNASR_MODEL_DIR", HERE)

import numpy as np
import torch

DEFAULT_TEXT = "你好，欢迎使用昇腾算力平台的语音合成服务。"

# ---- 1) g2p（生产同构）----
import g2p
import pipeline

eng = pipeline.get_g2p()


def synth(text: str):
    """文本 → (响应 dict, 波形)。与生产 /tts 同构的最小管线。"""
    chunks = g2p.split_text(text, g2p._chunk_cap(1.0))
    assert len(chunks) == 1, "样例文本应单 chunk（超长请截断）"
    ps = eng.phonemize(chunks[0])[0]

    from kokoro.model import KModel
    from batched_model import phonemes_to_ids, forward_batch

    model = KModel(config=os.path.join(HERE, "kokoro_cfg.json"),
                   model=os.path.join(HERE, "kokoro.pth")).eval()
    pack = torch.load(os.path.join(HERE, "zf_001.pt"), map_location="cpu",
                      weights_only=True)
    matrix = torch.stack(list(pack)).squeeze(1)
    ids = phonemes_to_ids(model, ps)
    input_ids = torch.tensor([ids])
    lengths = torch.tensor([len(ids)])
    ref_s = matrix[len(ps) - 1].unsqueeze(0)
    with torch.inference_mode():
        outs = forward_batch(model, input_ids, lengths, ref_s,
                             torch.tensor([1.0]))
    audio = np.concatenate(outs)
    dur = round(len(audio) / 24000, 3)

    import funasr_ts
    ts_res = funasr_ts.timestamp(audio, chunks[0], 24000)
    ts_list = funasr_ts.bind_char_timestamps(chunks[0], ts_res["timestamp"])
    return dur, audio, ts_res["text"], ts_list


def _wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    n = len(samples)
    buf.write(b"RIFF"); buf.write(struct.pack("<I", 36 + n * 2)); buf.write(b"WAVE")
    buf.write(b"fmt "); buf.write(struct.pack("<IHHIIHH", 16, 1, 1,
                                              24000, 48000, 2, 16))
    buf.write(b"data"); buf.write(struct.pack("<I", n * 2))
    buf.write(samples.astype("<i2").tobytes())
    return buf.getvalue()


def selfcheck(resp: dict) -> bool:
    """契约自检：恰好 6 键 / 段数==可对齐字符数 / 起点单调。"""
    keys = set(resp)
    ok = keys == {"audio_base64", "code", "duration", "message",
                  "text", "timestamp"}
    print(f"6 键契约: {ok}")
    import re
    alignable = sum(1 for c in resp["text"]
                    if re.match(r"[\u4e00-\u9fff0-9a-zA-Z]", c))
    seg_ok = len(resp["timestamp"]) == alignable
    print(f"段数契约: {seg_ok}（{len(resp['timestamp'])} == {alignable}）")
    ts = resp["timestamp"]
    mono = all(ts[k][0] <= ts[k + 1][0] for k in range(len(ts) - 1))
    print(f"起点单调: {mono}")
    return ok and seg_ok and mono


def diff(ref: dict, ours: dict) -> None:
    """与参考响应逐项对比（口径同 compare_responses.py）。"""
    print("== 与参考对比 ==")
    print("  键集一致:", sorted(ref.keys()) == sorted(ours.keys()))
    print("  text 一致:", ref["text"] == ours["text"])
    print("  duration: 参考", ref["duration"], " 本地", ours["duration"])
    ta, tb = ref["timestamp"], ours["timestamp"]
    print("  timestamp 段数: 参考", len(ta), " 本地", len(tb),
          " 完全一致:", ta == tb)
    ba = base64.b64decode(ref["audio_base64"])
    bb = base64.b64decode(ours["audio_base64"])
    print("  audio sha256: 参考", hashlib.sha256(ba).hexdigest()[:16],
          " 本地", hashlib.sha256(bb).hexdigest()[:16],
          " 逐字节一致:", ba == bb)
    if ba != bb:
        n = (min(len(ba), len(bb)) - 44) // 2
        sa = struct.unpack("<%dh" % n, ba[44:n * 2 + 44])
        sb = struct.unpack("<%dh" % n, bb[44:n * 2 + 44])
        diffn = sum(1 for x, y in zip(sa, sb) if x != y)
        mx = max(abs(x - y) for x, y in zip(sa, sb))
        rms = (sum((x - y) ** 2 for x, y in zip(sa, sb)) / n) ** 0.5
        print(f"  样本差异 {diffn}/{n} ({100.0 * diffn / n:.2f}%)  "
              f"最大幅差 {mx}  RMS差 {rms:.1f}（满幅 32767）")


if __name__ == "__main__":
    text = DEFAULT_TEXT
    ref = None
    args = sys.argv[1:]
    if "--text" in args:
        i = args.index("--text")
        text = args[i + 1]
        del args[i:i + 2]
    if args:
        with open(args[0], encoding="utf-8") as f:
            ref = json.load(f)

    print("phonemes 前向与对齐中…（CPU 首载约 20s）")
    dur, audio, unit_text, ts_list = synth(text)
    resp = {
        "audio_base64": base64.b64encode(_wav_bytes(audio)).decode("ascii"),
        "code": 0,
        "duration": dur,
        "message": "success",
        "text": text,
        "timestamp": ts_list,
    }
    print(f"audio: {len(audio)} samples = {dur}s")
    print("funasr 单元文本:", repr(unit_text))
    ok = selfcheck(resp)
    if ref:
        diff(ref, resp)
    out = os.path.join(HERE, "local_response.json")
    json.dump(resp, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("本地响应已存", out, " 自检:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)

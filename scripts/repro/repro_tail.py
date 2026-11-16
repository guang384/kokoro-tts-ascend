#!/usr/bin/env python3
"""CPU 复现：TTS 产出与时间戳争议的取证工作台（对齐验尸 20261115）。

结论回顾：模型/管线六层全过、CPU fp32 逐窗复现与生产一致（相关 0.94），
"此类行为"确实发声；段数差异源于 FunASR 对齐吞段（r6 起由
funasr_ts.bind_char_timestamps 摊平，见 tests/test_asr_bind.py）。

复跑准备（327MB 权重不入库，需从镜像现取）：
    wsl docker run --rm -v /tmp:/out --entrypoint sh kokoro-tts-ascend:<tag> -c \\
      "cp -L /opt/hf-cache/hub/models--hexgrad--Kokoro-82M-v1.1-zh/snapshots/*/config.json /out/ \\
       && cp -L /opt/hf-cache/hub/models--hexgrad--Kokoro-82M-v1.1-zh/snapshots/*/kokoro-v1_1-zh.pth /out/ \\
       && cp -L /opt/hf-cache/hub/models--hexgrad--Kokoro-82M-v1.1-zh/snapshots/*/voices/zf_001.pt /out/"
    然后放本目录：kokoro_cfg.json / kokoro.pth / zf_001.pt（路径见下方 REPRO 常量）。

生产音频证据：语音止于 seg250=绝（~55000ms），此类行为无语音。
已排除：g2p 切分 / 音素化 / 词表映射 / SHM 写入（全链路无损）。
本脚本在生产同构路径（phonemes_to_ids + voice_pack[len(ps)-1] +
forward_batch/forward_with_tokens）上对比：
  A. 单样本前向（B=1，fp32 CPU）每 chunk 的 pred_dur 与音频长度；
  B. 三 chunk 同批前向（B=3，pad 到 L_max）chunk2 的 pred_dur；
  C. 末 chunk 尾部 30 token 的逐 token 时长对照表。
"""
import json
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, ".venv/Lib/site-packages")

from kokoro.model import KModel  # noqa: E402

REPRO = "scripts/repro"
model = KModel(config="kokoro_cfg.json", model=f"{REPRO}/kokoro.pth").eval()

RESP_PATH = sys.argv[1] if len(sys.argv) > 1 else "response.json"
d = json.load(open(RESP_PATH, encoding="utf-8"))

import g2p  # noqa: E402
import pipeline  # noqa: E402

eng = pipeline.get_g2p()
chunks = g2p.split_text(d["g2p"], 120)
ps_list = [eng.phonemize(c)[0] for c in chunks]

pack = torch.load(f"{REPRO}/zf_001.pt", map_location="cpu", weights_only=True)
matrix = torch.stack(list(pack)).squeeze(1)  # [N,256]
print(f"voice pack rows={matrix.shape[0]}  ps 长度={[len(p) for p in ps_list]}")

from batched_model import phonemes_to_ids, forward_batch  # noqa: E402

SPEED = 1.0

# ---- A. 单样本前向（生产 B=1 路径）----
print("\n== A. 单样本 forward_with_tokens ==")
singles = []
for i, ps in enumerate(ps_list):
    ids = phonemes_to_ids(model, ps)
    input_ids = torch.LongTensor([ids])
    lengths = torch.LongTensor([len(ids)])
    ref = matrix[len(ps) - 1].unsqueeze(0)
    with torch.inference_mode():
        audio, pred_dur = model.forward_with_tokens(input_ids, ref, SPEED)
    n_frames = int(pred_dur.sum())
    audio_s = audio.shape[-1] / 24000
    singles.append((input_ids, lengths, ref))
    print(f"chunk{i}: tokens={len(ids)} frames={n_frames} audio={audio_s:.2f}s")

# ---- B. 三 chunk 同批（生产凑批路径：pad 到 L_max）----
print("\n== B. forward_batch B=3 ==")
L_max = max(x.shape[1] for x, _, _ in singles)
B = len(singles)
input_ids = torch.zeros(B, L_max, dtype=torch.long)
lengths = torch.cat([y for _, y, _ in singles])
for i, (x, _, _) in enumerate(singles):
    input_ids[i, : x.shape[1]] = x[0]
ref_s = torch.cat([z for _, _, z in singles])
with torch.inference_mode():
    batched = forward_batch(model, input_ids, lengths, ref_s,
                            torch.tensor([SPEED] * B))
for i, a in enumerate(batched):
    print(f"chunk{i}: batched audio={a.shape[0] / 24000:.2f}s")

# ---- C. 末 chunk 尾部 token 逐时长对照 ----
print("\n== C. 末 chunk 尾部 30 token pred_dur（单样本 vs 文本对照）==")
ps2 = ps_list[-1]
with torch.inference_mode():
    _, pd = model.forward_with_tokens(singles[-1][0], singles[-1][2], SPEED)
pd = pd.squeeze()
# token 与 ps 字符对应：phonemes_to_ids 是逐字符映射（1 字符 = 1 token），BOS/EOS 各 1
vocab = model.vocab
tok_chars = []
for ch in ps2:
    tok_chars.append(ch if vocab.get(ch) is not None else "")
assert len(tok_chars) == pd.shape[0] - 2, (len(tok_chars), pd.shape[0])
print("最后 30 个 token（跳过 EOS）:")
tail = pd[1:-1][-30:]          # 去 BOS/EOS
tail_chars = tok_chars[-30:]
print(" ".join(f"{c!r}:{int(t)}" for c, t in zip(tail_chars, tail)))
print(f"\n尾部 4 字 '此类行为' 的音素段（ps 末尾）: …{ps2[-24:]!r}")
print(f"Σ尾部30token帧数={int(tail.sum())} = {int(tail.sum()) / 80:.2f}s @80fps")
EOF = None

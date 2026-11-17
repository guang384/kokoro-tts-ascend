#!/usr/bin/env python3
"""对齐诊断：检验 /tts 响应里 时间戳 ↔ 音频 ↔ 文本 三者是否对齐。

检查层：
  A. 结构：时间戳条数 vs 可对齐字符数；单调性（后段起点 ≥ 前段终点）；
     末段终点 vs 音频时长（覆盖不足/越界）；段间大空隙（>300ms，chunk
     边界的正常间隙 ≈60ms 插静音 + 句间停顿）。
  B. 能量：逐段 RMS——有文字的段却是静音 → 发音/文字错位；音频里有声
     但无时间戳覆盖 → 文本缺失。首部/尾部静音单独报告。
  C. 文本：字数/帧数换算（≈4.7 字/秒@speed1）偏差过大 → duration 预测
     或拆分问题。

用法：python scripts/check_align.py out.json
"""
import base64
import json
import math
import struct
import sys

STRICT = "--strict" in sys.argv[1:]
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATH = _args[0] if _args else "out.json"
problems = []
d = json.load(open(PATH, encoding="utf-8"))
text = d.get("text") or ""
ts = [(int(s), int(e)) for s, e in (d.get("timestamp") or [])]
audio = base64.b64decode(d.get("audio_base64") or b"")

# ── WAV 解析 ──
if audio[:4] != b"RIFF":
    print("!! 音频不是 RIFF/WAV")
    sys.exit(1)
ch = struct.unpack("<H", audio[22:24])[0]
bits = struct.unpack("<H", audio[34:36])[0]
byte_rate = struct.unpack("<I", audio[28:32])[0]
sr = byte_rate // (ch * bits // 8)
pcm = audio[44:]
n = len(pcm) // 2
samples = struct.unpack(f"<{n}h", pcm[: n * 2])
dur_ms = n / sr * 1000
print(f"音频: {sr}Hz {ch}ch {bits}bit  {n} 采样  {dur_ms/1000:.2f}s")
print(f"文本: {len(text)} 字符   时间戳: {len(ts)} 段")

def rms(seg):
    if not seg:
        return 0.0
    return math.sqrt(sum(x * x for x in seg) / len(seg))

# ── A. 结构检查 ──
print("\n== A. 结构 ==")
mono_bad = [(i, ts[i - 1], ts[i]) for i in range(1, len(ts))
            if ts[i][0] < ts[i - 1][1] - 1]
print(f"时间倒退(起点<前段终点): {len(mono_bad)} 处"
      + (f"  首个@{mono_bad[0][0]}: {mono_bad[0][1]}→{mono_bad[0][2]}" if mono_bad else ""))
if mono_bad:
    problems.append(f"时间倒退 {len(mono_bad)} 处")
over = [(i, ts[i][1], round(dur_ms)) for i, (s, e) in enumerate(ts) if e > dur_ms + 50]
print(f"越界(段终点>音频时长): {len(over)} 处"
      + (f"  首个@{over[0][0]}: end={over[0][1]}ms > {over[0][2]}ms" if over else ""))
if over:
    problems.append(f"时间戳越界 {len(over)} 处")
last_end = ts[-1][1] if ts else 0
print(f"末段终点 {last_end}ms vs 音频 {dur_ms:.0f}ms → 尾部未覆盖 {dur_ms - last_end:.0f}ms")
gaps = [(ts[i - 1][1], ts[i][0] - ts[i - 1][1], i) for i in range(1, len(ts))
        if ts[i][0] - ts[i - 1][1] > 300]
print(f"段间大空隙(>300ms): {len(gaps)} 处"
      + ("".join(f"\n   @idx{i} 前 {p0}ms 空 {g}ms" for p0, g, i in gaps[:6]) if gaps else ""))
if gaps:
    problems.append(f"段间空隙 >300ms: {len(gaps)} 处")

# ── B. 能量检查 ──
print("\n== B. 能量（逐段 RMS）==")
seg_rms = [rms(samples[int(s / 1000 * sr):max(int(s / 1000 * sr) + 1,
                                              int(e / 1000 * sr))]) for s, e in ts]
global_rms = rms(samples)
thr = max(300.0, global_rms * 0.08)
silent = [(i, seg_rms[i]) for i, r in enumerate(seg_rms) if r < thr]
print(f"全局 RMS={global_rms:.0f}  静音阈值={thr:.0f}")
print(f"有文字却近静音的段: {len(silent)}/{len(ts)}"
      + (f"  前 8 个 idx: {[i for i, _ in silent[:8]]}" if silent else ""))
if silent:
    problems.append(f"有文字但近静音 {len(silent)} 段")
head_sil = ts[0][0] if ts else 0
tail_sil = dur_ms - last_end
print(f"首部静音(音频起点→首段起点): {head_sil:.0f}ms   尾部静音: {tail_sil:.0f}ms")
covered = []
cur = 0.0
holes = []
for s, e in ts:
    if s - cur > 300:
        seg = samples[int(cur / 1000 * sr):int(s / 1000 * sr)]
        if rms(seg) > thr:
            holes.append((round(cur), round(s - cur)))
    cur = max(cur, e)
if dur_ms - cur > 300:
    seg = samples[int(cur / 1000 * sr):]
    if rms(seg) > thr:
        holes.append((round(cur), round(dur_ms - cur)))
print(f"有声音但无时间戳覆盖的区间: {len(holes)} 处"
      + ("".join(f"\n   起点 {p}ms 长 {g}ms" for p, g in holes[:6]) if holes else ""))
if holes:
    problems.append(f"有声无时间戳覆盖 {len(holes)} 处")

# ── C. 文本↔时长换算 ──
print("\n== C. 文本↔时长 ==")
speech_ms = sum(e - s for s, e in ts)
rate = len(text) / (speech_ms / 1000) if speech_ms else 0
print(f"Σ段时长 {speech_ms:.0f}ms  文本 {len(text)} 字 → {rate:.2f} 字/秒"
      f"（speed=1 经验值 ≈4.6）")
n_align = sum(1 for c in text if (c.isascii() and c.isalnum())
              or '\u4e00' <= c <= '\u9fff')
print(f"可对齐字符≈{n_align} vs 时间戳 {len(ts)} 段（差 {len(ts) - n_align}）")
if len(ts) != n_align:
    problems.append(f"时间戳段数 {len(ts)} != 可对齐字符 {n_align}")
per = [((e - s) / 1000, i) for i, (s, e) in enumerate(ts)]
outliers = [(i, d_ms) for d_ms, i in per if d_ms > 1.5]
print(f"超长段(>1.5s): {len(outliers)} 段"
      + ("".join(f"\n   idx{i} {d_ms:.2f}s" for i, d_ms in outliers[:6]) if outliers else ""))

if STRICT:
    print("\n== STRICT 判定 ==")
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        sys.exit(1)
    print("  PASS  结构/能量/段数契约全部通过")

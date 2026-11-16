#!/usr/bin/env python3
"""解析 bench_matrix 产出的日志（v12 字段式格式），按 req/bid/chunk 三级聚合。

行格式（[wtag] asctime LEVEL:logger:k=v …，值可含单位后缀 s）：
  g2p 行     req=… queue_ms=… process_ms=… chunks=… ovr=… chars=… sents=…
  TTS 头行   bid=… n=… process_ms=… audio_s=… dt=…
  TTS chunk  chunk=… bid=… queue_ms=… audio_s=… chars=… sents=…
  ASR 头行   bid=… n=… process_ms=…
  ASR chunk  chunk=… bid=… queue_ms=… chars=…
  gw done    ep=/tts|/v1 req=… voice=… speed=… audio_s=… e2e_s=… agg_ms=… enc_ms=…
  req text   req=… chars=… text=…

聚合口径：
  批级 pad% = 1 − Σchunk audio_s / (max(chunk audio_s) × n)——按 bid join
  头行与 chunk 行计算；请求级 e2e/queue 分布按 req join done 行与 chunk 行。

用法：
    python scripts/analyze_bench.py bench_1115-0930
    python scripts/analyze_bench.py bench_1115-0930/baseline_logs.txt …
"""
from __future__ import annotations

import glob
import os
import re
import statistics as st
import sys

_KV = re.compile(r"(\w+)=(\S+)")
_WTAG = re.compile(r"^\[(\S+)\]")
_EP = re.compile(r"ep=(/\S+)")


def parse_line(line: str) -> tuple[str, str, dict] | None:
    """→ (wtag, ts, fields)；非事件行返回 None。"""
    m = _WTAG.match(line)
    if not m:
        return None
    wtag = m.group(1)
    fields = dict(_KV.findall(line))
    ts = ""
    tm = re.search(r"\] (\d[\d:,]+ )?", line)
    if tm:
        ts = (tm.group(1) or "").strip()
    return wtag, ts, fields


def s(v: list) -> str:
    if not v:
        return "-"
    v = sorted(v)
    p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
    return f"n={len(v)} mean={st.mean(v):.2f} med={st.median(v):.2f} p90={p90:.2f} max={v[-1]:.2f}"


def pct(v: list) -> str:
    if not v:
        return "-"
    v = sorted(v)
    p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
    return f"n={len(v)} mean={st.mean(v):.1f} med={st.median(v):.1f} p90={p90:.1f} max={v[-1]:.1f}"


def analyze(path: str) -> dict:
    """→ {tts_q/asr_q: 排队分布, tts_batches: 按 bid 聚合的批次(含 pad%),
    done: gw done 行, reqs/bids: 关联键集合}

    批头行不含 req=/chunk=（批级事件无单一归属）——采集时可能被过滤；
    批级聚合（n/pad%）全部由 chunk 行按 bid 自聚合，不依赖批头行。"""
    out = {"tts_q": [], "asr_q": [], "done": [], "reqs": set(),
           "bids": set(), "tts_bids": {}, "asr_n": 0, "asr_process_ms": []}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            p = parse_line(line.rstrip("\n"))
            if p is None:
                continue
            wtag, _ts, fl = p
            if "chunk" in fl:
                bid = fl.get("bid", "")
                out["bids"].add(bid)
                out["reqs"].add(fl.get("chunk", "").split("_")[0])
                if wtag.startswith("tts"):
                    q = float(fl.get("queue_ms", "0").rstrip("s"))
                    a = float(fl.get("audio_s", "0").rstrip("s"))
                    out["tts_q"].append(q)
                    b = out["tts_bids"].setdefault(bid, {"n": 0, "a": []})
                    b["n"] += 1
                    b["a"].append(a)
                elif wtag.startswith("asr"):
                    out["asr_q"].append(float(fl.get("queue_ms", "0").rstrip("s")))
                    out["asr_n"] += 1
                continue
            if wtag.startswith("asr") and "bid" in fl and "process_ms" in fl:
                out["asr_process_ms"].append(
                    float(fl.get("process_ms", "0").rstrip("s")))
                continue
            if "ep" in fl:
                m = _EP.search(line)
                out["done"].append({
                    "ep": m.group(1) if m else "?",
                    "e2e_s": float(fl.get("e2e_s", "0").rstrip("s")),
                    "agg_ms": float(fl.get("agg_ms", "0").rstrip("s")),
                    "enc_ms": float(fl.get("enc_ms", "0").rstrip("s"))})
                out["reqs"].add(fl.get("req", ""))
    # 批级聚合：pad% = 1 − Σa/(max(a)×n)，全由 chunk 行自聚合（不依赖批头行）
    out["tts_batches"] = [
        {"bid": bid, "n": b["n"], "a_max": max(b["a"]),
         "pad": (1 - sum(b["a"]) / (max(b["a"]) * b["n"])) * 100
         if b["n"] and max(b["a"]) > 0 else 0.0}
        for bid, b in out["tts_bids"].items()]
    return out


def report(label: str, d: dict) -> None:
    print(f"\n======== {label} ========")
    print(f"关联：reqs={len(d['reqs'])} bids={len(d['bids'])}")
    tb = d["tts_batches"]
    if tb:
        print(f"[TTS 批（按 bid 自聚合）] 批数={len(tb)}")
        print(f"  n(每批 chunk) : {pct([b['n'] for b in tb])}")
        print(f"  a_max(最长句) : {pct([b['a_max'] for b in tb])}")
        pads = [b["pad"] for b in tb]
        print(f"  pad%          : {pct(pads)}")
    if d["asr_n"]:
        print(f"[ASR] chunks={d['asr_n']}  process_ms(批) : "
              f"{s(d['asr_process_ms'])}")
    if d["tts_q"]:
        print(f"[TTS chunk] queue_ms : {s(d['tts_q'])}")
    if d["asr_q"]:
        print(f"[ASR chunk] queue_ms : {s(d['asr_q'])}")
    done = d["done"]
    for ep in sorted({x["ep"] for x in done}):
        rows = [x for x in done if x["ep"] == ep]
        print(f"[{ep}] e2e_s   : {s([x['e2e_s'] for x in rows])}")
        print(f"           agg_ms  : {s([x['agg_ms'] for x in rows])}")
        print(f"           enc_ms  : {s([x['enc_ms'] for x in rows])}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    paths: list[tuple[str, str]] = []
    for a in args:
        if os.path.isdir(a):
            for p in sorted(glob.glob(os.path.join(a, "*_logs.txt"))):
                paths.append((os.path.basename(p).replace("_logs.txt", ""), p))
        else:
            paths.append((os.path.basename(a).replace("_logs.txt", ""), a))
    if not paths:
        print("未找到 *_logs.txt")
        sys.exit(1)
    for label, p in paths:
        report(label, analyze(p))


if __name__ == "__main__":
    main()

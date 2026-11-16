#!/usr/bin/env python3
"""fp16 音质 A/B 实测：同一服务实例，同文本分别以 dtype=fp32/fp16 请求
参数各合成一遍，产出单个自包含 HTML（音频 base64 内嵌），浏览器打开做
盲听验收。

精度是请求级参数（协议 chunk 携带，认领类键隔离不混批）——单实例即可
同时获得两种音质，无需第二个容器/实例，生产服务全程不动。

每样本记录：audio_base64、时长、合成耗时、peak/RMS（客观指标）、
时间戳条数（fp16 若时长预测受损，时间戳漂移会直接可见）。

用法（真机容器内或宿主机，服务已在运行，镜像含请求级 dtype）：
    python3 scripts/ab_quality_test.py --base http://127.0.0.1:5174

退出码：0 成功；2 服务不可达/未就绪或 fp16 请求失败（CANN autocast
不支持时据此判断放弃）。注意：若 fp16 把 NPU worker 挂死（CANN 级 hang），
网关 504 后仍需等看门狗（STUCK_KILL_S）杀进程重生才恢复——建议在测试
实例或低峰期执行。
"""
import argparse
import array
import base64
import io
import json
import os
import secrets
import sys
import time
import wave

import httpx

# A/B 文本集：覆盖多音字/数字/英文混排/长短句/情感句
TEXTS = [
    "这是一段长文本测试，长度一直在增长，很长一段时间内都很长。",
    "2026年第三季度营收增长15.6%，共计12345件。",
    "Hello world，这是中英混合测试，Kokoro 引擎运行正常。",
    "重庆的银行在长街上重新长出了新芽，行长的行长证重发了。",
    "短句。",
    "语音合成的质量不仅取决于发音的准确，还取决于韵律的自然、停顿的恰当，"
    "以及重音位置的微妙处理；这些细节共同决定了听感的优劣。",
]


def synth(client: httpx.Client, url: str, text: str, voice: str,
          dtype: str = "fp32") -> dict:
    """请求 /tts 并提取客观指标（peak/RMS/时长/时间戳条数）。"""
    r = client.post(f"{url}/tts", json={"text": text, "voice": voice,
                                        "speed": 1.0, "dtype": dtype},
                    timeout=600)
    r.raise_for_status()
    d = r.json()
    wav = base64.b64decode(d["audio_base64"])
    w = wave.open(io.BytesIO(wav))
    sr = w.getframerate()
    samples = array.array("h")
    samples.frombytes(w.readframes(w.getnframes()))
    peak = max(abs(s) for s in samples) / 32767.0
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32767.0
    return {"wav_b64": base64.b64encode(wav).decode("ascii"),
            "duration": d["duration"],
            "ts_count": len(d.get("timestamp") or []), "sr": sr,
            "peak": round(peak, 4), "rms": round(rms, 5)}


def wait_ready(url: str, deadline_s: float):
    """等 /health ready；返回 health 字典，超时返回 None。

    串行启动（6 tts + 3 asr 各一次模型加载/冒烟）实测约 10 分钟，
    默认宽限 900s——刚拉起的容器直接跑本脚本不再误报"不可达"。"""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                h = r.json()
                if h.get("ready"):
                    return h
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return None


PAGE_HEAD = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>fp16 音质 A/B 验收</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;padding:24px;background:#14171c;color:#d8dee6;
      font:14px/1.6 "Segoe UI","Microsoft YaHei",sans-serif}
 h1{font-size:18px} h1 small{color:#7a8592;font-weight:normal;font-size:12px}
 .card{background:#1c2027;border:1px solid #2a2f38;border-radius:10px;
       padding:14px 16px;margin-top:14px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media(max-width:900px){.grid{grid-template-columns:1fr}}
 audio{width:100%;margin-top:6px}
 .tag{display:inline-block;padding:2px 10px;border-radius:6px;font-size:12px;
      font-weight:600;margin-right:6px}
 .A{background:#1d4ed8}.B{background:#b45309}
 .meta{color:#8b95a3;font-size:12px;margin-top:6px}
 table{border-collapse:collapse;width:100%;font-size:12px}
 td,th{border:1px solid #2a2f38;padding:6px 10px;text-align:left}
 th{color:#8b95a3;font-weight:600}
 .judge button{margin-right:8px;padding:8px 16px;border-radius:7px;border:0;
       cursor:pointer;font-size:13px}
 .b1{background:#1d4ed8;color:#fff}.b2{background:#b45309;color:#fff}
 .b3{background:#374151;color:#fff}
 textarea{width:100%;min-height:60px;background:#11141a;color:#d8dee6;
          border:1px solid #333a45;border-radius:6px;padding:8px;font:inherit}
 #verdict{margin-top:10px;color:#7dd3fc;font-size:14px;white-space:pre-wrap}
</style></head><body>
<h1>fp16 音质 A/B 验收 <small id="buildinfo"></small></h1>
<div class="card">盲听规则：整页固定一次随机 A/B 映射（每份报告不同种子，刷新不变）。
听完每组选"A/B 更自然"或"无差异"——评分期间不显示任何精度线索（音频旁不
放指标，客观指标表默认隐藏）。全部评完再点"显示客观指标"或总评（两者都会
揭示 A/B 对应的精度），页尾导出 JSON 记录结论与映射。</div>
<div id="pairs"></div>
<div class="card"><h3 style="font-size:15px">客观指标对照</h3>
<button onclick="revealMetrics(this)" style="padding:6px 14px;border:0;border-radius:7px;
        background:#374151;color:#fff;cursor:pointer">
显示客观指标（会揭示 A/B 精度）</button>
<div id="metricsBox" style="display:none;margin-top:10px"><table id="metrics"></table></div></div>
<div class="card judge"><h3 style="font-size:15px">总评</h3>
<p>整体判断（对照全部样本后）：</p>
<button class="b1" onclick="verdict('A')">A 组整体更自然</button>
<button class="b3" onclick="verdict('tie')">无可闻差异，可上线</button>
<button class="b2" onclick="verdict('B')">B 组整体更自然</button>
<textarea id="note" placeholder="补充听感备注（哪个样本哪一秒的问题）…"></textarea>
<div id="verdict"></div>
<button onclick="exportJson()" style="margin-top:10px;padding:8px 18px;
        border:0;border-radius:7px;background:#2563eb;color:#fff;cursor:pointer">
导出验收 JSON</button>
<pre id="out" style="margin-top:10px;white-space:pre-wrap;color:#8b95a3"></pre></div>
<script>
const DATA = __DATA__;
document.getElementById("buildinfo").textContent =
  `镜像=${DATA.build}  生成于 ${DATA.generated}`;
// 整页一次随机映射（seed 由生成脚本随机注入：刷新不变、跨报告真随机）
let seed = DATA.seed;
function rnd() { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; }
const abMap = rnd() < 0.5 ? {A: "fp16", B: "fp32"} : {A: "fp32", B: "fp16"};
const box = document.getElementById("pairs");
DATA.pairs.forEach((p, i) => {
  const side = (k) => p[abMap[k]];
  const d = document.createElement("div");
  d.className = "card";
  d.innerHTML = `
   <b>样本 ${i + 1}</b><div class="meta" style="margin:2px 0 8px">${p.text}</div>
   <div class="grid">
    <div><span class="tag A">A</span><audio controls preload="none"
      src="data:audio/wav;base64,${side("A").wav_b64}"></audio></div>
    <div><span class="tag B">B</span><audio controls preload="none"
      src="data:audio/wav;base64,${side("B").wav_b64}"></audio></div>
   </div>
   <div class="judge" style="margin-top:8px">
    <button class="b1" onclick="pick(${i},'A')">A 更自然</button>
    <button class="b2" onclick="pick(${i},'B')">B 更自然</button>
    <button class="b3" onclick="pick(${i},'tie')">无差异</button>
    <span id="res${i}" class="meta">未评</span></div>`;
  box.appendChild(d);
});
const picks = {};
function pick(i, v) {
  picks[i] = v;
  // 不揭示精度：整页映射一致，揭示一个样本就等于揭示全部
  document.getElementById("res" + i).textContent =
    v === "tie" ? "已评：无差异" : `已评：${v} 侧`;
}
// 明细表按精度列出（客观指标无需盲）
const mt = document.getElementById("metrics");
mt.innerHTML = "<tr><th>#</th><th>文本</th><th>精度</th><th>时长s</th><th>ts条数</th>"
  + "<th>peak</th><th>RMS</th><th>合成耗时s</th></tr>";
DATA.pairs.forEach((p, i) => {
  ["fp32", "fp16"].forEach((k) => {
    mt.innerHTML += `<tr><td>${i + 1}</td><td>${p.text.slice(0, 18)}…</td>
     <td>${k}</td><td>${p[k].duration}</td><td>${p[k].ts_count}</td>
     <td>${p[k].peak}</td><td>${p[k].rms}</td><td>${p[k].synth_s}</td></tr>`;
  });
});
// 客观指标默认隐藏：表格按精度标注，评分前看一眼就能认出哪边是 fp16
function revealMetrics(btn) {
  document.getElementById("metricsBox").style.display = "block";
  btn.textContent = `已显示（A=${abMap.A} / B=${abMap.B}）`;
  btn.disabled = true;
}
function verdict(v) {
  const text = v === "tie"
    ? "结论：无可闻差异——fp16 可上线"
    : abMap[v] === "fp16"
      ? `结论：${v} 组（fp16）整体更自然——fp16 可接受，建议灰度观察`
      : `结论：${v} 组（fp32）整体更自然——fp16 存在可闻劣化，保持 fp32`;
  document.getElementById("verdict").textContent =
    text + `　[A=${abMap.A} / B=${abMap.B}]`;
}
function exportJson() {
  const out = {verdict: document.getElementById("verdict").textContent,
               note: document.getElementById("note").value,
               picks: Object.fromEntries(Object.entries(picks).map(
                   ([i, v]) => [i, v === "tie" ? "tie" : abMap[v]])),
               ab_map: abMap, seed: DATA.seed, build: DATA.build,
               generated: DATA.generated};
  document.getElementById("out").textContent = JSON.stringify(out, null, 2);
  navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(out));
}
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5174",
                    help="服务地址（单实例，精度由请求参数控制）")
    ap.add_argument("--voice", default="zf_001")
    ap.add_argument("--out", default="/tmp/ab_dtype_test.html")
    ap.add_argument("--wait-s", type=float, default=900.0,
                    help="等待 /health ready 的上限秒数（串行启动实测约 10 分钟）")
    args = ap.parse_args()

    health = wait_ready(args.base, args.wait_s)
    if not health:
        print(f"ERR: {args.base} 在 {args.wait_s:.0f}s 内未就绪"
              "（/health 需 200 且 ready=true）")
        return 2
    build = health.get("version") or "unknown"
    print(f"目标镜像版本：{build}")
    print("要求：镜像含请求级 dtype；fp16 组若 CANN 不支持会在首批就 500，"
          "据此判断放弃")
    pairs = []
    with httpx.Client(timeout=600.0) as c:
        for i, text in enumerate(TEXTS):
            # 交错顺序：fp32 总是先跑会预热 G2P 缓存/流水，让后跑的 fp16
            # 耗时天然偏快（synth_s 是客观对照指标，不能带这种顺序偏差）
            order = ("fp16", "fp32") if i % 2 else ("fp32", "fp16")
            got = {}
            for dt in order:
                t1 = time.time()
                try:
                    got[dt] = synth(c, args.base, text, args.voice, dtype=dt)
                except httpx.HTTPError as exc:
                    if dt == "fp16":
                        print("ERR: fp16 请求失败——CANN autocast 不支持或算子"
                              f"缺失，fp16 路线放弃（{str(exc)[:120]}）")
                    else:
                        print(f"[{i+1}] fp32 请求失败（服务健康问题）: {exc}")
                    return 2
                got[dt]["synth_s"] = round(time.time() - t1, 2)
            a, b = got["fp32"], got["fp16"]
            pairs.append({"text": text, "fp32": a, "fp16": b})
            print(f"[{i + 1}/{len(TEXTS)}] 顺序={order[0]}→{order[1]}"
                  f"  dur {a['duration']:.2f}/{b['duration']:.2f}s"
                  f"  ts {a['ts_count']}/{b['ts_count']}"
                  f"  synth {a['synth_s']}/{b['synth_s']}s"
                  f"  peak {a['peak']}/{b['peak']}")

    page = PAGE_HEAD.replace("__DATA__", json.dumps({
        "pairs": pairs,
        "build": build,
        "seed": secrets.randbits(31),   # 每份报告随机映射；刷新不变
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    size_mb = os.path.getsize(args.out) / 2**20
    print(f"\nHTML 已生成：{args.out}（{size_mb:.1f}MB）"
          " —— 拷到本机浏览器打开做盲听验收")
    return 0


if __name__ == "__main__":
    sys.exit(main())

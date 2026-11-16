#!/usr/bin/env python3
"""concat 质量门 1 的 A/B 听测工具（纯客户端：httpx + 标准库，无 torch 依赖）。

对照实验：A 侧 = TTS_CHUNK_CHARS=120（现状，逐 120 字 chunk 独立合成）；
B 侧 = TTS_CHUNK_CHARS=450（≈510 token 上限内，多句合并为单序列单前向，
即 concat 核心机制：跨句注意力、单 BOS/EOS、连续韵律）。

用法：
    # A/B 两个容器分别跑起来后（A:5174 默认切句，B:5175 chunk=450）：
    python scripts/ab_quality.py --url-a http://127.0.0.1:5174 \
        --url-b http://127.0.0.1:5175 --out ab_out
    # 听测完成后揭晓盲测答案：
    python scripts/ab_quality.py --out ab_out --reveal

产出：
    <out>/A/*.wav、<out>/B/*.wav          原始对照
    <out>/blind/X*_样本N_*.wav            盲测对（随机交换 A/B 位置）
    <out>/manifest.json                   真实映射（--reveal 可读）
    <out>/听测指南.txt                     听什么、怎么打分
    客观指标：时长差 / 句间停顿分布 / 短时能量突跳计数
"""
import argparse
import json
import math
import random
import shutil
import struct
import sys
import time
import wave
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("需要 httpx：pip install httpx（容器内已自带）")

SR = 24000

# 纯中文测试文本（无数字/英文：数字与英文的音素展开可能顶破 510 token 截断）
TEXTS = {
    "T1_短段": (
        "深夜的巷子里只剩下几盏昏黄的灯。她推开门，熟悉的声音从院子里传来，"
        "像是很久以前的回响。风把梧桐叶吹得沙沙作响，仿佛在替谁诉说着心事。"
        "她在原地听了许久，直到声音渐渐淡去，才慢慢转身离开。"
    ),
    "T2_中段核心": (
        "清晨的集市比想象中热闹得多。卖菜的摊主把新鲜的青菜码得整整齐齐，招呼声此起彼伏；"
        "不远处，炸油条的香气顺着人流飘散开来。孩子们背着书包穿过巷口，笑闹声惊起了檐下的鸽子；"
        "老人提着鸟笼慢悠悠地走着，笼里的画眉叫得婉转清亮。阳光越过屋脊，把青石板路照得发亮，"
        "来来往往的脚步在这片光里来去匆匆。有人停下脚步，只为看一眼糖画师傅手腕的翻转；"
        "有人匆匆而过，心里惦记着还没办完的事。这座城每天都是从这样的清晨开始的，"
        "喧嚣、琐碎，却生机勃勃。等日头再高一些，这些声音就会汇成一片，把整条街都填得满满当当。"
    ),
    "T3_长段": (
        "山里的四季比平原来得分明。春天漫山的杜鹃开得泼辣，风一吹，整面山坡都像在燃烧；"
        "夏天雨水多，溪水涨起来，夜里躺在床上能听见水声一路奔下山去。"
        "秋天是村里最忙的时候，家家户户的屋顶上都晒着红辣椒和金黄的玉米，"
        "空气里全是收获的味道。冬天雪一下，山路就封了，村子安静得像睡着了，"
        "只有偶尔的犬吠提醒你，这里还有人烟。"
        "祖辈在这片山里生活了几代人，守着几亩薄田和一片林地，日子清苦，却也踏实。"
        "年轻时总想走出去，看看山外面的世界；真正在外面漂泊多年之后才明白，"
        "有些东西是长在骨头里的，走多远都带不走。比如清晨的雾，比如灶台上的烟火气，"
        "比如村口那棵老槐树下，一代代人坐着说话的声音。"
        "如今山路修通了，年轻人还是陆续离开，村子一天天安静下来。"
        "留下来的老人不识字，却记得每一场雨、每一场雪，记得谁家的孩子考上了学堂，"
        "谁家的老人在哪个冬天离开。他们守着这片山，像守着一本没有字的书。"
    ),
    "T4_超长段": (
        "老师傅修了一辈子的钟表。铺子不大，柜台玻璃擦得透亮，里面静静躺着各式各样的表，"
        "有的早已停摆，有的还在滴答作响。他说，每一只表里都藏着一段日子："
        "结婚时的对表，退休时的纪念表，还有孩子长途跋涉带回来越洋买的手表。"
        "修表是个慢工活。镊子夹起比米粒还小的齿轮，对准，落位，上油，"
        "呼吸都要放轻；稍有差池，一个零件的损伤就可能让整只表再也走不起来。"
        "年轻时他手快，一天能修十几只；如今眼睛花了，一天修三四只，"
        "却觉得每一只都比从前修得更用心。铺子里的灯光总是暖黄的，"
        "照着他的白发，也照着那些沉默的零件。"
        "常有老主顾拿着旧表进来，坐下就不愿走，讲起表的来历，一讲就是半天。"
        "他总是安静地听着，手上的活不停。他说修表修的其实不是表，"
        "是人家托付过来的一份念想；表走了，日子就还在走。"
        "铺子对面开了新式的商场，橱窗里的电子表又亮又便宜，"
        "来修老表的客人一年比一年少。有人劝他关了铺子歇着，他总是笑笑，"
        "说不着急，还能修一天是一天。傍晚收铺子的时候，他习惯把每一只待取的表"
        "贴在耳边听一遍，确认它们都在好好地走着，才关灯锁门。"
        "巷子深处，滴答声和远处的车流声混在一起，像两个时代的低声交谈。"
        "他知道，总有一天铺子会关，那些齿轮和游丝会安静下来；"
        "但他也相信，只要还有人愿意为一秒钟的误差跑一趟，这门手艺就还能再走很久。"
        "灯熄了，滴答声留在黑夜里，像时间自己均匀的呼吸。"
    ),
}


def wav_bytes_ok(content: bytes) -> bool:
    return len(content) > 44 and content[:4] == b"RIFF"


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == SR, f"{path} 格式异常"
        n = w.getnframes()
        raw = w.readframes(n)
    samples = struct.unpack(f"<{n}h", raw)
    return samples


def window_rms(samples, win=480):  # 480 样本 = 20ms
    out = []
    for i in range(0, len(samples) - win + 1, win):
        acc = 0
        for s in samples[i:i + win]:
            acc += s * s
        out.append(math.sqrt(acc / win))
    return out


def analyze(path: Path) -> dict:
    """客观指标：时长 / 停顿（静音段）统计 / 短时能量突跳计数。"""
    samples = read_wav(path)
    dur = len(samples) / SR
    rms = window_rms(samples)
    peak = max(rms) if rms else 0
    thr = max(peak * 0.02, 50.0)  # 静音阈值：峰值的 2%（绝对下限兜底）

    pauses, run = [], 0
    jumps = 0
    prev = None
    for v in rms:
        if v < thr:
            run += 1
        else:
            if run * 0.02 >= 0.12:  # ≥120ms 的连续静音记为一次停顿
                pauses.append(run * 0.02)
            run = 0
        if prev is not None and v > thr * 2 and prev > thr * 2:
            ratio = max(v, prev) / max(min(v, prev), 1e-9)
            if ratio > 8:  # 非静音窗口间能量突降 >8 倍：疑似切点
                jumps += 1
        prev = v
    return {"duration_s": round(dur, 2), "pauses": len(pauses),
            "pause_total_s": round(sum(pauses), 2),
            "energy_jumps": jumps}


def synth(url: str, text: str, out_path: Path, timeout: float = 600.0) -> float:
    t0 = time.time()
    r = httpx.post(url + "/v1/audio/speech",
                   json={"model": "kokoro-v1.1-zh", "input": text,
                         "voice": "zf_001", "response_format": "wav"},
                   timeout=timeout)
    dt = time.time() - t0
    if r.status_code != 200 or not wav_bytes_ok(r.content):
        sys.exit(f"合成失败 {url}: HTTP {r.status_code} {r.text[:200]}")
    out_path.write_bytes(r.content)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser(description="concat 质量门 1：A/B 听测工具")
    ap.add_argument("--url-a", default="http://127.0.0.1:5174", help="A 侧（chunk=120 现状）")
    ap.add_argument("--url-b", default="http://127.0.0.1:5175", help="B 侧（chunk=450 单序列多句）")
    ap.add_argument("--out", default="ab_out", help="产出目录")
    ap.add_argument("--reveal", action="store_true", help="显示盲测映射答案（听完后用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不合成")
    args = ap.parse_args()

    out = Path(args.out)
    print("── 测试文本 ──")
    for name, text in TEXTS.items():
        n_chunks_a = -(-len(text) // 120)
        n_chunks_b = -(-len(text) // 450)
        print(f"  {name}: {len(text)} 字 → A 侧约 {n_chunks_a} 个 chunk / B 侧约 {n_chunks_b} 个")

    if args.dry_run:
        print(f"\n计划：A={args.url_a}  B={args.url_b} → {out}/A、{out}/B、盲测对 {out}/blind/")
        return 0

    if args.reveal:
        mf_path = out / "manifest.json"
        if not mf_path.exists():
            sys.exit(f"找不到 {mf_path}，先跑合成")
        mf = json.loads(mf_path.read_text(encoding="utf-8"))
        print("── 盲测答案（B 侧 = chunk 450 单序列多句）──")
        for sample, info in sorted(mf["blind"].items()):
            print(f"  {sample}: X1={info['x1']}  X2={info['x2']}")
        return 0

    for side, url in (("A", args.url_a), ("B", args.url_b)):
        d = out / side
        d.mkdir(parents=True, exist_ok=True)
        print(f"── {side} 侧合成：{url} ──")
        for name, text in TEXTS.items():
            dt = synth(url, text, d / f"{name}.wav")
            m = analyze(d / f"{name}.wav")
            print(f"  {name}: {dt:.1f}s 合成 → 时长 {m['duration_s']}s "
                  f"停顿 {m['pauses']}处/{m['pause_total_s']}s 突跳 {m['energy_jumps']}")
        (d / "metrics.json").write_text(json.dumps(
            {n: analyze(d / f"{n}.wav") for n in TEXTS}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # 盲测打包：每条文本随机交换 A/B 位置
    blind = out / "blind"
    blind.mkdir(exist_ok=True)
    rng = random.Random(20261106)
    manifest = {"blind": {}}
    for idx, name in enumerate(TEXTS, 1):
        x1_is_a = rng.random() < 0.5
        x1, x2 = ("A", "B") if x1_is_a else ("B", "A")
        shutil.copy(out / x1 / f"{name}.wav", blind / f"X1_样本{idx}_{name}.wav")
        shutil.copy(out / x2 / f"{name}.wav", blind / f"X2_样本{idx}_{name}.wav")
        manifest["blind"][f"样本{idx}"] = {"x1": x1, "x2": x2, "text": name}

    guide = (
        "concat 质量门 1 · 盲听指南\n"
        "========================\n"
        "用耳机。每对样本是同一段文本的两种合成方式（X1/X2 顺序已随机），\n"
        "其中一种为现状（120 字分片独立合成），另一种为多句单序列连续合成。\n\n"
        "重点听「句与句之间」：\n"
        "  1. 句间停顿的长度与自然度（是否像自然换气）\n"
        "  2. 语调衔接（上一句的收尾和下一句的起始是否连贯）\n"
        "  3. 整段听起来是流畅的叙述，还是一句一句\"报清单\"\n"
        "  4. 有无突兀的切换感、电子味、时长异常\n\n"
        "建议顺序：先听 样本2（中段核心样本，差异最集中），再 3、4，最后 1。\n"
        "每对三选一：X1 更好 / X2 更好 / 接近分不出，备注写具体位置与感受。\n\n"
        "客观参考（同目录 metrics.json 与 A/、B/ 侧 metrics.json）：\n"
        "  时长差应 <2%；能量突跳数异常偏高的一侧可能有切点瑕疵。\n\n"
        "判定：\n"
        "  多句单序列一侧句间衔接不劣于分片侧 → 门 1 通过\n"
        "  多句单序列一侧更连贯 → 加分\n"
        "  明显劣化（拖长音/语义拉偏/电子感）→ 门 1 未通过\n"
    )
    (out / "听测指南.txt").write_text(guide, encoding="utf-8")
    manifest["texts"] = {n: len(t) for n, t in TEXTS.items()}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"── 完成：盲测对在 {blind}/，指南在 {out/'听测指南.txt'}（答案用 --reveal 查看）──")
    return 0


if __name__ == "__main__":
    sys.exit(main())

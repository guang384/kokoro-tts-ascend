#!/usr/bin/env python3
"""Kokoro TTS 服务并发压测：找单卡吞吐拐点，决定 GATEWAY_WORKERS/容器副本数。

用法（宿主机装了 httpx，或容器内已自带）:
    python scripts/benchmark.py --url http://127.0.0.1:8000 -c 1,2,4,8 -n 16
    python scripts/benchmark.py --url http://127.0.0.1:8000 -c 4 --random 50 --voice zm_001
    python scripts/benchmark.py --url http://127.0.0.1:8000 -c 4 --text "更长的一段中文……" --voice zm_001

指标:
    req/s          每秒完成的请求数
    realtime       聚合倍速 = 音频总时长 / 总耗时（>1 表示比实时快）
    chars/s        每秒合成字符数
    mean/p95       请求延迟

注意：服务端 g2p 句子级 LRU 缓存（G2P_CACHE_SIZE）会让固定 --text 的压测
全部命中缓存、吞吐虚高——测真实路径请用 --random N（N 条互不相同文本轮转）。
"""
import argparse
import asyncio
import math
import random
import statistics
import sys
import time

import httpx

# 随机文本语料：中文词表拼接（无语义，仅作打穿缓存的载荷）
_WORDS = ["华为", "昇腾", "处理器", "人工智能", "训练", "推理", "性能",
          "语音", "合成", "并发", "吞吐", "延迟", "服务", "稳定", "监控",
          "日志", "批次", "队列", "音频", "文本", "模型", "显卡", "集群",
          "容器", "编排", "网关", "弹性", "扩容", "压测", "拐点"]
_ENDINGS = "。！？；"


def random_texts(n: int, seed: int = 20261108, max_chars: int = 800) -> list:
    """生成 n 条互不相同、长短分层混合的随机文本（固定 seed 可复现）。

    长度分层模拟真实流量形态，确保压到不同代码路径：
      短(~2-8字)    问候/重试类——命中 g2p 缓存的主力
      中(~8-24字)   常规单句
      长(~120-320字) 超过 TTS_CHUNK_CHARS → 多句 chunk + 多批聚合路径
      超长(400~max_chars 字) 多句长单 → 对齐多组切分（>30s 音频）+
                    长占用闸门额度的最坏情况
    长文本按句（词组+终止标点）构造，与 g2p 切句规则一致。"""
    rng = random.Random(seed)

    def _sentence(k_lo, k_hi):
        return ("".join(rng.choice(_WORDS) for _ in range(rng.randint(k_lo, k_hi)))
                + rng.choice(_ENDINGS))

    def _multi_sentence(target_chars, k_lo, k_hi):
        sents, used = [], 0
        while used < target_chars:
            s = _sentence(k_lo, k_hi)
            sents.append(s)
            used += len(s)
        return "".join(sents)

    def _short():
        return "".join(rng.choice(_WORDS) for _ in range(rng.randint(1, 2)))

    def _medium():
        return "".join(rng.choice(_WORDS) for _ in range(rng.randint(4, 12)))

    def _long():
        return _multi_sentence(rng.randint(120, 320), 2, 6)

    def _xlong():
        return _multi_sentence(rng.randint(400, max_chars), 3, 8)

    # 分层配额：20% 短 / 40% 中 / 28% 长 / 12% 超长（n 小时各层保底 1 条）
    quotas = {"s": max(1, n * 20 // 100), "m": max(1, n * 40 // 100),
              "l": max(1, n * 28 // 100), "x": max(1, n * 12 // 100)}
    gens = ([_short] * quotas["s"] + [_medium] * quotas["m"]
            + [_long] * quotas["l"] + [_xlong] * quotas["x"])
    gens = (gens + [_medium] * n)[:n]   # 配额取整损耗用中层补齐/截断
    rng.shuffle(gens)   # 打散长度顺序：避免同长度扎堆影响攒批窗口形态

    out, seen = [], set()
    for gen in gens:
        for _attempt in range(64):
            t = gen()
            if t not in seen:
                seen.add(t)
                out.append(t)
                break
        else:            # 该层撞重（几乎不可能）：退中层
            out.append(_medium())
    return out


# 已知竞态（README 排障记录）：服务端 KEEP_ALIVE_TIMEOUT_S=0 逐请求 accept
# 与客户端连接池的关闭竞态 → 连接复用瞬间撞上服务端已关闭的连接，抛
# RemoteProtocolError("Server disconnected...")——请求未被受理，服务端零痕迹。
# httpx transport 的 retries=2 只覆盖建连失败，覆盖不到这种复用中段失败。
# 默认不复用连接后此路径理论为 0；应用级重试保留作保险（重试成功单独
# 计 retried——若不复用模式下 retried 仍 >0，说明还有别的连接层问题）。
_RACE_SIGNS = ("disconnected", "connection reset", "remoteprotocolerror")


def _is_race(exc: Exception) -> bool:
    return any(k in str(exc).lower() for k in _RACE_SIGNS)


async def _one_attempt(client: httpx.AsyncClient, url: str, text: str, voice: str,
                       fmt: str, endpoint: str, speed: float, t0: float):
    """单次请求尝试：返回 (ok, latency, dur, err)。异常向上抛（调用方区分竞态）。"""
    if endpoint == "tts":
        payload = {"model": "kokoro-v1.1-zh", "text": text, "voice": voice,
                   "speed": speed}
        r = await client.post(f"{url}/tts", json=payload)
    else:
        r = await client.post(
            f"{url}/v1/audio/speech",
            json={"model": "kokoro-v1.1-zh", "input": text, "voice": voice,
                  "response_format": fmt, "speed": speed},
        )
    latency = time.perf_counter() - t0
    if r.status_code != 200 or len(r.content) < 100:
        return False, latency, 0.0, f"HTTP {r.status_code}: {r.text[:120]}"
    if endpoint == "tts":
        try:
            dur = float(r.json().get("duration", 0))
        except Exception:
            dur = 0.0
    else:
        dur = float(r.headers.get("x-audio-duration", 0))
    return True, latency, dur, ""


async def one_request(client: httpx.AsyncClient, url: str, text: str, voice: str, fmt: str,
                      endpoint: str, speed: float = 1.0):
    """带竞态重试：第 1 次走连接池；命中已知 keep-alive 竞态（请求未受理、
    服务端零痕迹）→ 换新连接重试一次，重试成功以 retried 单独计数，
    不污染失败统计。延迟如实包含白跑的那一小段。"""
    t0 = time.perf_counter()
    race_exc = None
    try:
        ok, lat, dur, err = await _one_attempt(client, url, text, voice, fmt,
                                               endpoint, speed, t0)
        return ok, lat, dur, err, False
    except Exception as e:
        if not _is_race(e):
            return False, time.perf_counter() - t0, 0.0, str(e), False
        race_exc = e
    try:  # 竞态命中：换新连接重试一次
        async with httpx.AsyncClient(
                transport=httpx.AsyncHTTPTransport(retries=2),
                timeout=httpx.Timeout(600.0, connect=10.0)) as c2:
            ok, lat, dur, err = await _one_attempt(c2, url, text, voice, fmt,
                                                   endpoint, speed, t0)
            return ok, lat, dur, err, True
    except Exception as exc2:
        return False, time.perf_counter() - t0, 0.0, \
            f"{str(race_exc)[:80]} -> retry: {str(exc2)[:80]}", True


async def run_level(url: str, conc: int, n: int, texts: list, voice: str, fmt: str,
                    endpoint: str, speed: float = 1.0, keepalive: bool = False,
                    speed_mix: list | None = None):
    # 默认不复用连接：max_keepalive_connections=0 → 每请求新建连接、用完即关，
    # 结构性规避服务端 KEEP_ALIVE_TIMEOUT_S=0 与客户端连接池的关闭竞态
    # （Server disconnected——曾污染压测 ~0.3% 结果）。服务端本就是逐请求
    # accept，两侧语义对齐；回环建连开销（亚毫秒）相对秒级 TTS 可忽略。
    # --keepalive 才启用连接池，专门用于对比测量竞态本身的影响。
    client = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(
            retries=2,  # 建连失败的传输级重试（不复用模式下仅覆盖初始建连）
            limits=httpx.Limits(
                max_connections=n,
                max_keepalive_connections=20 if keepalive else 0),
        ),
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    # 信号量控并发：真 -c 语义（旧实现用连接数上限且 c+1，档位失真）；
    # 计时在信号量获取之后开始，延迟不含客户端排队时间。
    # 文本轮转 texts[i % len]：--random 模式下每请求不同文本打穿 g2p 缓存
    sem = asyncio.Semaphore(conc)

    async def bounded(i: int):
        async with sem:
            sp = speed_mix[i % len(speed_mix)] if speed_mix else speed
            return await one_request(client, url, texts[i % len(texts)], voice,
                                     fmt, endpoint, sp)

    try:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(bounded(i) for i in range(n)))
        wall = time.perf_counter() - t0
    finally:
        await client.aclose()

    oks = [r for r in results if r[0]]
    lats = [r[1] for r in oks]
    audio_total = sum(r[2] for r in oks)
    chars_total = sum(len(texts[i % len(texts)]) for i in range(n))
    fails = [r[3] for r in results if not r[0]]
    retried = sum(1 for r in results if r[4])
    return {
        "conc": conc,
        "n": n,
        "wall": wall,
        "ok": len(oks),
        "fail": len(fails),
        "retried": retried,
        "errs": sorted(set(f.strip()[:80] for f in fails))[:3],
        "reqps": len(oks) / wall if wall else 0,
        "realtime": audio_total / wall if wall else 0,
        "charsps": chars_total / wall if wall else 0,
        "mean": statistics.mean(lats) if lats else 0,
        # 最近邻秩法（nearest-rank 的 x≥ceil(p*n) 取上界）：小样本下不偏乐观
        "p95": sorted(lats)[max(0, math.ceil(len(lats) * 0.95) - 1)] if lats else 0,
    }


async def main():
    ap = argparse.ArgumentParser(description="Kokoro TTS 并发压测")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("-c", "--concurrency", default="1,2,4,8", help="逗号分隔的并发档位")
    ap.add_argument("-n", "--requests", type=int, default=16, help="每档位总请求数")
    ap.add_argument("--text", default="华为昇腾910B是一颗面向人工智能训练与推理的高性能处理器，"
                                     "支持全场景人工智能应用。今天我们来测试语音合成的并发吞吐能力。")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="改用 N 条互不相同的随机文本轮转（推荐：固定 --text 会全部"
                         "命中 g2p 句子缓存，吞吐虚高、测不到真实路径）")
    ap.add_argument("--voice", default="zf_001")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="语速；探显存上界用 0.5（帧数翻倍的最坏情况）")
    ap.add_argument("--speed-mix", default="",
                    help="按请求轮转的混合语速，如 0.8,1.0,1.0,1.0,1.5"
                         "（以 1.0 为主、少量快/慢——测速度混批对吞吐的影响；"
                         "设置后覆盖 --speed）")
    ap.add_argument("--format", default="wav", choices=["wav", "pcm", "mp3", "flac"])
    ap.add_argument("--endpoint", default="speech", choices=["speech", "tts"],
                    help="压测端点: speech=OpenAI 兼容 / tts=业务接口(含时间戳)")
    ap.add_argument("--keepalive", action="store_true",
                    help="复用连接（默认不复用：每请求新建连接规避 keep-alive "
                         "关闭竞态；此开关用于对比测量竞态本身的影响）")
    args = ap.parse_args()

    if args.random > 0:
        texts = random_texts(args.random)
        lens = sorted(len(t) for t in texts)
        desc = (f"随机文本 x{len(texts)}（互异长短混合，轮转打穿 g2p 缓存）"
                f"  长度 min/中位/max={lens[0]}/{lens[len(lens) // 2]}/{lens[-1]} 字")
    else:
        texts = [args.text]
        desc = f"文本={len(args.text)} 字（注意：固定文本会命中 g2p 缓存）"
    speed_mix = None
    if args.speed_mix:
        speed_mix = [float(x) for x in args.speed_mix.split(",") if x.strip()]
        if not speed_mix:
            sys.exit("--speed-mix 解析为空")
    mix_desc = (f"  语速混合={speed_mix}" if speed_mix
                else f"  语速={args.speed}")
    print(f"目标={args.url}  端点=/{args.endpoint}  档位={args.concurrency}  每档={args.requests} 请求\n"
          f"{desc}{mix_desc}  音色={args.voice}  格式={args.format}  "
          f"连接={'keepalive池' if args.keepalive else '不复用(每请求新建)'}\n")

    # 预热
    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.get(f"{args.url}/health")
        r.raise_for_status()
        print(f"health: {r.json()}")
        ok, lat, _, err, _ = await one_request(c, args.url, "预热。", args.voice, args.format, args.endpoint)
        if not ok:
            sys.exit(f"预热请求失败: {err}")
        print(f"预热请求 OK ({lat:.2f}s)\n")

    rows = []
    for conc in [int(x) for x in args.concurrency.split(",")]:
        row = await run_level(args.url, conc, args.requests, texts, args.voice,
                              args.format, args.endpoint, args.speed,
                              keepalive=args.keepalive, speed_mix=speed_mix)
        rows.append(row)
        line = (f"并发={row['conc']:>2}  成功={row['ok']}/{row['n']}  "
                f"req/s={row['reqps']:7.2f}  realtime={row['realtime']:6.1f}x  "
                f"chars/s={row['charsps']:8.1f}  mean={row['mean']:6.2f}s  p95={row['p95']:6.2f}s")
        if row["retried"]:
            line += f"  [竞态重试成功 x{row['retried']}]"
        print(line)
        for e in row["errs"]:
            print(f"    失败原因: {e}")

    print("\n>>> 拐点判断: realtime 不再随并发上涨、p95 明显恶化 => 该并发档位已到单卡上限，"
          "GATEWAY_WORKERS/副本数取拐点前一级即可。")


if __name__ == "__main__":
    asyncio.run(main())

"""固定 seed 的随机测试数据生成：可复现、防固定语料过拟合。

约定：每个测试自建 random.Random(seed)，只经本模块取语料，不直接 choice。
"""
import random

# 中文音节词表：拼出来的句子无语义，仅作载荷/断言锚点
SYLLABLES = ["你好", "世界", "今天", "天气", "很好", "语音", "合成", "模型",
             "服务", "稳定", "性能", "测试", "生产", "部署", "监控", "日志",
             "队列", "批次", "音频", "文本"]
_ENDINGS = "。！？；"


def text(rng: random.Random, min_chars: int = 2, max_chars: int = 24) -> str:
    """随机长度的拼接文本（无标点）。"""
    n = rng.randint(min_chars, max_chars)
    out = ""
    while len(out) < n:
        out += rng.choice(SYLLABLES)
    return out[:n]


def sentence(rng: random.Random, min_chars: int = 2, max_chars: int = 24) -> str:
    """随机单句（带随机终止标点，供切句/G2P 载荷用）。"""
    return text(rng, min_chars, max_chars) + rng.choice(_ENDINGS)


def distinct_ints(rng: random.Random, lo: int, hi: int, k: int) -> list:
    """[lo, hi] 内 k 个互不相同的随机整数（k ≤ 区间宽度）。"""
    pool = list(range(lo, hi + 1))
    rng.shuffle(pool)
    return pool[:k]

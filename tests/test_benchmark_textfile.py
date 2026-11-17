"""benchmark --text-file 文档摘录回归。

契约（scripts/benchmark.py::load_doc / file_texts）：
  - 预检先于压测：文件缺失 → OSError；UTF-8（含 BOM）与 GBK 解码自适应；
    两者都失败 → ValueError；空文档/长度 < max_chars → ValueError
  - 摘录均为文档子串、长度 ≤ max_chars、固定 seed 复现、互异
  - 长短分层保留（与 random_texts 同配额），起止对齐句末标点

测试文档是测试文件内联的 ~45 条互异真实感句子（总长 ≥800 字，不引入
任何语料生成机制）。"""
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def benchmark_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark

    return benchmark


# 内联压测语料：~45 条互不相同的真实感句子（长短混合、句末标点齐全），
# 总长 ≈840 字 ≥ 默认摘录上限。不用重复块拼长度——短层摘录取整句，
# 重复块会让互异性断言失真（真实文档也不会逐字重复）。
_CORPUS = (
    "压测开始。"
    "华为昇腾910B是一颗面向人工智能训练与推理的高性能处理器。"
    "语音合成服务在高并发场景下的吞吐表现直接决定用户体验！"
    "本次压测覆盖短句、常规句以及多句长单的混合流量。"
    "网关受理、音素化、批量前向与字级对齐各有独立的处理路径？"
    "音频块经共享内存零拷贝传递，聚合完成后统一编码返回。"
    "批处理窗口把同长度档位的请求拼成一次前向计算。"
    "认领队列用原子脚本完成组批与故障重派登记。"
    "字级时间戳由对齐模型按句分组推理后拼接而成。"
    "多音字覆盖在请求级注入词典，用完立即还原。"
    "显存预算按批宽与句长自动推导并提前预占。"
    "心跳文件记录每个工人进程的忙闲状态。"
    "看门狗发现主循环卡死会强制重启工人。"
    "队列深度与排队时长一起构成背压信号。"
    "拒绝响应携带机器可读的原因标记。"
    "日志按请求标识串起全程阶段耗时。"
    "共享内存水位超过九成时拒绝新请求。"
    "清道夫定期回收无主的音频块。"
    "预热批次按最坏形状提前撑满缓存池。"
    "断连请求的中间产物会被立即回收。"
    "指标端点聚合全部工人的统计视图。"
    "慢工人的在途任务由监督者精确重派。"
    "音色列表在启动时扫描一次模型缓存。"
    "浮点精度按请求携带进认领键防混批。"
    "语速下限保护长句不会超出等待预算。"
    "健康检查区分就绪与存活两种探针。"
    "优雅退出先摘心跳再排空在途队列。"
    "随机文本轮转是打穿缓存的前提。"
    "并发档位逐级抬升直到吞吐出现拐点。"
    "预热完成后长文本的分配器不再增长。"
    "对齐账本与音频账本在网关收齐后返回。"
    "内嵌缓存服务走套接字免去端口管理。"
    "时间戳段数必须与可对齐字符数一致。"
    "零时长窗口也要按字符数展开成段。"
    "多句长单会触发对齐的多组切分路径。"
    "超长请求是占用闸门额度的最坏情况。"
    "压测拐点决定单卡该跑几个工人。"
    "短句命中的缓存与长单的实时路径要分开看。"
    "文本轮转条数至少覆盖请求总数才会全部互异。"
    "文档摘录模式让压测语料贴近真实业务分布。"
    "语速混合档位能测出批内速度分桶的影响。"
    "音频时长与合成耗时之比是实时倍速的定义。"
    "第95百分位延迟比平均值更能反映长尾体验。"
    "输出格式由编码器按请求参数即时转换。"
    "错误码区分校验失败与下游过载两类。"
)
assert len(_CORPUS) >= 800, "内联语料须满足默认摘录上限"


def _doc() -> str:
    return _CORPUS


def _doc_prefix(chars: int) -> str:
    """截取前 chars 字符（可能切在句中——文档本身无需整句收尾）。"""
    return _CORPUS[:chars]


def _write(tmp_path, data, name="corpus.txt", encoding="utf-8"):
    p = tmp_path / name
    if isinstance(data, bytes):
        p.write_bytes(data)
    else:
        p.write_text(data, encoding=encoding)
    return str(p)


# ── 预检：可读性 / 编码 / 长度 ──────────────────────────────────

def test_missing_file_raises_oserror(benchmark_mod, tmp_path):
    with pytest.raises(OSError):
        benchmark_mod.load_doc(str(tmp_path / "nope.txt"))


def test_utf8_gbk_bom_decode_identically(benchmark_mod, tmp_path):
    """同一文档的三种编码形态（UTF-8 / 带 BOM / GBK）必须读出同一内容，
    摘录结果逐字节一致。"""
    doc = _doc()
    utf8 = _write(tmp_path, doc)
    bom = _write(tmp_path, doc.encode("utf-8-sig"), name="bom.txt")
    gbk = _write(tmp_path, doc.encode("gbk"), name="gbk.txt")
    assert benchmark_mod.load_doc(utf8) == benchmark_mod.load_doc(bom) \
        == benchmark_mod.load_doc(gbk) == doc
    assert benchmark_mod.file_texts(utf8, 10, seed=7) \
        == benchmark_mod.file_texts(bom, 10, seed=7) \
        == benchmark_mod.file_texts(gbk, 10, seed=7)


def test_undecodable_bytes_raise_with_both_encodings(benchmark_mod, tmp_path):
    """UTF-8 与 GBK 都解不开的字节 → ValueError，消息写明已尝试的两种编码。"""
    p = _write(tmp_path, b"\xff\xfe\x81", name="bad.txt")
    with pytest.raises(ValueError, match="UTF-8.*GBK|GBK.*UTF-8"):
        benchmark_mod.load_doc(p)


def test_empty_document_rejected(benchmark_mod, tmp_path):
    for data in ("", "  \n\t  ", "\xef\xbb\xbf".encode("latin-1")):
        p = _write(tmp_path, data, name="empty.txt")
        with pytest.raises(ValueError, match="文档为空"):
            benchmark_mod.load_doc(p)


def test_too_short_document_rejected_with_lengths(benchmark_mod, tmp_path):
    doc = _doc_prefix(300)
    assert len(doc) < 800
    p = _write(tmp_path, doc, name="short.txt")
    with pytest.raises(ValueError) as ei:
        benchmark_mod.load_doc(p)
    msg = str(ei.value)
    assert str(len(doc)) in msg and "800" in msg, msg


def test_max_chars_relaxes_short_document(benchmark_mod, tmp_path):
    """文档偏短的逃生口：--max-chars 调低后可正常摘录，摘录仍 ≤ 上限。"""
    doc = _doc_prefix(510)                     # < 800：默认拒绝
    p = _write(tmp_path, doc, name="mid.txt")
    with pytest.raises(ValueError, match="文档太短"):
        benchmark_mod.load_doc(p)
    texts = benchmark_mod.file_texts(p, 10, max_chars=500)
    assert len(texts) == 10
    assert all(len(t) <= 500 for t in texts)


# ── 摘录契约 ────────────────────────────────────────────────────

def test_excerpts_are_distinct_substrings_capped(benchmark_mod, tmp_path):
    doc = _doc()
    p = _write(tmp_path, doc)
    texts = benchmark_mod.file_texts(p, 30)
    assert len(texts) == 30 and len(set(texts)) == 30      # 互异
    assert all(t in doc for t in texts)                     # 全部是文档子串
    assert all(0 < len(t) <= 800 for t in texts)            # 非空且 ≤ 上限
    assert all(t == t.strip() for t in texts)               # 无首尾空白


def test_reproducible_with_fixed_seed(benchmark_mod, tmp_path):
    p = _write(tmp_path, _doc())
    assert benchmark_mod.file_texts(p, 30) == benchmark_mod.file_texts(p, 30)


def test_length_layering_and_sentence_alignment(benchmark_mod, tmp_path):
    """长短分层与句边界契约：短层=整句形态（≤30 字）、长/超长层存在、
    全部摘录以句末标点收尾（g2p 按句切组可测）。内联文档句长 20~29 字，
    句边界对齐的回退/延伸最多漂移一句，故阈值留出该漂移量。"""
    doc = _doc()
    p = _write(tmp_path, doc)
    texts = benchmark_mod.file_texts(p, 30)
    lens = [len(t) for t in texts]
    assert min(lens) <= 30                                  # 短层存在（整句）
    assert sum(1 for L in lens if L > 90) >= 8              # 长/超长层整体
    assert max(lens) >= 350                                 # 超长层（接近上限）
    assert all(t[-1] in "。！？；" for t in texts), \
        [t[-8:] for t in texts if t[-1] not in "。！？；"]


def test_long_excerpts_trigger_multichunk_paths(benchmark_mod, tmp_path):
    """>TTS_CHUNK_CHARS(120) 的长单与 ≥350 的超长单都要出现——多句 chunk、
    多批聚合与对齐多组切分路径才有覆盖。"""
    texts = benchmark_mod.file_texts(_write(tmp_path, _doc()), 30)
    lens = [len(t) for t in texts]
    assert sum(1 for L in lens if L > 120) >= 6
    assert max(lens) >= 350


def test_custom_count_via_random_arg_semantics(benchmark_mod, tmp_path):
    """--text-file 模式条数沿用 --random N：n=5 各层保底 1 条。"""
    texts = benchmark_mod.file_texts(_write(tmp_path, _doc()), 5)
    assert len(texts) == 5
    lens = [len(t) for t in texts]
    assert max(lens) >= 350 and min(lens) <= 30             # 分层保底仍成立

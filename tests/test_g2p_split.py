"""g2p 语速感知切句上限（_chunk_cap）：单 chunk 音频 ≲28s 的根因层保证。

asr 组帧数硬封顶（npu_ops._asr_auto_budget_mb 预算前提）两层保证的
第一层；第二层 funasr_ts._split_part 硬切在 tests/test_funasr_math.py。"""
import g2p


def test_chunk_cap_default_speed_unchanged(monkeypatch):
    """speed 1.0/4.0：128×speed ≥ 默认上限 → 与固定 120 字行为完全一致。"""
    monkeypatch.setattr(g2p, "_max_chunk", 120)
    assert g2p._chunk_cap(1.0) == 120
    assert g2p._chunk_cap(4.0) == 120


def test_chunk_cap_low_speed_tightens(monkeypatch):
    """低速线性收紧：0.5 → 64 字（≈28s）；0.7 → 89；0.9375 恰等于上限。"""
    monkeypatch.setattr(g2p, "_max_chunk", 120)
    assert g2p._chunk_cap(0.5) == 64
    assert g2p._chunk_cap(0.7) == 89
    assert g2p._chunk_cap(0.9375) == 120


def test_chunk_cap_floor(monkeypatch):
    """极端低速钳 floor 8（= TTS_CHUNK_CHARS 下限），不出现 0/负数。"""
    monkeypatch.setattr(g2p, "_max_chunk", 120)
    assert g2p._chunk_cap(0.001) == 8


def test_split_text_respects_speed_cap(monkeypatch):
    """split_text 在收紧上限下合并/硬切均遵守 cap，内容不丢。"""
    monkeypatch.setattr(g2p, "_max_chunk", 120)
    text = "今天天气不错。我们去公园散步吧顺便买点咖啡。" * 4
    cap = g2p._chunk_cap(0.5)
    sents = g2p.split_text(text, cap)
    assert sents
    assert all(len(s) <= cap for s in sents)
    assert len("".join(sents)) == len(text)    # 逐句拼接 = 原长（内容不丢）


def test_worker_uses_speed_aware_cap():
    """调用点锁定：g2p 主循环必须用 _chunk_cap(speed)，不能回退成固定
    _max_chunk——否则根因层的语速收紧静默失效（单测只测函数本身测不出）。"""
    from pathlib import Path

    src = Path(g2p.__file__).read_text(encoding="utf-8")
    assert 'split_text(req["text"], _chunk_cap(speed))' in src


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_g2p_split.py")

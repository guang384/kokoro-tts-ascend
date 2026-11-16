"""bind_char_timestamps：单元→字符时间戳绑定契约（无模型，纯函数）。

单元清单按 FunASR 分词规则从输入文本重建（_tokenize_units），不依赖
generate 返回的 text 字段——group_timestamp/batch_group_timestamp 回填的
text 是输入原文（"".join(texts)），不是分词回显（20261116 生产 WARNING
取证，探针脚本见 git 历史）。

分词规则真机实证：中文逐字、ASCII 字母串/数字串整体（FunASR 回显英文
大写，绑定用 casefold 归一）、标点不产生单元。"""
import funasr_ts


def _assert_contract(out, n_expect):
    """公共断言：段数 == 期望；起点单调不减；段内有序。"""
    assert len(out) == n_expect, f"段数 {len(out)} != {n_expect}"
    for a, b in zip(out, out[1:]):
        assert a[0] <= b[0] and a[1] <= b[1], f"非单调: {out}"


def test_exact_chinese_passthrough():
    """中文逐字单元：窗口原样透传（1:1 不动窗口）。"""
    out = funasr_ts.bind_char_timestamps("你好。", [[0, 100], [100, 200]])
    assert out == [[0, 100], [100, 200]]


def test_english_word_split_evenly():
    """英文词多字单元："Token" 一段 760ms → 五字母均分（真实案例形态）。"""
    out = funasr_ts.bind_char_timestamps(
        "刷Token调用行为",
        [[0, 100], [100, 860], [860, 1020], [1020, 1180], [1180, 1340],
         [1340, 1500]])
    _assert_contract(out, 10)           # 刷 T o k e n 调 用 行 为
    assert out[0] == [0, 100]
    assert out[1][0] == 100 and out[5][1] == 860        # Token 段的边界守恒
    assert out[6] == [860, 1020] and out[-1] == [1340, 1500]
    spans = [e - s for s, e in out[1:6]]
    assert max(spans) - min(spans) <= 1                 # 五份等宽


def test_tokenizer_rule_matches_funasr_probe():
    """分词规则与真机探针实证一致（含英文大写归一）。"""
    assert funasr_ts._tokenize_units("刷Token调用行为") == \
        ["刷", "Token", "调", "用", "行", "为"]
    echo = ["刷", "TOKEN", "调", "用", "行", "为"]      # FunASR 回显（探针实测）
    assert [u.casefold() for u in funasr_ts._tokenize_units("刷Token调用行为")] == \
        [e.casefold() for e in echo]


def test_digit_group_split():
    """数字串整体单元："2024" 一段 → 四个数字均分。"""
    out = funasr_ts.bind_char_timestamps("2024年", [[0, 800], [800, 1000]])
    _assert_contract(out, 5)
    assert out[0][0] == 0 and out[3][1] == 800 and out[4] == [800, 1000]


def test_punctuation_never_counts():
    """标点不产生单元：带标点文本的重建单元数 == 可对齐字符数
    （生产 blob 场景——原文回填当分词用——在此契约下不可能复现）。"""
    out = funasr_ts.bind_char_timestamps(
        "你好，世界。", [[0, 100], [100, 200], [200, 300], [300, 400]])
    _assert_contract(out, 4)
    assert out == [[0, 100], [100, 200], [200, 300], [300, 400]]


def test_swallowed_units_positional_fallback():
    """② 对齐吞段（重建单元数 > 时间戳数）→ 位置比例铺，段数契约不破。"""
    out = funasr_ts.bind_char_timestamps(
        "杜绝此类行为", [[0, 200], [200, 400]])
    _assert_contract(out, 6)
    assert out[0][0] == 0 and out[-1][1] == 400         # 铺满既有窗口


def test_unit_count_surplus_fallback():
    """② 时间戳多于单元（异常多段）→ 位置比例铺，段数仍正确。"""
    out = funasr_ts.bind_char_timestamps(
        "你好世界", [[0, 100], [100, 200], [200, 300], [300, 400],
                     [400, 500]])
    _assert_contract(out, 4)
    assert out[0][0] == 0


def test_empty_paths_return_empty():
    """失败契约：无时间戳 / 无可对齐字符 → 空列表。"""
    assert funasr_ts.bind_char_timestamps("你好。", []) == []
    assert funasr_ts.bind_char_timestamps("。！？", [[0, 100]]) == []
    assert funasr_ts.bind_char_timestamps("", [[0, 100]]) == []


def test_monotone_enforced_on_malformed_ts():
    """异常输入（时间戳倒退）→ 输出仍被钳制为单调不减。"""
    out = funasr_ts.bind_char_timestamps("你好", [[100, 50], [40, 200]])
    _assert_contract(out, 2)
    assert out[0][0] <= out[1][0] <= out[1][1]

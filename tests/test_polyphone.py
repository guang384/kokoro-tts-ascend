"""多音字修正：内置表 + KOKORO_G2P_DICT（文件/内联）+ 解析健壮性。

生产反馈：'长文本' 念成 zhǎng（misaki 靠 jieba 单字消歧的默认错误）。
修复机制 = jieba.add_word（词组切整词）+ pypinyin.load_phrases_dict
（读音），业务方经 env 词典自行扩展。"""
import os
import subprocess
import sys

import pytest

import pipeline


def _env_with(dict_value: str) -> dict:
    env = dict(os.environ)
    env["KOKORO_G2P_DICT"] = dict_value
    env["DEVICE"] = "cpu"
    return env


def test_builtin_long_fixes_loaded():
    """内置表包含高频误判；解析函数无 env 时不低于内置条数。"""
    fixes = pipeline._load_polyphone_fixes()
    assert fixes["长文本"] == ["chang2", "wen2", "ben3"]
    assert fixes["很长"] == ["hen3", "chang2"]


def test_inline_dict_parsed():
    """内联模式（分号分隔）解析进词典。"""
    old = os.environ.get("KOKORO_G2P_DICT")
    os.environ["KOKORO_G2P_DICT"] = "银行 yin2 xing2; 角色 jiao3 se4"
    try:
        fixes = pipeline._load_polyphone_fixes()
        assert fixes["银行"] == ["yin2", "xing2"]
        assert fixes["角色"] == ["jiao3", "se4"]
    finally:
        if old is None:
            os.environ.pop("KOKORO_G2P_DICT", None)
        else:
            os.environ["KOKORO_G2P_DICT"] = old


def test_file_dict_parsed(tmp_path):
    """文件模式（K8s ConfigMap 挂载形态）。"""
    p = tmp_path / "g2p.dict"
    p.write_text("重庆 chong2 qing4\n", encoding="utf-8")
    old = os.environ.get("KOKORO_G2P_DICT")
    os.environ["KOKORO_G2P_DICT"] = str(p)
    try:
        fixes = pipeline._load_polyphone_fixes()
        assert fixes["重庆"] == ["chong2", "qing4"]
    finally:
        if old is None:
            os.environ.pop("KOKORO_G2P_DICT", None)
        else:
            os.environ["KOKORO_G2P_DICT"] = old


def test_bad_path_and_bad_line_do_not_crash(tmp_path):
    """坏路径静默回退内置；坏行（单字段）跳过不崩。"""
    old = os.environ.get("KOKORO_G2P_DICT")
    os.environ["KOKORO_G2P_DICT"] = str(tmp_path / "nope.dict")
    try:
        assert "长文本" in pipeline._load_polyphone_fixes()
        p = tmp_path / "bad.dict"
        p.write_text("只有词组没有拼音\n好 hao3\n", encoding="utf-8")
        os.environ["KOKORO_G2P_DICT"] = str(p)
        fixes = pipeline._load_polyphone_fixes()
        assert fixes["好"] == ["hao3"]      # 坏行跳过，好行使能
    finally:
        if old is None:
            os.environ.pop("KOKORO_G2P_DICT", None)
        else:
            os.environ["KOKORO_G2P_DICT"] = old


@pytest.mark.parametrize("text,expect_ipa,unexpect_ipa", [
    ("长文本", "ㄔㄤ2", "ㄓㄤ3"),      # 内置修复
    ("生长", "ㄓㄤ3", None),           # 不回归
])
def test_g2p_end_to_end(text, expect_ipa, unexpect_ipa):
    """端到端（子进程隔离 jieba 全局态）：真 G2P 输出含期望音素。"""
    code = (
        "import os;os.environ['DEVICE']='cpu';"
        "import pipeline;"
        f"out=pipeline.get_g2p().phonemize({text!r})[0];print(out)"
    )
    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/python_data/hf-cache")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stderr[-500:]
    assert expect_ipa in r.stdout, r.stdout
    if unexpect_ipa:
        assert unexpect_ipa not in r.stdout, r.stdout

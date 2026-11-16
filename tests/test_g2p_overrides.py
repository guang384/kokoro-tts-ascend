"""per-request 多音字覆盖（g2p_overrides）：校验 / 请求契约 / 生效与还原 / 并发隔离。

生产需求：专有名词只在某次请求里读特定音（"长乐" 平时 chang2 le4，某次
请求要 zhang3 le4）——不能改全局词典（并发请求互相污染），实现为
"锁内临时注入 pypinyin PHRASES_DICT → 音素化 → 精确还原"。
"""
import asyncio
import os
import subprocess
import sys

import pytest

import protocol


# ── 规范化/校验（纯函数）──
def test_normalize_string_and_list_values():
    assert protocol.normalize_overrides({"长文本": "chang2 wen2 ben3"}) \
        == {"长文本": ["chang2", "wen2", "ben3"]}
    assert protocol.normalize_overrides({"长文本": ["chang2", "wen2", "ben3"]}) \
        == {"长文本": ["chang2", "wen2", "ben3"]}
    assert protocol.normalize_overrides(None) == {}
    assert protocol.normalize_overrides({}) == {}


@pytest.mark.parametrize("bad,frag", [
    ({"长文本": "chang wen ben"}, "TONE3"),          # 无声调数字
    ({"长文本": "chang2 wen2"}, "音节数"),            # 音节数≠字数
    ({" ": "chang2"}, "非法"),                        # 空/空白词
    ({"a" * 33: "chang2"}, "非法"),                   # 超长词
    ({"长文本": []}, "缺少拼音"),                      # 空读音
    ({"长文本": ["chang2", "wen2", "ben3", "x"]}, "TONE3"),
    ({"3D打印": "san1 dian4"}, "常用汉字"),            # 混合词：走英文路径，覆盖不可能生效
    ({"㐀字": "qi1 zi4"}, "常用汉字"),                # ext-A 生僻字：jieba 标 x 直接跳过音素化
])
def test_normalize_rejects_bad_input(bad, frag):
    with pytest.raises(ValueError) as ei:
        protocol.normalize_overrides(bad)
    assert frag in str(ei.value)


def test_normalize_word_count_cap():
    with pytest.raises(ValueError, match="词条过多"):
        protocol.normalize_overrides({f"词{i}": "ci2" for i in range(201)})


def test_new_request_carries_overrides_only_when_set():
    base = protocol.new_request("r1", 0, "你好", "zf_001", 1.0, True)
    assert "g2p_overrides" not in base            # 普通请求报文不膨胀
    msg = protocol.new_request("r2", 0, "你好", "zf_001", 1.0, True,
                               g2p_overrides={"长乐": ["zhang3", "le4"]})
    assert msg["g2p_overrides"] == {"长乐": ["zhang3", "le4"]}


def test_injection_failure_rolls_back(monkeypatch):
    """注入中途失败（词典坏态/异常）→ 分词器、读音词典、jieba 词频零残留。

    曾漏 finally（注入在 try 之外）：第 2 个词失败时第 1 个词的临时读音
    与 freq=10000 永久留在全局态，后续请求被悄悄污染。"""
    import jieba
    import pypinyin.core as pc

    import pipeline

    words = ("垚词", "垚语")           # 词典外生造词
    seg_before = pc.seg
    dict_before = {w: pc.PHRASES_DICT.get(w) for w in words}
    freq_before = {w: jieba.dt.FREQ.get(w, 0) for w in words}
    real_add = jieba.add_word
    calls = {"n": 0}

    def flaky(word, freq=0, **kw):
        calls["n"] += 1
        if calls["n"] == 2:           # 第 2 个词注入时炸
            raise RuntimeError("injected failure")
        return real_add(word, freq=freq, **kw)

    monkeypatch.setattr(jieba, "add_word", flaky)
    with pytest.raises(RuntimeError, match="injected failure"):
        pipeline._apply_request_overrides({
            "垚词": ["yao2", "ci2"], "垚语": ["yao2", "yu3"]})
    assert pc.seg is seg_before
    for w in words:
        assert pc.PHRASES_DICT.get(w) == dict_before[w], w
        assert jieba.dt.FREQ.get(w, 0) == freq_before[w], w


# ── API 边界校验（handler 在最前面拒绝，不会触达流水线）──
def test_tts_api_rejects_bad_overrides():
    from gateway import BizError, TTSRequest, tts_business

    req = TTSRequest(text="你好", g2p_overrides={"长文本": "chang wen ben"})
    with pytest.raises(BizError) as ei:
        asyncio.run(tts_business(req))
    assert ei.value.status_code == 400
    assert "TONE3" in ei.value.message


def test_v1_api_rejects_bad_overrides():
    from fastapi import HTTPException

    from gateway import SpeechRequest, create_speech

    req = SpeechRequest(input="你好", g2p_overrides={"长文本": "chang2 wen2"})
    with pytest.raises(HTTPException) as ei:
        asyncio.run(create_speech(req))
    assert ei.value.status_code == 400
    assert "音节数" in str(ei.value.detail)


# ── 真 G2P 端到端（子进程隔离 jieba/pypinyin 全局态）──
_ENV_CODE = """
import os;os.environ.setdefault('DEVICE','cpu')
import pipeline
g = pipeline.get_g2p()
def ph(s, o=None): return g.phonemize(s, o)[0]
print('DEFAULT', ph('欢迎长乐平台'))                                  # chang2 le4
print('OVERRIDE', ph('欢迎长乐平台', {'长乐': 'zhang3 le4'}))          # zhang3 le4
print('RESTORED', ph('欢迎长乐平台'))                                  # 回到 chang2 le4
print('BUILTIN', ph('这是长文本测试'))                                 # 内置修复 chang2
print('FLIPPED', ph('这是长文本测试', {'长文本': 'zhang3 wen2 ben3'}))  # 请求级翻转
print('BUILTIN2', ph('这是长文本测试'))                                # 内置修复未被破坏
"""


def test_override_effective_and_restored():
    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/python_data/hf-cache")
    r = subprocess.run([sys.executable, "-c", _ENV_CODE], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    out = r.stdout
    assert "OVERRIDE" in out

    lines = {ln.split(" ", 1)[0]: ln.split(" ", 1)[1] for ln in out.splitlines()
             if ln.startswith(("DEFAULT", "OVERRIDE", "RESTORED", "BUILTIN",
                               "FLIPPED", "BUILTIN2"))}
    # 长乐：默认 chang2(ㄔㄤ2) → 覆盖 zhang3(ㄓㄤ3) → 还原 chang2
    assert "ㄔㄤ2" in lines["DEFAULT"] and "ㄓㄤ3" not in lines["DEFAULT"]
    assert "ㄓㄤ3" in lines["OVERRIDE"] and "ㄔㄤ2" not in lines["OVERRIDE"]
    assert "ㄔㄤ2" in lines["RESTORED"] and "ㄓㄤ3" not in lines["RESTORED"]
    # 内置全局修复（长文本→chang2）可被请求级覆写、且随后完好
    assert "ㄔㄤ2" in lines["BUILTIN"]
    assert "ㄓㄤ3" in lines["FLIPPED"]
    assert "ㄔㄤ2" in lines["BUILTIN2"]


_CONC_CODE = """
import os;os.environ.setdefault('DEVICE','cpu')
import threading
import pipeline
g = pipeline.get_g2p()
def ph(s, o=None): return g.phonemize(s, o)[0]
stop, bad = False, []
def plain():
    while not stop:
        out = ph('欢迎长乐平台')
        if 'ㄓㄤ3' in out: bad.append(('plain-saw-override', out))
def ovr_zhang():
    for _ in range(40):
        out = ph('欢迎长乐平台', {'长乐': 'zhang3 le4'})
        if 'ㄓㄤ3' not in out or 'ㄔㄤ2' in out: bad.append(('zhang-miss', out))
def ovr_chang():
    for _ in range(40):
        out = ph('欢迎长乐平台', {'长乐': 'chang2 le4'})
        if 'ㄔㄤ2' not in out or 'ㄓㄤ3' in out: bad.append(('chang-miss', out))
ts = [threading.Thread(target=plain) for _ in range(3)]
o1 = threading.Thread(target=ovr_zhang)
o2 = threading.Thread(target=ovr_chang)
for t in ts: t.start()
o1.start(); o2.start(); o1.join(); o2.join(); stop = True
for t in ts: t.join()
print('BAD', bad[:3])
print('AFTER', ph('欢迎长乐平台'))
"""


def test_override_thread_isolation():
    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/python_data/hf-cache")
    r = subprocess.run([sys.executable, "-c", _CONC_CODE], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    assert "BAD []" in r.stdout, r.stdout[-500:]   # 普通请求永不见临时读音
    after = [ln for ln in r.stdout.splitlines() if ln.startswith("AFTER")][0]
    assert "ㄔㄤ2" in after and "ㄓㄤ3" not in after


_EMBED_CODE = """
import os;os.environ.setdefault('DEVICE','cpu')
import pipeline
g = pipeline.get_g2p()
def ph(s, o=None): return g.phonemize(s, o)[0]
print('CHAO_DEFAULT', ph('朝阳区'))                                   # 内置 chao2
print('CHAO_OVR', ph('朝阳区', {'朝阳': 'zhao1 yang2'}))               # 覆盖优先于内置长词条
print('CHAO_EXPLICIT', ph('朝阳区', {'朝阳区': 'zhao1 yang2 qu1'}))     # 显式长词条
print('OVERLAP', ph('长文本', {'长文': 'zhang3 wen2',
                               '长文本': 'chang2 wen2 ben3'}))          # 最长匹配
print('AFTER', ph('朝阳区'))                                          # 还原
"""


def test_override_embedded_and_longest_match():
    """覆盖词嵌在更长 token/内置短语内：最长匹配、显式覆盖优先。

    内置短语词典含"朝阳区"(chao2)，用户只覆盖"朝阳"(zhao1)时旧 retrain
    实现会被长词条压过；现实现让用户覆盖生效（请求级意图优先）。"""
    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/python_data/hf-cache")
    r = subprocess.run([sys.executable, "-c", _EMBED_CODE], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    lines = {ln.split(" ", 1)[0]: ln.split(" ", 1)[1]
             for ln in r.stdout.splitlines() if " " in ln}
    assert "ㄔㄠ2" in lines["CHAO_DEFAULT"]
    assert "ㄓㄠ1" in lines["CHAO_OVR"] and "ㄔㄠ2" not in lines["CHAO_OVR"]
    assert "ㄓㄠ1" in lines["CHAO_EXPLICIT"]
    assert "ㄔㄤ2" in lines["OVERLAP"] and "ㄓㄤ3" not in lines["OVERLAP"]
    assert "ㄔㄠ2" in lines["AFTER"]


_SANDHI_CODE = """
import os;os.environ.setdefault('DEVICE','cpu')
import pipeline
g = pipeline.get_g2p()
def ph(s, o=None): return g.phonemize(s, o)[0]
print('PLAIN', ph('淼垚好'))                                  # 无覆盖基线
print('OVR33', ph('淼垚好', {'淼垚': 'miao3 yao3'}))          # 3-3：sandhi 把首字变 2 声
print('OVR_NEUTRAL', ph('买东西', {'东西': 'dong1 xi1'}))     # 轻声词表把 xi1 归 5 声
print('NO_UNK', all('❓' not in x for x in [ph('淼垚好', {'淼垚': 'miao3 yao3'})]))
"""


def test_override_is_lexical_sandhi_still_applies():
    """覆盖语义 = 词典读音（lexical tone），变调规则照常作用——与内置词条
    行为一致（3-3 连读变调、强制轻声词表都会改写输出声调）。这不是缺陷：
    TTS 应产出自然口语读音；文档已注明。锁定该语义防无意改变。"""
    env = dict(os.environ)
    env.setdefault("HF_HOME", "D:/python_data/hf-cache")
    r = subprocess.run([sys.executable, "-c", _SANDHI_CODE], capture_output=True,
                       text=True, env=env, timeout=300)
    assert r.returncode == 0, r.stderr[-800:]
    lines = {ln.split(" ", 1)[0]: ln.split(" ", 1)[1]
             for ln in r.stdout.splitlines() if " " in ln}
    # 3-3 覆盖 → 首字实际读 2 声（连读变调）
    assert "ㄇ要2要3" in lines["OVR33"]
    # 东西 属强制轻声词表：xi1 被归为 xi5（词典读音语义）
    assert "ㄒㄧ5" in lines["OVR_NEUTRAL"]
    # 全程不产生 unk 字符
    assert lines["NO_UNK"] == "True"


# ── 英文 G2P 惰性构建（torch 解耦）──
def test_torch_import_probe_handles_all_failures(monkeypatch):
    """_torch_import_ok：子进程超时/启动失败一律按"不可用"处理（返回
    False 走 espeak 兜底），绝不向上抛——QEMU/坏环境下 g2p worker
    构造期必须能活。"""
    import subprocess

    import pipeline

    def timeout_run(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="python", timeout=120)

    monkeypatch.setattr(subprocess, "run", timeout_run)
    assert pipeline._torch_import_ok() is False

    def missing_run(*a, **kw):
        raise FileNotFoundError("python")

    monkeypatch.setattr(subprocess, "run", missing_run)
    assert pipeline._torch_import_ok() is False


def test_en_callable_lazy_and_cached(monkeypatch):
    """en_callable：special 词不走解析（零 torch 接触）；首次普通英文词
    才解析且结果缓存（第二次不再探测/构建）。mode=spacy 且 torch 不可
    导入时按显式配置语义不回退 espeak（禁用并告警）。"""
    import pipeline

    calls = {"probe": 0}

    def fake_probe():
        calls["probe"] += 1
        return False

    monkeypatch.setattr(pipeline, "_torch_import_ok", fake_probe)
    monkeypatch.setenv("KOKORO_EN_G2P", "spacy")
    fn = pipeline._build_en_callable()
    assert fn("Kokoro") == "kˈOkəɹO"       # special：不触发解析
    assert calls["probe"] == 0
    assert fn("Sol") == "sˈOl"
    assert calls["probe"] == 0
    assert fn("hello") == ""               # torch 不可导入 + 显式 spacy → 禁用
    assert calls["probe"] == 1             # 恰好探测一次
    assert fn("world") == ""
    assert calls["probe"] == 1             # 缓存生效，不重复探测

    # auto 模式：spacy 路径失败 → 自动落 espeak（输出非空且与旧实现一致）
    monkeypatch.setenv("KOKORO_EN_G2P", "auto")
    fn2 = pipeline._build_en_callable()
    out = fn2("hello")
    assert out and out != ""               # espeak 兜底产出
    assert calls["probe"] == 2             # 各 builder 独立探测一次（互不共享缓存）

"""默认值单一事实来源：Dockerfile ENV 必须与 proc_common.TUNABLE_PARAMS 一致。

历史漂移：Dockerfile 仍钉旧默认 6 tts / 3 asr / B4，而 README、memcalc、
supervisor 默认是 5 / 4 / B8——按文档核算容量会直接超卡。"""
import re
from pathlib import Path

import proc_common

_ROOT = Path(__file__).resolve().parents[1]


def _docker_env(name: str) -> str:
    text = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
    m = re.search(rf"^\s*{name}=(\S+?)\s*\\?$", text, re.M)
    assert m, f"Dockerfile 未设置 {name}"
    return m.group(1)


def test_docker_defaults_match_tunables():
    tunables = dict(proc_common.TUNABLE_PARAMS)
    assert _docker_env("TTS_WORKERS") == tunables["TTS_WORKERS"] == "4"
    assert _docker_env("ASR_WORKERS") == tunables["ASR_WORKERS"] == "4"
    assert _docker_env("TTS_MAX_BATCH_SIZE") == tunables["TTS_MAX_BATCH_SIZE"] == "8"
    assert _docker_env("ASR_MAX_BATCH_SIZE") == tunables["ASR_MAX_BATCH_SIZE"] == "8"


def test_requirements_fakeredis_has_lua():
    """claim/ack 走 EVAL；缺 [lua] 时干净本地安装会直接报 unknown command。"""
    text = (_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^fakeredis\[lua\]", text, re.M)

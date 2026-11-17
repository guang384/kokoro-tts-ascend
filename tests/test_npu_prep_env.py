"""镜像 ENV 漂移门控（scripts/npu_prep.py）的纯函数回归（不需要 Docker）。

锁住三类历史问题：增量 FROM 旧镜像继承 6 tts/3 asr/B4；Dockerfile 未显式
设置、由代码默认兜底的键被底座改成旧值；底座残留已废弃的业务 env。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import npu_prep


def test_docker_env_map_handles_continuations_and_ignores_comments():
    text = "\n".join([
        "# 示例：GATEWAY_WORKERS=N 不是真实默认值",
        "ENV PIP_INDEX_URL=https://example/simple \\",
        "    TTS_WORKERS=4 \\",
        "    DEVICE=",
        "LABEL org.opencontainers.image.title=x",
    ])
    env = npu_prep._docker_env_map(text)
    assert env["TTS_WORKERS"] == "4"
    assert env["DEVICE"] == ""
    assert env["PIP_INDEX_URL"] == "https://example/simple"
    assert "GATEWAY_WORKERS" not in env


def test_env_drift_flags_inherited_old_defaults_and_legacy_env():
    tunables = {"TTS_WORKERS": "4", "ASR_WORKERS": "4",
                "TTS_MAX_BATCH_SIZE": "8", "ASR_MAX_BATCH_SIZE": "8",
                "HEARTBEAT_STALE_S": "30"}
    docker = {"TTS_WORKERS": "4", "ASR_WORKERS": "4",
              "TTS_MAX_BATCH_SIZE": "8", "ASR_MAX_BATCH_SIZE": "8"}
    image = {"TTS_WORKERS": "6", "ASR_WORKERS": "3",
             "TTS_MAX_BATCH_SIZE": "4", "ASR_MAX_BATCH_SIZE": "8",
             "HEARTBEAT_STALE_S": "99", "KOKORO_LANG_CODE": "z",
             "FUNASR_MODEL_DIR": "/opt/funasr-models", "WORKER_TAG": "gw-1"}
    mismatched, missing, legacy = npu_prep._env_drift(image, docker, tunables)
    assert set(mismatched) == {"TTS_WORKERS", "ASR_WORKERS",
                               "TTS_MAX_BATCH_SIZE", "HEARTBEAT_STALE_S"}
    assert missing == {}
    assert legacy == {"KOKORO_LANG_CODE": "z"}


def test_env_drift_flags_missing_dockerfile_keys():
    tunables = {"TTS_WORKERS": "4"}
    docker = {"TTS_WORKERS": "4"}
    mismatched, missing, legacy = npu_prep._env_drift({}, docker, tunables)
    assert mismatched == {}
    assert missing == {"TTS_WORKERS": "4"}
    assert legacy == {}


def test_env_drift_checks_non_tunable_deployment_constants():
    """FUNASR_MODEL_DIR/PYTHONPATH 等不在 TUNABLE 但由 Dockerfile 固定的
    部署常量，底座镜像带旧值同样会覆盖——必须纳入门控；含 ${...} 的
    build-arg 模板跳过。"""
    tunables = {"TTS_WORKERS": "4"}
    docker = {"TTS_WORKERS": "4", "FUNASR_MODEL_DIR": "/opt/funasr-models",
              "PYTHONPATH": "/app", "HF_ENDPOINT": "${HF_ENDPOINT_BUILD}"}
    image = {"TTS_WORKERS": "4", "FUNASR_MODEL_DIR": "/old/models",
             "PYTHONPATH": "/app", "HF_ENDPOINT": "https://hf-mirror.com"}
    mismatched, missing, legacy = npu_prep._env_drift(image, docker, tunables)
    assert mismatched == {
        "FUNASR_MODEL_DIR": "image='/old/models' Dockerfile='/opt/funasr-models'"}
    assert missing == {}
    assert legacy == {}


def test_env_drift_accepts_consistent_image():
    tunables = {"TTS_WORKERS": "4", "HEARTBEAT_STALE_S": "30"}
    docker = {"TTS_WORKERS": "4"}
    image = {"TTS_WORKERS": "4", "HEARTBEAT_STALE_S": "30",
             "FUNASR_MODEL_DIR": "/opt/funasr-models"}
    assert npu_prep._env_drift(image, docker, tunables) == ({}, {}, {})


def test_env_drift_accepts_dockerfile_managed_business_key():
    """Dockerfile 显式声明的非 TUNABLE 业务键是合法部署配置，不应被当成
    底座遗留 env（否则新增一个 Dockerfile 键就会误报 legacy）。"""
    tunables = {"TTS_WORKERS": "4"}
    docker = {"TTS_WORKERS": "4", "KOKORO_NEW_FLAG": "on"}
    image = {"TTS_WORKERS": "4", "KOKORO_NEW_FLAG": "on"}
    mismatched, missing, legacy = npu_prep._env_drift(image, docker, tunables)
    assert mismatched == {} and missing == {} and legacy == {}

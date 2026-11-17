"""打包清单单一来源回归。

同一份事实曾被抄在多个地方：运行时模块清单在三个 Dockerfile 的 COPY 行、
`npu_prep.IMPORT_MODULES` 各一份；发布依赖版本在 requirements.txt 与
`npu_prep` 的检查字面量各一份；`Dockerfile.npuprep` 还手工重放了生产
Dockerfile 的 ENV 块。任何一处漏改都只在发布预检/真机启动时才暴露。

本文件把它们变成离线测试：改漏一处，pytest 立即失败。
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import npu_prep  # noqa: E402

IMAGES = ("Dockerfile", "Dockerfile.cpu", "Dockerfile.npuprep")


def _root_modules() -> set[str]:
    """仓库根目录的运行时模块文件（conftest 是测试配置，不进模块 COPY 行）。"""
    return {p.name for p in ROOT.glob("*.py")} - {"conftest.py"}


def _copy_modules(name: str) -> set[str]:
    """解析 Dockerfile 的模块 COPY 行（`COPY a.py b.py ... ./`）。"""
    text = (ROOT / name).read_text(encoding="utf-8")
    m = re.search(r"^COPY\s+((?:[A-Za-z0-9_]+\.py\s+)+)\./$", text, re.M)
    assert m, f"{name}: 找不到模块 COPY 行"
    return set(m.group(1).split())


@pytest.mark.parametrize("name", IMAGES)
def test_image_copies_all_root_modules(name):
    """新增模块漏 COPY 会在镜像内 import/启动时炸；这里离线拦住。"""
    assert _copy_modules(name) == _root_modules()


def test_images_share_one_module_manifest():
    """三个镜像的模块清单必须完全一致（CPU/arm64 只差基础镜像）。"""
    manifests = {name: _copy_modules(name) for name in IMAGES}
    assert len({frozenset(v) for v in manifests.values()}) == 1, manifests


def test_npu_prep_import_list_matches_repo_modules():
    """预检的 import 冒烟清单必须覆盖全部非 torch 根模块。"""
    expected = {p[:-3] for p in _root_modules() - {"batched_model.py"}}
    assert set(npu_prep.IMPORT_MODULES) == expected


def test_npuprep_replays_production_env_defaults():
    """增量验证镜像必须显式重放生产 Dockerfile 的全部运行期 ENV。

    底座镜像可能带旧默认值（6 tts/3 asr/B4）；只有显式重放才能覆盖。
    build-arg 模板（${...}）与 pip 构建开关不参与运行语义，排除。
    """
    prod = npu_prep._docker_env_map(
        (ROOT / "Dockerfile").read_text(encoding="utf-8"))
    verify = npu_prep._docker_env_map(
        (ROOT / "Dockerfile.npuprep").read_text(encoding="utf-8"))
    replay = {k: v for k, v in prod.items()
              if "${" not in v and not k.startswith("PIP_")}
    missing = sorted(replay.keys() - verify.keys())
    bad = {k: (verify[k], v) for k, v in replay.items()
           if k in verify and verify[k] != v}
    assert not missing and not bad, f"missing={missing} mismatched={bad}"


def test_release_pins_come_from_requirements():
    """发布门禁的期望版本必须来自 requirements.txt，而不是脚本里的字面量。"""
    pins = npu_prep._pinned_versions(
        (ROOT / "requirements.txt").read_text(encoding="utf-8"))
    missing = [n for n in npu_prep.RELEASE_PINS if n not in pins]
    assert not missing, f"requirements.txt 未精确钉住: {missing}"


def test_npuprep_funasr_pin_matches_requirements():
    """增量验证镜像的 funasr 钉版必须与 requirements.txt 一致。

    Dockerfile.npuprep 无法在 RUN 里直接读 requirements（未 COPY），因此
    这里用测试锁住两处版本；升级 funasr 时漏改任一处都会失败。
    """
    text = (ROOT / "Dockerfile.npuprep").read_text(encoding="utf-8")
    m = re.search(r'"funasr==([0-9][^"]*)"', text)
    assert m, "Dockerfile.npuprep 未显式钉 funasr 版本"
    pins = npu_prep._pinned_versions(
        (ROOT / "requirements.txt").read_text(encoding="utf-8"))
    assert m.group(1) == pins["funasr"], (
        f"Dockerfile.npuprep funasr=={m.group(1)} != requirements "
        f"funasr=={pins['funasr']}")


def test_pinned_versions_parser_ignores_non_exact_specs():
    text = ("# comment funasr==9.9.9\n"
            "funasr==1.4.21\n"
            "kokoro == 0.9.4  # trailing\n"
            "misaki[zh]==0.9.4\n"
            "numpy>=1.26\n"
            "bad==\n")
    assert npu_prep._pinned_versions(text) == {
        "funasr": "1.4.21", "kokoro": "0.9.4"}

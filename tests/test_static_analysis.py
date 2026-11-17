"""静态未定义名门禁（pyflakes）。

模块拆分曾引入 3 处 NameError：tts_worker 缺 `import os`、asr_worker 缺
`import npu_ops`、npu_ops 的 `_FRAMES_PER_SEC` 留在旧模块。这些都不在
import 期触发，Windows 离线 pytest 全部漏过，只有 pyflakes 能在跑测试前
发现（仓库无 CI，因此把检查放进 pytest 本身）。

pyflakes 已随 requirements.txt 入镜像；本机未安装时 skip 并提示。
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _targets() -> list[str]:
    """根目录 + scripts + tests 的全部 Python 源文件（不含 venv/缓存）。"""
    targets = []
    for base in (ROOT, ROOT / "scripts", ROOT / "tests"):
        targets.extend(sorted(str(p) for p in base.glob("*.py")))
    return targets


def test_pyflakes_clean():
    if importlib.util.find_spec("pyflakes") is None:
        pytest.skip("pyflakes 未安装（pip install pyflakes）；镜像内由 "
                    "requirements.txt 保证存在")
    targets = _targets()
    assert len(targets) >= 40, f"只收集到 {len(targets)} 个文件，路径检查失效"
    proc = subprocess.run([sys.executable, "-m", "pyflakes", *targets],
                          capture_output=True, text=True, cwd=ROOT, timeout=180)
    report = (proc.stdout + proc.stderr).strip()
    assert proc.returncode == 0, "pyflakes findings:\n" + report[-4000:]

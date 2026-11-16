"""导入期安全性：生产启动时序下 import gateway 不得调用 collect/告警。

生产事故（fix13）：supervisor 先写心跳，gateway 导入时 registry.register()
立即执行 collect → 发现缺员 → 调 _throttled_warn（定义在文件更后）→
NameError，gateway 起不来。测试须在**预置心跳**的子进程里导入才能复现。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _populated_health_dir(tmp_path: Path) -> Path:
    """模拟 supervisor 已先行落盘、角色尚未就绪（缺员必然触发告警路径）。"""
    snap = {"wtag": "supervisor", "role": "supervisor", "ready": True,
            "expected": {"gateway": 2, "tts": 6, "asr": 3},
            "stats": {}, "mailboxes": {}}
    (tmp_path / "worker-1.json").write_text(json.dumps(snap), encoding="utf-8")
    # 一个尚未就绪的 tts（缺员 → 告警路径；同时提供 worker 指标行）
    (tmp_path / "worker-2.json").write_text(json.dumps(
        {"wtag": "tts-0", "role": "tts", "device": None, "ready": False,
         "warm": {"kokoro": "pending"}, "stats": {}}), encoding="utf-8")
    return tmp_path


def test_gateway_import_with_heartbeat_present(tmp_path):
    """导入期不得执行 collect（懒注册），且不得因缺员告警而崩溃。"""
    d = _populated_health_dir(tmp_path)
    env = {**os.environ, "KOKORO_HEALTH_DIR": str(d), "METRICS_ENABLED": "1"}
    r = subprocess.run(
        [sys.executable, "-c", "import gateway; print('IMPORT_OK')"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"import gateway 崩溃:\n{r.stderr[-2000:]}"
    assert "IMPORT_OK" in r.stdout


def test_metrics_works_after_import_in_production_order(tmp_path):
    """导入后再抓 /metrics：懒注册首次构建应成功且指标有值。"""
    d = _populated_health_dir(tmp_path)
    env = {**os.environ, "KOKORO_HEALTH_DIR": str(d), "METRICS_ENABLED": "1"}
    code = (
        "import asyncio, gateway\n"
        "body = asyncio.run(gateway.metrics()).body.decode()\n"
        "assert 'kokoro_e2e_requests_total' in body\n"
        "assert 'kokoro_worker_ready{' in body\n"
        "print('METRICS_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"/metrics 失败:\n{r.stderr[-2000:]}"
    assert "METRICS_OK" in r.stdout


def test_health_gate_in_production_order(tmp_path):
    """同一时序下 /health 必须是 503（缺员，不得 fail-open）。"""
    d = _populated_health_dir(tmp_path)
    env = {**os.environ, "KOKORO_HEALTH_DIR": str(d)}
    code = (
        "import asyncio, gateway\n"
        "r = asyncio.run(gateway.health())\n"
        "assert r.status_code == 503, r.status_code\n"
        "print('GATE_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"/health 门闸失败:\n{r.stderr[-2000:]}"
    assert "GATE_OK" in r.stdout

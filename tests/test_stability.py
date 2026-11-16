"""稳定性改造回归：看门狗判决、心跳失败告警、setup_logging 幂等、
ack 有界投递、env 解析回退。全部离线（无模型、无真实进程）。"""
import json
import logging
import os
import threading
import time

import pytest

import proc_common
import supervisor
from proc_common import BusyWatch


# ── BusyWatch（工人侧看门狗状态）──

def test_busywatch_stuck_only_while_busy():
    w = BusyWatch()
    assert w.stuck_s() == 0.0          # 闲置恒 0：绝无误判
    w.enter()
    time.sleep(0.15)                   # stuck_s 保留 1 位小数，须睡过取整粒度
    assert w.stuck_s() >= 0.1          # busy 中增长（hang 可检测）
    w.exit()
    assert w.stuck_s() == 0.0


# ── classify_heartbeat 判决（stale / stuck / ok 三态）──

@pytest.fixture()
def hdir(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEALTH_DIR", str(tmp_path))
    monkeypatch.setattr(supervisor, "HEARTBEAT_STALE_S", 30)
    monkeypatch.setattr(supervisor, "STUCK_KILL_S", 300.0)
    return tmp_path


def _hb(hdir, pid, stuck_s=None, mtime_age=0.0, corrupt=False):
    path = hdir / f"worker-{pid}.json"
    if corrupt:
        path.write_text("{half", encoding="utf-8")
    else:
        snap = {"wtag": "tts-0", "pid": pid, "role": "tts"}
        if stuck_s is not None:
            snap["stuck_s"] = stuck_s
        path.write_text(json.dumps(snap), encoding="utf-8")
    old = time.time() - mtime_age
    os.utime(path, (old, old))
    return str(path)


def test_classify_fresh_and_idle_is_ok(hdir):
    verdict, _ = supervisor.classify_heartbeat(_hb(hdir, 101, stuck_s=0.0), time.time())
    assert verdict == "ok"


def test_classify_stale_heartbeat(hdir):
    verdict, detail = supervisor.classify_heartbeat(
        _hb(hdir, 102, mtime_age=100.0), time.time())
    assert verdict == "stale" and "stale" in detail


def test_classify_stuck_inference_hang(hdir):
    """心跳新鲜但 busy 卡死超线 → stuck（mtime 检测的盲区）。"""
    verdict, detail = supervisor.classify_heartbeat(
        _hb(hdir, 103, stuck_s=420.0), time.time())
    assert verdict == "stuck" and "inference hang" in detail


def test_classify_stuck_below_threshold_ok(hdir):
    verdict, _ = supervisor.classify_heartbeat(
        _hb(hdir, 104, stuck_s=12.0), time.time())
    assert verdict == "ok"


def test_classify_corrupt_fresh_file_tolerated(hdir):
    verdict, _ = supervisor.classify_heartbeat(_hb(hdir, 105, corrupt=True), time.time())
    assert verdict == "ok"


class _StubProc:
    def __init__(self, pid):
        self.pid = pid
        self.killed = False

    def kill(self):
        self.killed = True


def test_reap_hung_kills_stuck_child(hdir, monkeypatch):
    """看门狗闭环：心跳上报 stuck → supervisor 强杀该子进程。"""
    stub = _StubProc(pid=9999)
    monkeypatch.setattr(supervisor, "procs", {("tts", 3): stub})
    _hb(hdir, 9999, stuck_s=400.0)
    supervisor.reap_hung()
    assert stub.killed


def test_reap_hung_spares_healthy_child(hdir, monkeypatch):
    stub = _StubProc(pid=8888)
    monkeypatch.setattr(supervisor, "procs", {("tts", 0): stub})
    _hb(hdir, 8888, stuck_s=0.5)
    supervisor.reap_hung()
    assert not stub.killed


# ── 心跳失败告警（不再 except-pass 隐身）──

def test_heartbeat_once_raises_on_write_failure(monkeypatch, tmp_path):
    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(proc_common, "write_heartbeat", _boom)
    with pytest.raises(OSError):
        proc_common._heartbeat_once("tts-0", 1, dict, str(tmp_path))


def test_heartbeat_loop_warns_on_consecutive_failures(monkeypatch, tmp_path, caplog):
    """持续失败：第 1 次 + 每第 12 次告警（≈每分钟一条，不刷屏也不隐身）。"""
    def _boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(proc_common, "write_heartbeat", _boom)
    stop = threading.Event()
    t = threading.Thread(target=proc_common.heartbeat_loop,
                         args=("tts-9", 424242, dict, str(tmp_path)),
                         kwargs={"interval": 0.01, "stop": stop}, daemon=True)
    with caplog.at_level(logging.WARNING, logger="kokoro.heartbeat"):
        t.start()
        time.sleep(0.45)   # ≈45 轮 → 告警点 1/12/24/36
        stop.set()
        t.join(timeout=2)
    # 只统计本测试线程的日志：其它测试文件遗留的心跳线程也会在 _boom
    # 生效期间写同类告警（全套偶发失败的历史原因）
    warns = [r for r in caplog.records
             if r.name == "kokoro.heartbeat" and "[tts-9]" in r.message
             and "heartbeat write failed" in r.message]
    # 语义：首发 + 每 12 次一条（0.45s/0.01s ≈ 45 轮 → 4 条；Windows
    # 定时器抖动会上下浮动，只锁"有告警且未刷屏"）
    assert 2 <= len(warns) <= 10, f"告警节奏异常: {len(warns)}"


# ── setup_logging 幂等（单例 tag 过滤器，不随调用堆积）──

def test_setup_logging_idempotent(monkeypatch):
    # 清场：剥掉 root handlers 上已有 tag 过滤器，重置单例
    for h in logging.getLogger().handlers:
        h.filters = [f for f in h.filters if not hasattr(f, "tag")]
    monkeypatch.setattr(proc_common, "_wtag_filter", None)

    proc_common.setup_logging("role-a")
    proc_common.setup_logging("role-b")   # g2p 在模块 import 与 main 各调一次
    tag_filters = [f for h in logging.getLogger().handlers
                   for f in h.filters if hasattr(f, "tag")]
    # 单例可挂到 root 的每个 handler 上——不变量是"唯一实例"，不是挂载次数
    assert len({id(f) for f in tag_filters}) == 1
    assert tag_filters[0].tag == "role-b"  # 后调者生效（同进程单角色语义）



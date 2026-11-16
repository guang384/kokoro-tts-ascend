"""进程公共件·心跳层：worker 心跳原子落盘线程（supervisor 判 hang 依据）。"""
import json
import logging
import os
import threading
import time

from config import HEALTH_DIR

def write_heartbeat(health_dir: str, pid: int, snapshot: dict) -> None:
    """心跳落盘（原子替换）：worker-<pid>.json，supervisor 据此做 hang 检测。"""
    os.makedirs(health_dir, exist_ok=True)
    tmp = os.path.join(health_dir, f".tmp-{pid}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(health_dir, f"worker-{pid}.json"))


def _heartbeat_once(tag: str, pid: int, stats_fn, health_dir: str) -> None:
    """单次心跳：快照组装 + 原子落盘。失败向上抛（由循环侧计频告警）。
    独立成函数供测试直接驱动。"""
    snap = {"wtag": tag, "pid": pid, "ts": time.time()}
    snap.update(stats_fn())
    write_heartbeat(health_dir, pid, snap)


def heartbeat_loop(tag: str, pid: int, stats_fn, health_dir: str = None,
                   interval: float = 5.0, stop: threading.Event = None):
    """心跳线程主体：周期性把 stats_fn() 快照落盘；阻塞调用方线程。

    持续失败只告警不打死：第 1 次与每第 12 次（≈每分钟一条，5s 周期）
    打 warning。历史实现 except-pass 让磁盘满/健康目录被删完全隐身，
    直到 worker 被 supervisor 按 mtime 误杀才暴露——且误杀日志完全对
    不上根因。"""
    health_dir = health_dir or HEALTH_DIR
    os.makedirs(health_dir, exist_ok=True)
    fails = 0
    while True:
        if stop is not None and stop.is_set():
            return
        try:
            _heartbeat_once(tag, pid, stats_fn, health_dir)
            fails = 0
        except Exception as exc:  # noqa: BLE001
            fails += 1
            if fails == 1 or fails % 12 == 0:
                logging.getLogger("kokoro.heartbeat").warning(
                    "[%s] heartbeat write failed (%d consecutive): %s",
                    tag, fails, exc)
        time.sleep(interval)


def start_heartbeat(tag: str, pid: int, stats_fn, health_dir: str = None,
                    interval: float = 5.0) -> threading.Thread:
    t = threading.Thread(target=heartbeat_loop,
                         args=(tag, pid, stats_fn, health_dir, interval),
                         daemon=True)
    t.start()
    return t

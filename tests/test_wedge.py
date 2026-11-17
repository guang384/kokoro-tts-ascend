"""supervisor 端到端假死检测（WEDGE_RESTART_S）与死进程心跳清理。

历史上 mp.Queue 的 POSIX semaphore 被 SIGKILL 持有者泄漏后，所有存取方静默
卡死，worker 心跳与 BusyWatch 都还活着——只能靠“请求进得去、结果出不来”这一
服务级事实判死重启（redis-server 死亡等假死同理）。本文件锁定该判据的触发与
全部不触发条件。"""
import json
import os

import pytest

import supervisor

ROLES = ["gateway", "g2p", "tts", "asr"]


def _write(health_dir, pid, role, ready=True, **stats):
    path = os.path.join(health_dir, f"worker-{pid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"role": role, "ready": ready, "pid": pid,
                   "wtag": f"{role}-x", "stats": stats}, f)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """隔离心跳目录 + 缩小判死窗口（10s），并复位楔子状态。"""
    monkeypatch.setattr(supervisor, "HEALTH_DIR", str(tmp_path))
    monkeypatch.setattr(supervisor, "WEDGE_RESTART_S", 10.0)
    monkeypatch.setattr(supervisor, "HEARTBEAT_STALE_S", 3600)
    monkeypatch.setattr(supervisor, "_wedge_state",
                        {"last_res": None, "last_res_ts": 0.0,
                         "last_tmo": None, "last_tmo_ts": 0.0,
                         "last_5xx": None, "last_5xx_ts": 0.0})
    fired = []
    monkeypatch.setattr(supervisor.os, "_exit", lambda code: fired.append(code))
    return tmp_path, fired


def _gw(health_dir, requests=None, results=None, timeouts=None,
        server_5xx=None):
    """更新网关心跳（未给出的字段沿用旧值需显式传入）。"""
    _write(health_dir, 100, "gateway", requests=requests, results=results,
           timeouts=timeouts, server_5xx=server_5xx)
    for i, role in enumerate(ROLES[1:], start=1):
        _write(health_dir, 100 + i, role)


def test_wedge_fires_after_zero_success_window(env):
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0)
    supervisor.check_wedge(1000.0)              # 基线
    supervisor.check_wedge(1005.0)              # 尚无新超时
    _gw(hd, requests=2, results=0, timeouts=1)  # 受理在涨、零成功、超时在涨
    supervisor.check_wedge(1010.0)
    assert fired == [1]


def test_no_wedge_while_results_advance(env):
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0)
    supervisor.check_wedge(1000.0)
    for k, t in enumerate((1003.0, 1006.0, 1009.0, 1012.0), start=1):
        _gw(hd, requests=k, results=k, timeouts=0)
        supervisor.check_wedge(t)
    assert fired == []


def test_no_wedge_during_startup(env):
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0)
    _write(hd, 103, "tts", ready=False)         # tts 还在热身
    supervisor.check_wedge(1000.0)
    _gw(hd, requests=5, results=0, timeouts=4)
    _write(hd, 103, "tts", ready=False)
    supervisor.check_wedge(1010.0)
    supervisor.check_wedge(1030.0)
    assert fired == []


def test_no_wedge_when_timeouts_stopped(env):
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0)
    supervisor.check_wedge(1000.0)
    _gw(hd, requests=2, results=0, timeouts=1)  # 一次超时后服务静默
    supervisor.check_wedge(1001.0)
    supervisor.check_wedge(1010.0)              # 超时不再新鲜（>WEDGE/2）
    supervisor.check_wedge(1020.0)
    assert fired == []


def test_wedge_fires_on_5xx_growth(env):
    """Redis 不可达等快速 5xx 不产生超时：5xx 增长 + 零结果同样判死。"""
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0, server_5xx=0)
    supervisor.check_wedge(1000.0)
    _gw(hd, requests=1, results=0, timeouts=0, server_5xx=1)
    supervisor.check_wedge(1010.0)
    assert fired == [1]


def test_wedge_ignores_stale_5xx(env):
    """很久以前的 5xx（>WEDGE/2）不再作为新鲜失败信号。"""
    hd, fired = env
    _gw(hd, requests=1, results=0, timeouts=0, server_5xx=1)
    supervisor.check_wedge(1000.0)
    _gw(hd, requests=1, results=0, timeouts=0, server_5xx=1)
    supervisor.check_wedge(1010.0)
    supervisor.check_wedge(1020.0)
    assert fired == []


class _DeadProc:
    pid = 999
    exitcode = -9

    def is_alive(self):
        return False


def test_dead_worker_heartbeat_removed(tmp_path, monkeypatch):
    """死进程心跳立即删除：否则 mtime 过期前 /health 会把幽灵工人算进角色计数。"""
    monkeypatch.setattr(supervisor, "HEALTH_DIR", str(tmp_path))
    monkeypatch.setattr(supervisor, "procs", {("tts", 0): _DeadProc()})
    monkeypatch.setattr(supervisor, "_next_ok", {})
    monkeypatch.setattr(supervisor, "_fail_count", {})
    monkeypatch.setattr(supervisor, "_role_start_busy", lambda role: False)
    monkeypatch.setattr(supervisor, "spawn_one", lambda role, i: None)
    ghost = os.path.join(tmp_path, "worker-999.json")
    with open(ghost, "w", encoding="utf-8") as f:
        f.write("{}")
    supervisor.monitor_once()
    assert not os.path.exists(ghost)


def test_no_wedge_after_gateway_respawn_baseline_reset(env):
    """部分网关重生 → 聚合累计和跌破陈旧基线 → 基线必须重置。

    不重置的话 quiet 从崩溃前起算：恢复期一次普通 504 就能在"结果明明在
    流"时误 exit(1)（真实事故推演）。重置只延迟检测——计数随后冻结仍会触发。"""
    hd, fired = env
    _gw(hd, requests=1, results=100, timeouts=0)   # 基线 res=100
    supervisor.check_wedge(1000.0)
    _gw(hd, requests=2, results=30, timeouts=0)    # 网关 A 重生：聚合跌到 30
    supervisor.check_wedge(1002.0)                 # 下降 → 基线重置（ts=1002）
    _gw(hd, requests=3, results=30, timeouts=1)    # 恢复期 504、零新成功
    supervisor.check_wedge(1007.0)                 # quiet=5：不触发
    assert fired == []
    _gw(hd, requests=4, results=30, timeouts=2)    # 此后计数冻结
    supervisor.check_wedge(1012.0)                 # quiet=10：延迟到位后照常触发
    assert fired == [1]



def test_sweeper_wiring_resolves_and_runs(fakeredis_client, monkeypatch, tmp_path):
    """深度审计 F1 回归：supervisor._sweep_inflight 必须真实调用成功——
    曾调用不存在的模块级 sweep_inflight，AttributeError 被吞 → sweeper
    从未跑过（重派/取消清扫静默失效）。"""
    import proc_common
    import redis_transport
    import supervisor

    # 符号可解析性（模块级 sweep_inflight 不存在是设计内——方法才是入口）
    monkeypatch.setattr(redis_transport, "_make_redis_client",
                        lambda: fakeredis_client)
    q = proc_common.TtsClaimQueue(16, index=45, name="sweep-wire")
    monkeypatch.setattr(supervisor, "_tts_queue", q)
    monkeypatch.setattr(supervisor, "_asr_queue", None)
    monkeypatch.setattr(supervisor, "procs", {})   # 无存活 pid → 死亡路径
    monkeypatch.setattr(supervisor, "_redis_client", fakeredis_client)

    # 死者在途条目（认领后不 ack；worker_id 带 pid 供死亡判定）
    q.put({"req_id": "dead", "idx": 0, "ps": "a" * 30, "voice": "v",
           "speed": 1.0, "origin": 0, "need": 1, "text": "t"})
    claimed = q.claim(4, "tts-0-999")
    assert len(claimed) == 1

    supervisor._sweep_inflight()          # 不得抛、不得静默 no-op
    assert q.depth() == 1                 # 死亡精确重派回队列（不等 TTL）
    assert fakeredis_client.zcard(f"{q._key}:infl:tts-0-999") == 0


def test_monitor_respawn_auto_requeues_inflight(monkeypatch):
    """自愈链②的串联锁定：monitor_once 发现死进程 → respawn 前自动
    requeue_worker（此前只测过函数本体，'supervisor 自动检测→重派'的
    串联路径无测试驱动）。"""
    import supervisor

    calls = []

    class FakeQ:
        def requeue_worker(self, wid, max_redispatch=0):
            calls.append((wid, max_redispatch))
            return 3

    class DeadProc:
        pid = 777

        def is_alive(self):
            return False

        exitcode = -9

    monkeypatch.setattr(supervisor, "procs", {("tts", 0): DeadProc()})
    monkeypatch.setattr(supervisor, "_tts_queue", FakeQ())
    monkeypatch.setattr(supervisor, "_asr_queue", None)
    monkeypatch.setattr(supervisor, "_redis_client", None)
    monkeypatch.setattr(supervisor, "_next_ok", {})
    monkeypatch.setattr(supervisor, "_fail_count", {})
    monkeypatch.setattr(supervisor, "_role_start_busy", lambda role: False)
    monkeypatch.setattr(supervisor, "spawn_one", lambda role, i: None)
    monkeypatch.setattr(supervisor, "HEALTH_DIR", "/nonexistent-health")

    supervisor.monitor_once()
    assert calls == [("tts-0-777", supervisor.REDISPATCH_MAX)]
    # 死者在途被自动 requeue（role-i-pid），且死亡主路径传入毒块上限


def test_monitor_step_swallows_exception(monkeypatch):
    """monitor_once 残余异常不能杀 supervisor 主线程（原裸调会退出 main）。"""
    def _boom():
        raise RuntimeError("redis guardian exploded")

    monkeypatch.setattr(supervisor, "monitor_once", _boom)
    supervisor._monitor_step()          # 不抛 = 主循环继续
    assert True


def test_crash_labels_match_gateway_wtag(monkeypatch):
    """/health crashes 的 key 必须与 gateway 心跳 wtag 同源（gw-N），
    否则 gateway_obs._worker_row 永远查不到、/metrics 标签分裂。"""
    monkeypatch.setattr(supervisor, "_crash_total",
                        {("gateway", 0): 2, ("tts", 1): 3})
    monkeypatch.setattr(supervisor, "mailboxes", {})
    snap = supervisor.supervisor_snapshot()
    assert snap["stats"]["crashes"] == {"gw-1": 2, "tts-1": 3}


def test_respawn_failure_keeps_retry_slot(monkeypatch, tmp_path):
    """spawn_one 失败后 (role,i) 不能从 procs 消失——否则永久缺员。"""
    class DeadProc:
        pid = 778
        exitcode = -9

        def is_alive(self):
            return False

    monkeypatch.setattr(supervisor, "procs", {("tts", 0): DeadProc()})
    monkeypatch.setattr(supervisor, "_tts_queue", None)
    monkeypatch.setattr(supervisor, "_asr_queue", None)
    monkeypatch.setattr(supervisor, "_next_ok", {})
    monkeypatch.setattr(supervisor, "_fail_count", {})
    monkeypatch.setattr(supervisor, "_role_start_busy", lambda role: False)
    monkeypatch.setattr(supervisor, "HEALTH_DIR", str(tmp_path))

    def _boom(role, i):
        raise RuntimeError("import failed")

    monkeypatch.setattr(supervisor, "spawn_one", _boom)
    supervisor.monitor_once()
    assert ("tts", 0) in supervisor.procs


def test_queue_bad_increment_warns(monkeypatch):
    """坏 member 增量告警不依赖 METRICS_ENABLED：首轮记基线，之后每次
    增量一条 WARNING（同值不重复）——载荷损坏/外部误写的主动信号。"""
    import supervisor

    class Q:
        n = 0

        def stats(self):
            return {"bad": self.n}

    q = Q()
    warns = []
    monkeypatch.setattr(supervisor, "_tts_queue", q)
    monkeypatch.setattr(supervisor, "_asr_queue", None)
    monkeypatch.setattr(supervisor, "_bad_seen", {"n": None})
    monkeypatch.setattr(supervisor.log, "warning",
                        lambda msg, *a: warns.append(msg % a))
    supervisor._watch_queue_bad()
    assert warns == []                      # 首轮只记基线
    q.n = 2
    supervisor._watch_queue_bad()
    assert warns and "+2" in warns[0] and "total 2" in warns[0]
    supervisor._watch_queue_bad()
    assert len(warns) == 1                  # 无增量不重复告警


def test_check_redis_revives_dead_server(monkeypatch):
    """自愈链④的守护锁定：redis-server 单独死亡（supervisor 存活的窄窗口）
    → 自动重启；队列重置为已知边界（请求 504 兜底）。"""
    import redis_transport
    import supervisor

    class DeadRedis:
        returncode = 1

        def poll(self):
            return 1

    revived = []

    monkeypatch.setattr(supervisor, "_redis_proc", DeadRedis())
    monkeypatch.setattr(supervisor, "_redis_client", object())
    monkeypatch.setattr(redis_transport, "start_redis_server",
                        lambda: revived.append(1) or "new-proc")
    supervisor._check_redis()
    assert revived == [1]                 # 自动重启
    assert supervisor._redis_proc == "new-proc"
    assert supervisor._redis_client is None   # 旧连接废弃（下轮重建）

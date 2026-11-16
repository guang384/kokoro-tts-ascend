"""/health 契约（单一精简形态）：一屏结论卡 + 队列统计 {depth,in,out,
avg_wait_ms} + workers 精选指标；诊断细节一律靠日志。直接驱动 handler
（不起 HTTP）。另含 proc_common 邮箱统计的回归（列表展开/异常容错/
InstrumentedQueue EMA 计数语义）。"""
import asyncio
import json
import os
import time
import unittest.mock as mock

import pytest

import gateway
import gateway_state
import gateway_obs
import proc_common
import redis_transport
import supervisor
from tests.fake_queue import FakeQueue


def _write_heartbeat(health_dir: str, wtag: str, snapshot: dict) -> None:
    path = os.path.join(health_dir, f"worker-{wtag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"wtag": wtag, **snapshot}, f, ensure_ascii=False)


@pytest.fixture()
def gw(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_state, "_health_dir", str(tmp_path))
    monkeypatch.setattr(gateway_state, "_result_q", None)
    return gateway


def test_default_view_shape(gw):
    """_collect() 全量视图：status/ready/version/shm/e2e/queues/workers；
    queues 按请求流经顺序；workers 统一三件套（tts 加 realtime_x）；
    supervisor 不出现在 workers；健康 worker 无噪音。"""
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True,
        "uptime_s": 100.0, "pid": 123, "warm": {"kokoro": "ok"},
        "stats": {"jobs": 3, "forward_ms": 420.0, "sentences": 3,
                  "audio_s": 71.1, "forward_errors": 0},
        "npu_mem": {"reserved_mb": 5120.0}})
    _write_heartbeat(gateway_state._health_dir, "gw-1", {
        "role": "gateway", "device": None, "ready": True, "warm": {"g2p": "off"},
        "stats": {"requests": 120, "results": 118, "latency_ms_sum": 489725.0,
                  "timeouts": 0, "rejections": 1, "pending": 0,
                  "recent_count": 2, "recent_ms_sum": 1500.0,
                  "recent_chars_sum": 90.0,
                  "recent_failures": [{"time": "10:00:01", "status": 504,
                                       "reason": "推理或对齐超时"}]}})
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True,
        "expected": {"tts": 1, "gateway": 1},
        "mailboxes": {"g2p_queue": {"depth": 2, "in": 5, "out": 3, "avg_wait_ms": 110.5},
                      "tts_queue": {"depth": 0, "in": 9, "out": 9, "avg_wait_ms": 75},
                      "gateway_queues[0]": {"depth": 1, "in": 4, "out": 3, "avg_wait_ms": 55.25}}})
    d = gateway_obs._collect()
    assert {"status", "ready", "version", "shm", "e2e",
            "queues", "workers"} <= set(d)
    assert d["shm"] is None or (
        isinstance(d["shm"], dict)
        and {"total_mb", "used_mb", "pct"} <= set(d["shm"]))
    assert d["status"] == "ok" and d["ready"] is True
    # 顶层 e2e：跨网关聚合（本用例单网关）
    assert d["e2e"]["requests"] == 120 and d["e2e"]["successes"] == 118
    assert d["e2e"]["timed_out"] == 0 and d["e2e"]["rejected"] == 1
    assert d["e2e"]["avg_ms_recent"] == 750.0    # 1500ms / 2 次（加权合并）
    assert d["e2e"]["avg_chars_recent"] == 45.0  # 90 / 2
    assert d["e2e"]["recent_failures"][0]["status"] == 504
    assert list(d["queues"]) == ["g2p", "tts", "gateway"]
    assert d["queues"]["g2p"] == {"depth": 2, "in": 5, "out": 3,
                                  "avg_wait_ms": 110.5, "bad": 0}
    # 多实例 gateway 合并：depth/in/out 求和、avg_wait 取最慢实例
    assert d["queues"]["gateway"] == {"depth": 1, "in": 4, "out": 3,
                                      "avg_wait_ms": 55.25, "bad": 0}
    by_tag = {w["wtag"]: w for w in d["workers"]}
    assert set(by_tag) == {"tts-0", "gw-1"}
    # tts 三件套：done=jobs、avg_ms=forward_ms/jobs、realtime_x=audio_s/forward_s
    assert by_tag["tts-0"]["done"] == 3
    assert by_tag["tts-0"]["avg_ms"] == 140.0
    assert by_tag["tts-0"]["realtime_x"] == 169.3
    assert "err" not in by_tag["tts-0"]          # err 仅 >0 出现
    # gateway 三件套：done=results、avg_ms=latency_ms_sum/results、err 聚合
    assert by_tag["gw-1"]["done"] == 118
    assert by_tag["gw-1"]["avg_ms"] == 4150.21
    assert by_tag["gw-1"]["err"] == 1            # timeouts 0 + rejections 1
    assert "pid" not in by_tag["tts-0"] and "npu_mem" not in by_tag["tts-0"]


def test_warm_only_when_abnormal(gw):
    """warm 值只在异常时出现在 workers[]；无任何 ready 工人 → starting
    （不可服务是外层门槛，优先于 degraded）。"""
    _write_heartbeat(gateway_state._health_dir, "tts-1", {
        "role": "tts", "device": "npu:0", "ready": False,
        "uptime_s": 5.0, "warm": {"kokoro": "pending"}})
    _write_heartbeat(gateway_state._health_dir, "tts-2", {
        "role": "tts", "device": "npu:0", "ready": False,
        "warm": {"kokoro": "fail"}})
    d = gateway_obs._collect()
    assert d["status"] == "starting"
    assert d["ready"] is False
    by_tag = {w["wtag"]: w for w in d["workers"]}
    assert by_tag["tts-1"]["warm"] == {"kokoro": "pending"}
    assert by_tag["tts-2"]["warm"] == {"kokoro": "fail"}


def test_warm_fail_degrades_when_serviceable(gw):
    """可服务（有 ready 工人）但存在 warm=fail → degraded。"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {"tts": 2},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 1, "forward_ms": 100.0, "audio_s": 5.0}})
    _write_heartbeat(gateway_state._health_dir, "tts-2", {
        "role": "tts", "device": "npu:0", "ready": False,
        "warm": {"kokoro": "fail"}})
    d = gateway_obs._collect()
    assert d["status"] == "degraded"
    assert d["ready"] is True  # tts-0 可服务
def test_ramp_up_ready_semantics(gw):
    """串行启动扩容期（核心新语义）：期望 6 个 tts，只有 1 个 ready——
    服务整体就算 ready（可接流量），status=ok；热身中的 worker 带待处理
    的 warm 标记可见。"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True,
        "expected": {"gateway": 2, "g2p": 2, "tts": 6, "asr": 3},
        "mailboxes": {}})
    for role, n in (("gateway", 2), ("g2p", 2), ("asr", 3)):
        for i in range(n):
            _write_heartbeat(gateway_state._health_dir, f"{role}-{i}", {
                "role": role, "ready": True, "warm": {}, "stats": {}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 2, "forward_ms": 260.0, "audio_s": 8.0}})
    for i in range(1, 6):  # 其余 5 个热身中
        _write_heartbeat(gateway_state._health_dir, f"tts-{i}", {
            "role": "tts", "device": "npu:0", "ready": False,
            "warm": {"kokoro": "pending"}})
    d = gateway_obs._collect()
    assert d["ready"] is True        # 能力就绪：每角色 ≥1 → 整个服务 ready
    assert d["status"] == "ok"       # 扩容期无异常 → ok（可接流量）
    by_tag = {w["wtag"]: w for w in d["workers"]}
    assert by_tag["tts-0"]["ready"] is True
    for i in range(1, 6):
        assert by_tag[f"tts-{i}"]["ready"] is False
        assert by_tag[f"tts-{i}"]["warm"] == {"kokoro": "pending"}


def test_missing_roles_reflected_in_status(gw):
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True,
        "expected": {"tts": 2, "asr": 1},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {}})
    d = gateway_obs._collect()
    assert d["status"] == "starting"
    assert d["queues"] == {}


def test_backpressure_logged_not_returned(gw):
    """队列积压：只写日志（不进响应体）；多实例 gateway 深度求和后合并。"""
    logged = []
    with mock.patch.object(gateway.log, "warning",
                           side_effect=lambda msg, *a: logged.append(msg % a if a else msg)):
        _write_heartbeat(gateway_state._health_dir, "supervisor", {
            "role": "supervisor", "ready": True,
            "expected": {"tts": 1},
            "mailboxes": {"tts_queue": {"depth": gateway_state._WORKER_MAILBOX_SIZE,
                                        "in": 1200, "out": 176, "avg_wait_ms": 6100},
                          "gateway_queues[0]": {"depth": int(gateway_state._WORKER_MAILBOX_SIZE * 0.5),
                                                "in": 10, "out": 10, "avg_wait_ms": 15},
                          "gateway_queues[1]": {"depth": int(gateway_state._WORKER_MAILBOX_SIZE * 0.4),
                                                "in": 10, "out": 10, "avg_wait_ms": 12}}})
        _write_heartbeat(gateway_state._health_dir, "tts-0", {
            "role": "tts", "device": "npu:0", "ready": True, "warm": {}})
        d = gateway_obs._collect()
    assert d["queues"]["tts"]["depth"] == gateway_state._WORKER_MAILBOX_SIZE
    assert d["queues"]["gateway"]["depth"] == int(gateway_state._WORKER_MAILBOX_SIZE * 0.5) \
        + int(gateway_state._WORKER_MAILBOX_SIZE * 0.4)
    assert d["queues"]["gateway"]["avg_wait_ms"] == 15  # 多实例取最慢
    assert any("queue backlog: tts_queue" in line for line in logged)
    assert not any("gateway_queues[1]" in line for line in logged)  # 40% 不判


def test_stale_heartbeat_excluded(gw):
    _write_heartbeat(gateway_state._health_dir, "tts-dead", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {}})
    old = time.time() - gateway_state._HEALTH_STALE_AFTER - 10
    os.utime(os.path.join(gateway_state._health_dir, "worker-tts-dead.json"), (old, old))
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {}})
    d = gateway_obs._collect()
    assert [w["wtag"] for w in d["workers"]] == ["tts-0"]


def test_asr_metrics_in_view(gw):
    """asr 三件套 + err 聚合（drops+errors）。batcher 角色已删除（认领式）。"""
    _write_heartbeat(gateway_state._health_dir, "asr-0", {
        "role": "asr", "device": "npu:0", "ready": True,
        "warm": {"funasr": "ok"},
        "stats": {"jobs": 57, "asr_ms": 46467.7, "job_drops": 1,
                  "job_errors": 2}})
    d = gateway_obs._collect()
    by_tag = {w["wtag"]: w for w in d["workers"]}
    assert by_tag["asr-0"]["done"] == 57
    assert by_tag["asr-0"]["avg_ms"] == 815.22
    assert by_tag["asr-0"]["err"] == 3            # job_drops 1 + job_errors 2


def test_role_metrics_div_zero_and_err(gw):
    """除零防护：无完成量时 avg_ms 不出现；err 按角色聚合口径；g2p/asr 覆盖。"""
    _write_heartbeat(gateway_state._health_dir, "tts-9", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {"kokoro": "ok"},
        "stats": {"jobs": 0, "forward_ms": 0.0, "audio_s": 0.0, "forward_errors": 0}})
    _write_heartbeat(gateway_state._health_dir, "g2p-0", {
        "role": "g2p", "ready": True,
        "stats": {"requests": 0, "g2p_ms": 0.0, "errors": 0}})
    _write_heartbeat(gateway_state._health_dir, "g2p-1", {
        "role": "g2p", "ready": True,
        "stats": {"requests": 40, "g2p_ms": 1522.0, "errors": 1}})
    _write_heartbeat(gateway_state._health_dir, "asr-2", {
        "role": "asr", "device": "npu:0", "ready": True, "warm": {"funasr": "ok"},
        "stats": {"jobs": 15, "asr_ms": 13206.0, "job_drops": 1,
                  "job_errors": 0}})
    d = gateway_obs._collect()
    by_tag = {w["wtag"]: w for w in d["workers"]}
    assert "avg_ms" not in by_tag["tts-9"]        # 除零：无前向不出现 avg_ms
    assert "realtime_x" not in by_tag["tts-9"]
    assert by_tag["g2p-0"]["done"] == 0 and "avg_ms" not in by_tag["g2p-0"]
    assert "err" not in by_tag["g2p-0"]           # err=0 不出现
    assert by_tag["g2p-1"]["avg_ms"] == 38.05
    assert by_tag["g2p-1"]["err"] == 1
    assert by_tag["asr-2"]["done"] == 15
    assert by_tag["asr-2"]["avg_ms"] == 880.4
    assert by_tag["asr-2"]["err"] == 1            # job_drops 1 + job_errors 0


# ── proc_common：邮箱统计（supervisor 心跳修复 + InstrumentedQueue）──

def test_mailbox_depths_expands_lists():
    """gateway_queues 是列表：必须展开为 name[i]（此前 AttributeError 让
    supervisor 心跳从未落盘）。普通队列只有 depth。"""
    class _Q:
        def __init__(self, n):
            self._n = n

        def qsize(self):
            return self._n

    d = proc_common.mailbox_depths({"g2p_queue": _Q(3),
                                    "gateway_queues": [_Q(1), _Q(0)]})
    assert d == {"g2p_queue": {"depth": 3},
                 "gateway_queues[0]": {"depth": 1},
                 "gateway_queues[1]": {"depth": 0}}


def test_mailbox_depths_swallows_probe_errors():
    """qsize 抛任何异常 → depth=None（心跳线程存活优先于探针完整性）。"""
    class _Broken:
        def qsize(self):
            raise ValueError("Queue is closed")

    d = proc_common.mailbox_depths({"dead": _Broken()})
    assert d == {"dead": {"depth": None}}


def test_instrumented_queue_counts(fakeredis_client, monkeypatch):
    """InstrumentedQueue（Redis 后端）：in/out 服务端 INCR 计数、depth=LLEN
    真实等待处理数、avg_wait_ms（EMA，首样本=实际等待）；_q_ts 用后即除
    （消息体不带时间戳污染下游）。"""
    monkeypatch.setattr(redis_transport, "_make_redis_client", lambda: fakeredis_client)
    q = proc_common.InstrumentedQueue(0, index=42)
    q.put_nowait({"req_id": "held"})
    time.sleep(0.1)                            # 压着不取：等待 ~100ms
    assert q.get(timeout=2)["req_id"] == "held"
    st = q.stats()
    assert st["in"] == 1 and st["out"] == 1
    assert st["avg_wait_ms"] >= 80, st         # EMA 首样本 = 实际等待
    q.put_nowait({"req_id": "b"})
    got = q.get(timeout=2)
    assert got["req_id"] == "b"
    assert "_q_ts" not in got
    st = q.stats()
    assert st["in"] == 2 and st["out"] == 2
    assert st["avg_wait_ms"] >= 0              # 短等待拉低 EMA
    assert set(st) == {"depth", "in", "out", "avg_wait_ms"}
    q.close()


def test_depth_is_true_queue_length(fakeredis_client, monkeypatch):
    """depth 语义 = LLEN = 队列里真实等待处理的消息数：put 后立即可见、
    消费后归零、空闲必归零（无 enq-deq 记账漂移——113/57 幻影的根治）。"""
    monkeypatch.setattr(redis_transport, "_make_redis_client", lambda: fakeredis_client)
    q = proc_common.InstrumentedQueue(0, index=44)
    try:
        q.put_nowait({"i": 0})
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and q.stats()["depth"] == 0:
            time.sleep(0.02)
        assert q.stats()["depth"] == 1          # 等待处理 = 1（put 即入列）
        assert q.stats()["in"] == 1 and q.stats()["out"] == 0
        assert q.get(timeout=2)["i"] == 0
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and q.stats()["depth"] != 0:
            time.sleep(0.02)
        assert q.stats()["depth"] == 0          # 消费完归零（无漂移）
        assert q.stats()["in"] == 1 and q.stats()["out"] == 1
    finally:
        q.close()




def test_degraded_serviceable_returns_200(gw):
    """degraded = 可服务但部分异常（如一个 tts 预热失败、另一个在服务）：
    /health 必须 200——liveness 绝不重启仍能接单的容器。
    （CPU 降级功能已移除：NPU 持续失败走 fail-fast 503→容器重启）"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {"tts": 2},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 1, "forward_ms": 100.0, "audio_s": 5.0}})
    _write_heartbeat(gateway_state._health_dir, "tts-1", {
        "role": "tts", "device": None, "ready": False,
        "warm": {"kokoro": "fail"}, "stats": {}})
    d = gateway_obs._collect()
    assert d["status"] == "degraded" and d["ready"] is True
    r = asyncio.run(gateway.health())
    assert r.status_code == 200
    assert json.loads(r.body)["status"] == "degraded"

    # 反向：tts 全灭（零 ready）→ 503 → 编排层重启容器（fail-fast 兜底）
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": None, "ready": False,
        "warm": {"kokoro": "fail"}, "stats": {}})
    r = asyncio.run(gateway.health())
    assert r.status_code == 503




def test_worker_crash_count_exposed(gw):
    """supervisor 心跳的 crashes 统计透传到 workers[] 对应行（>0 才出现）。"""
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True,
        "expected": {"tts": 1},
        "stats": {"crashes": {"tts-0": 3}},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True,
        "warm": {"kokoro": "ok"},
        "stats": {"jobs": 1, "forward_ms": 100.0, "audio_s": 5.0}})
    d = gateway_obs._collect()
    entry = [w for w in d["workers"] if w["wtag"] == "tts-0"][0]
    assert entry["crashes"] == 3




def test_shm_usage_exposed(gw, monkeypatch):
    """/dev/shm 使用情况进 /health（POSIX 上有值；水位 = used/total）。"""
    class _St:
        f_frsize = 4096
        f_blocks = 256      # 1GB
        f_bavail = 64       # 剩余 25% → used 75%

    real_statvfs = getattr(supervisor.os, "statvfs", None)
    if real_statvfs is None:   # Windows 开发机：给 os 挂 statvfs（仅测试期）
        monkeypatch.setattr(gateway.os, "statvfs", lambda p: _St(), raising=False)
    else:
        monkeypatch.setattr(gateway.os, "statvfs", lambda p: _St())
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {},
        "mailboxes": {}})
    d = gateway_obs._collect()
    total = round(4096 * 256 / 2**20, 1)          # 1.0 MiB
    used = round(4096 * (256 - 64) / 2**20, 1)    # 0.75 MiB（round 后 0.8）
    assert d["shm"] == {"total_mb": total, "used_mb": used,
                        "total_bytes": 4096 * 256,
                        "pct": 75.0}   # pct 用原始字节算（192/256），不用舍入后的 MB




def test_orphan_age_floor_covers_both_waits():
    """回收线不低于 2×WAIT+60：/tts 音频等待 + 时间戳等待串行 = 块的最长
    合法持有期；显式配小会被公式抬高（防误扫活块）。"""
    assert gateway_state.SHM_ORPHAN_MAX_AGE_S >= 2 * gateway_state._entry_wait_s + 60


def test_orphan_shm_swept_by_age(gw, monkeypatch, tmp_path):
    """清道夫：只回收 kokora_ 前缀且超龄的块；新鲜块与非本服务文件保留。"""
    monkeypatch.setattr(gateway_state, "_SHM_SWEEP_DIR", str(tmp_path))
    monkeypatch.setattr(gateway_state, "SHM_ORPHAN_MAX_AGE_S", 300.0)

    old_f = tmp_path / "kokora_old"          # 超龄孤儿 → 回收
    new_f = tmp_path / "kokora_new"          # 新鲜块（合法生命周期内）→ 保留
    other = tmp_path / "psm_notours"         # 非 kokora 前缀 → 不碰
    for f in (old_f, new_f, other):
        f.write_bytes(b"x")
    import os as _os
    t = __import__("time").time() - 400
    _os.utime(old_f, (t, t))

    n = gateway_obs._sweep_orphan_shm()
    assert n == 1
    assert not old_f.exists()
    assert new_f.exists() and other.exists()


def test_orphan_sweep_tolerates_missing_dir(gw, monkeypatch):
    monkeypatch.setattr(gateway_state, "_SHM_SWEEP_DIR",
                        gateway_state._health_dir + "/nope")   # 不存在的目录
    assert gateway_obs._sweep_orphan_shm() == 0   # 目录不存在 → 0，不抛


def test_ship_batch_names_blocks_kokora(tmp_path):
    """音频块统一 kokora_ 前缀（清道夫按名回收的前提）。"""
    import numpy as np
    from multiprocessing import shared_memory

    import compute as compute_mod
    from tests.test_compute import _PosStub

    gq = FakeQueue()
    log = proc_common.setup_logging("test-naming")
    chunk = __import__("protocol").chunk(
        req_id="r1", idx=0, ps="ə", voice="zf_001", speed=1.0,
        origin=0, need=1, text="句")
    from tests.fake_queue import FakeQueue as _FQ
    compute_mod.ship_batch(_PosStub(), {"batch_id": "b-1", "chunks": [chunk]},
                           [gq], _FQ(), log, np, shared_memory)
    part = gq.get_nowait()
    assert part["block"]["shm_name"].startswith("kokora_")
    shm = shared_memory.SharedMemory(name=part["block"]["shm_name"])
    shm.close()
    try:
        shm.unlink()
    except Exception:  # noqa: BLE001
        pass




# ── e2e 指标：请求/成功总数、滑动平均（近 50）、最近 3 次失败 ──

def test_e2e_metrics_merge_across_gateways(gw):
    """顶层 e2e 全局聚合：双网关的受理/成功求和，滑动平均按窗口计数加权，
    失败跨网关合并按时间取最新 3 条；workers[] 恢复纯实例事实（不含 e2e）。"""
    gateway_state._recent_latencies.clear()
    gateway_state._recent_failures.clear()
    gateway._record_failure(504, "超时A")
    gateway._record_failure(502, "前向失败B")

    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {"gateway": 2},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "gw-1", {
        "role": "gateway", "device": None, "ready": True, "warm": {},
        "stats": {"requests": 100, "results": 90, "timeouts": 2, "rejections": 5,
                  "latency_ms_sum": 1000.0,
                  "recent_count": 2, "recent_ms_sum": 800.0, "recent_chars_sum": 40.0,
                  "recent_failures": [
                      {"time": "10:00:01", "status": 504, "reason": "超时A"},
                      {"time": "10:00:02", "status": 502, "reason": "前向失败B"}]}})
    _write_heartbeat(gateway_state._health_dir, "gw-2", {
        "role": "gateway", "device": None, "ready": True, "warm": {},
        "stats": {"requests": 50, "results": 45, "timeouts": 1, "rejections": 3,
                  "latency_ms_sum": 500.0,
                  "recent_count": 1, "recent_ms_sum": 1000.0, "recent_chars_sum": 60.0,
                  "recent_failures": [{"time": "10:00:03", "status": 500,
                                       "reason": "合成失败C"}]}})
    d = gateway_obs._collect()
    e = d["e2e"]
    assert e["requests"] == 150 and e["successes"] == 135
    assert e["timed_out"] == 3 and e["rejected"] == 8
    assert e["avg_ms_recent"] == 600.0     # (800+1000)/(2+1) 加权
    assert e["avg_chars_recent"] == 33.3   # (40+60)/3
    assert [f["status"] for f in e["recent_failures"]] == [500, 502, 504]  # 最新在前
    entry = [w for w in d["workers"] if w["wtag"] == "gw-1"][0]
    assert entry["done"] == 90 and entry["requests"] == 100
    assert "avg_ms_recent" not in entry and "recent_failures" not in entry


def test_recent_latencies_window_bounded():
    """滑动窗口上限 50：追加 60 条只留最近 50 条。"""
    gateway_state._recent_latencies.clear()
    for i in range(60):
        gateway_state._recent_latencies.append((i * 0.001, i))
    assert len(gateway_state._recent_latencies) == 50
    assert gateway_state._recent_latencies[0] == (10 * 0.001, 10)   # 最旧 9 条已滚出


def test_record_failure_truncates_reason():
    gateway_state._recent_failures.clear()
    gateway._record_failure(500, "很长的原因" * 100)
    f = gateway_state._recent_failures[-1]
    assert len(f["reason"]) <= 80
    gateway_state._recent_failures.clear()




def test_health_route_status_code_gate(gw):
    """K8s readinessProbe 门闸：ready=false → HTTP 503（JSON 体完整）；
    ready=true → 200。"""
    # supervisor 期望 gateway+tts，tts 零就绪 → ready=False → 503
    _write_heartbeat(gateway_state._health_dir, "supervisor", {
        "role": "supervisor", "ready": True, "expected": {"gateway": 1, "tts": 1},
        "mailboxes": {}})
    _write_heartbeat(gateway_state._health_dir, "gw-1", {
        "role": "gateway", "device": None, "ready": True, "warm": {},
        "stats": {"requests": 0, "results": 0, "latency_ms_sum": 0.0}})
    # /health = 三探针共用：未就绪 503（readiness 摘流量 + startupProbe 宽限保护）
    d = asyncio.run(gateway.health())
    assert d.status_code == 503
    body = json.loads(d.body)
    assert body["ready"] is False and body["status"] == "starting"
    assert set(body) == {"status", "ready", "version", "shm", "recent_failures"}

    # tts-0 就绪 → 每角色 ≥1 → 200（K8s 开始接流量）
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {"kokoro": "ok"},
        "stats": {"jobs": 1, "forward_ms": 100.0, "audio_s": 5.0}})
    d = asyncio.run(gateway.health())
    assert d.status_code == 200
    assert json.loads(d.body)["ready"] is True


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_health.py")


def test_ipc_dropped_surfaces_in_worker_err(gw):
    """ipc_dropped 正向链路：worker stats 携带该键时计入 workers[].err
    （/metrics kokoro_worker_errors_total 同源）——受控丢弃直接回答"504
    是谁丢的"。缺键路径已由其余 fixture 隐式覆盖（.get or 0 不炸）。"""
    _write_heartbeat(gateway_state._health_dir, "tts-0", {
        "role": "tts", "device": "npu:0", "ready": True, "warm": {"kokoro": "ok"},
        "stats": {"jobs": 5, "forward_ms": 100.0,
                  "forward_errors": 0, "ipc_dropped": 2}})
    _write_heartbeat(gateway_state._health_dir, "g2p-0", {
        "role": "g2p", "device": None, "ready": True, "warm": {"g2p": "ok"},
        "stats": {"requests": 3, "g2p_ms": 10.0, "errors": 0, "ipc_dropped": 0}})
    d = gateway_obs._collect()
    rows = {w["wtag"]: w for w in d["workers"]}
    assert rows["tts-0"]["err"] == 2          # ipc_dropped 计入 err
    assert "err" not in rows["g2p-0"]         # 0 不显示（既有口径）


def test_recent_latency_quantiles():
    """滑动窗口分位数：p50/p90/p99 线性插值正确、进心跳 stats、空窗不上报。"""
    gateway_state._recent_latencies.clear()
    for s_ in (0.5, 1.0, 2.0, 4.0, 8.0):
        gateway_state._recent_latencies.append((s_, 10))
    snap = gateway_obs._worker_snapshot()
    assert snap["stats"]["p50_s"] == 2.0
    assert snap["stats"]["p90_s"] == round(4.0 * 0.4 + 8.0 * 0.6, 3)   # 6.4（pos=3.6：40% 处于 4.0）
    assert snap["stats"]["p99_s"] == round(4.0 * 0.04 + 8.0 * 0.96, 3)  # 7.84
    gateway_state._recent_latencies.clear()
    assert "p50_s" not in gateway_obs._worker_snapshot()["stats"]

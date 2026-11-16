"""supervisor 布线一致性：任意池尺寸组合下，spawn_one 的全部邮箱引用必须存在、
参数数量必须与目标函数签名匹配。

历史事故回归：asr 工人曾按自己的序号取 per-aligner 结果队列，
ASR_WORKERS>ALIGNER_WORKERS 时 KeyError 崩掉编排——该路由表随 aligner
进程一并删除（chunk 自带 origin 直投网关）；本文件用 stub 进程把全部
组合跑一遍，杜绝同类布线断裂。"""
import inspect

import pytest

import supervisor


class _StubProcess:
    """记录 target/args 不真启动（布线测试不关心子进程行为）。"""
    _n = 0

    def __init__(self, target=None, args=(), name=None):
        self.target = target
        self.args = args
        self.name = name
        _StubProcess._n += 1
        self.pid = 4000 + _StubProcess._n

    def start(self):
        pass

    def is_alive(self):
        return False


def _positional_params(func) -> int:
    sig = inspect.signature(func)
    return len([p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)])


class _StubCtx:
    """supervisor 现仅用 ctx.Process（队列对象由 proc_common 直接构造）。"""
    Process = _StubProcess


@pytest.fixture()
def stubbed(monkeypatch):
    monkeypatch.setattr(supervisor, "ctx", _StubCtx)
    monkeypatch.setattr(supervisor, "mailboxes", {})
    monkeypatch.setattr(supervisor, "procs", {})
    monkeypatch.setattr(supervisor, "_spawn_ts", {})
    monkeypatch.setattr(supervisor, "starting", {})
    monkeypatch.setattr(supervisor, "SPAWN_READY_TIMEOUT_S", 0.0)  # 布线测试不串行
    return supervisor


COMBOS = [
    # (GATEWAY, G2P, TTS, ASR)
    {"GATEWAY_WORKERS_N": 2, "G2P_WORKERS_N": 2, "TTS_N": 6, "ASR_N": 3},
    {"GATEWAY_WORKERS_N": 1, "G2P_WORKERS_N": 1, "TTS_N": 1, "ASR_N": 0},
    {"GATEWAY_WORKERS_N": 1, "G2P_WORKERS_N": 2, "TTS_N": 1, "ASR_N": 5},
    {"GATEWAY_WORKERS_N": 4, "G2P_WORKERS_N": 4, "TTS_N": 8, "ASR_N": 1},
    {"GATEWAY_WORKERS_N": 1, "G2P_WORKERS_N": 1, "TTS_N": 2, "ASR_N": 3},
]


@pytest.mark.parametrize("combo", COMBOS,
                         ids=[f"g{c['GATEWAY_WORKERS_N']}p{c['G2P_WORKERS_N']}"
                              f"t{c['TTS_N']}s{c['ASR_N']}"
                              for c in COMBOS])

def test_wiring_all_pool_combos(stubbed, monkeypatch, combo):
    """全部邮箱引用存在 + asr 领全网关列表 + gateway origin 下标合法 + 签名匹配。"""
    for name, val in combo.items():
        assert hasattr(stubbed, name), name
        monkeypatch.setattr(stubbed, name, val)
    stubbed.create_mailboxes()
    stubbed.spawn_all()

    counts = {}
    for (role, _i) in stubbed.procs:
        counts[role] = counts.get(role, 0) + 1

    assert counts["gateway"] == combo["GATEWAY_WORKERS_N"]
    assert counts["g2p"] == combo["G2P_WORKERS_N"]
    assert counts["tts"] == combo["TTS_N"]
    assert counts.get("asr", 0) == combo["ASR_N"]

    for (role, i), p in stubbed.procs.items():
        # 参数数量不超目标函数的位置参数（生产事故即 KeyError 级布线断裂）
        assert len(p.args) <= _positional_params(p.target),             f"{role}-{i}: {p.target.__name__} 期望≤{_positional_params(p.target)} 个位置参数，实传 {len(p.args)}"
        if role == "asr":
            # asr 领全网关列表（chunk 自带 origin → chunk_ts 直投网关）
            assert len(p.args) == 3
            assert p.args[2] is stubbed.mailboxes["gateway_queues"]
        if role == "gateway":
            assert i < len(stubbed.mailboxes["gateway_queues"])
            assert p.args[3] is stubbed.mailboxes["gateway_queues"][i]



def test_wiring_dedup_rerun_fresh_state(stubbed):
    """重复布线不残留旧 procs（重跑 create_mailboxes+spawn_all 幂等可用）。"""
    stubbed.create_mailboxes()
    stubbed.spawn_all()
    n_first = len(stubbed.procs)
    assert n_first > 0
    stubbed.procs.clear()
    stubbed.spawn_all()
    assert len(stubbed.procs) == n_first




def test_spawn_order_capability_first(stubbed, monkeypatch):
    """启动顺序回归：能力优先——tts-0→asr-0 就位后 /v1、/tts 均可服务，
    再 tts/asr 交替串行扩容（单路 init 不抢卡）。"""
    # 钉住池大小：断言序列按拓扑写死；生产容器 docker exec 会继承
    # TTS_WORKERS/ASR_WORKERS 等业务 env，不钉住则测试随部署漂移而误报
    for name, val in (("GATEWAY_WORKERS_N", 2), ("G2P_WORKERS_N", 2),
                      ("TTS_N", 6), ("ASR_N", 3)):
        monkeypatch.setattr(stubbed, name, val)
    stubbed.SPAWN_READY_TIMEOUT_S = 0.0
    order = []

    def recording_spawn(role, i):
        order.append(f"{role}-{i}")

    stubbed.create_mailboxes()
    monkeypatch.setattr(stubbed, "spawn_one", recording_spawn)
    stubbed.spawn_all()
    assert order == [
        "g2p-0", "g2p-1",
        "gateway-0", "gateway-1",
        "tts-0", "asr-0",                        # 能力相位：NPU 串行各 1
        "tts-1", "asr-1", "tts-2", "asr-2",     # 交替扩容（asr 池先耗尽）
        "tts-3", "tts-4", "tts-5",              # tts 余量
    ]


def test_queue_index_mirror_matches_mailboxes(stubbed):
    """QUEUE_INDEX 镜像锁定：网关 fast_reject 准入闸按固定序号直读 redis
    键 kokoro:q:{idx}:{name}（proc_common.QUEUE_INDEX）——序号/名字与
    create_mailboxes 实际分配漂移会静默探错队列（如探到 gateway 邮箱）。"""
    import proc_common

    stubbed.create_mailboxes()
    for name, idx in proc_common.QUEUE_INDEX.items():
        # redis 名（QUEUE_INDEX 键）≠ mailbox 字典键（*_queue，除 g2p 本就带后缀）
        q = stubbed.mailboxes[name if name.endswith("_queue") else name + "_queue"]
        assert (q._index, q._name) == (idx, name)
        assert q._key == f"kokoro:q:{idx}:{name}"


if __name__ == "__main__":
    raise SystemExit("请用 pytest 运行：python -m pytest tests/test_supervisor_wiring.py")

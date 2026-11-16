"""真实前向回归（可选，需模型权重）：覆盖 batched_model 的合成路径。

离线套件全 mock，batched_model.forward_batch 无任何自动化覆盖——历史上
两次生产事故都出在这里（fade 未定义、TF32/形状）。本测试用 CPU 真实加载
KModel 跑 smoke + 混合批，代价 ~60s 与几百 MB 内存。

运行（任一台有权重的机器/容器）：
    RUN_REAL_FORWARD=1 HF_HOME=/opt/hf-cache python -m pytest tests/test_real_forward.py -v
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_REAL_FORWARD", "0") != "1",
    reason="真实前向需模型权重与 torch；设 RUN_REAL_FORWARD=1 开启")


def test_real_forward_smoke_and_mixed_batch():
    os.environ["DEVICE"] = "cpu"      # 强制 CPU：不占 NPU，可与在线服务共存
    import numpy as np

    import pipeline

    eng = pipeline.get_compute()
    assert eng.synth_batch_smoke() == 1        # 生产 init 同款路径（曾在此炸 fade）

    audios = eng.synth_batch([
        ("n i3", "zf_001", 0.5),               # 极短 + 最慢语速
        ("sh ih4 j ie4 h ui4", "zf_001", 2.0),
        ("zh ong1 g uo2 r en2 g ong1", "zf_001", 1.0),
    ])
    assert len(audios) == 3
    for a in audios:
        assert a.ndim == 1 and len(a) > 0
        assert a.dtype == np.int16             # 新契约：卡上量化直出
        assert int(np.abs(a).max()) <= 32767
        # 5ms 淡出必须生效：末样本 × ramp[-1]=0.0 → 精确 0
        assert int(a[-1]) == 0, f"fade tail: {a[-3:]}"


def test_fade_ramp_direction():
    """淡出斜坡方向锁定：首样本 1.0 → 末样本 0.0（真淡出）。

    曾误用 0→1（淡入）：句尾不衰减反而把窗口首样本强压到 0，切帧爆音
    没消掉还引入新阶跃毛刺——首提交起的原生笔误，方向必须钉死。"""
    import numpy as np
    import torch

    import batched_model
    r = batched_model._fade_ramp(torch.device("cpu"), int(0.005 * 24000))
    r = r.numpy()
    assert r[0] == 1.0 and r[-1] == 0.0
    assert (np.diff(r) < 0).all()              # 单调递减

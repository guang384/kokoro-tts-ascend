#!/usr/bin/env python3
"""对照实验：FunASR 单条 vs 批推路径的 text 回显格式（20261116 WARNING 取证）。

生产 WARNING "对齐单元数 1 != 时间戳数 N" 的根因取证：同样的 (音频, 文本)，
分别走 timestamp()（单条）与 timestamp_batch()（批推），打印 FunASR 原始
text 回显的 repr。真机原生 arm64 秒级；容器内 CPU 也可跑（需
TORCH_DEVICE_BACKEND_AUTOLOAD=0 免驱动加载 torch_npu）。
"""
import os

os.environ["FUNASR_DEVICE"] = "cpu"
import sys

sys.path.insert(0, "/app")
import numpy as np

import funasr_ts

a1 = (0.1 * np.sin(np.linspace(0, 400 * 2 * np.pi, 24000))).astype("float32")
a2 = (0.1 * np.sin(np.linspace(0, 800 * 2 * np.pi, 28800))).astype("float32")
texts = ["你好世界", "刷Token调用行为"]

print("== single timestamp() ==", flush=True)
single = [funasr_ts.timestamp(a, t, 24000) for a, t in zip([a1, a2], texts)]
for i, r in enumerate(single):
    print("  [%d] text=%r ts=%d" % (i, r["text"], len(r["timestamp"])), flush=True)

print("== batch timestamp_batch() ==", flush=True)
batch = funasr_ts.timestamp_batch([a1, a2], texts, 24000)
for i, r in enumerate(batch):
    print("  [%d] text=%r ts=%d" % (i, r["text"], len(r["timestamp"])), flush=True)

print("== _generate_batch（真批推，batch_size=N）==", flush=True)
model, _dev = funasr_ts.get_model()
res = funasr_ts._generate_batch(model, [a1, a2], texts, 24000)
for i, r in enumerate(res):
    print("  [%d] text=%r ts=%d" % (i, r["text"], len(r["timestamp"])), flush=True)
print("PROBE_DONE", flush=True)

#!/usr/bin/env python3
"""asr 批推 vs 逐条单推 的一致性验证（自动选设备：有 NPU 上卡，无卡 CPU）。

四个判定关卡（进退出码）:
  A. 批推一致性:   _generate_batch(B=6, 变长音频+变长文本) 与逐条
     timestamp() 的字级时间戳必须逐条完全相等（int ms）——混合长度批
     经 _install_batch_safe_predictor 补丁逐条分解 predictor 段
  B. B=1 退化回归: 批入口单条 == 单条入口（同解码函数同前向）
  D. 等长批回归:   等长满宽批走补丁透传的原批路径（热身同形状）——
     该路径本身必须与单条逐位一致（无 padding 时上游批前向无损）
  E. 生产入口对照: batch_group_timestamp（切组/展平/偏移回拼 + 超长
     part 硬切 _split_part 路径）批跑 vs 逐 job group_timestamp 逐毫秒
     相等——两者共用确定性 _split_groups，对照含硬切路径

信息项（不设门，不进退出码）:
  C. 批量吞吐:     批 vs 串行的墙钟对比——NPU 生产收益以 /tts done
     日志的 q_asr/asr 对比为准

用法:
    python tests/validate_asr_batch.py            # 自动：有 NPU 上卡，无卡 CPU
    python tests/validate_asr_batch.py 8          # 指定生效档
    FUNASR_DEVICE=cpu python tests/validate_asr_batch.py   # 强制 CPU
    FUNASR_DEVICE=npu python tests/validate_asr_batch.py   # 强制 NPU
    docker exec kokoro-tts python tests/validate_asr_batch.py  # 自动加载 CANN 后上卡
注意：镜像/pod 内复跑时 B 必须匹配该 pod 的 ASR_MAX_BATCH_SIZE（默认自动取
env，也可显式传参）；验证进程不走 prime_asr_memory 的预占/预算，会额外占卡。
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

# 设备自动选择：FUNASR_DEVICE 未设置/auto 时跟随 pipeline._resolve_device
# （有可用 NPU 就上卡；无 torch_npu 的开发机回落 CPU）；显式 cpu/npu 覆盖。
# docker exec 不会执行镜像 ENTRYPOINT，CANN set_env.sh 未加载时 tbe 缺失；
# 这里自动通过 scripts/with_ascend_env.sh 重入一次，避免手写一长串 source。
_want_npu = (os.getenv("FUNASR_DEVICE", "").strip().lower() != "cpu")
_wrapper = Path(__file__).resolve().parents[1] / "scripts" / "with_ascend_env.sh"


def _cann_env_needed() -> bool:
    if os.getenv("_KOKORO_CANN_REEXEC") == "1":
        return False
    if not _want_npu:
        return False
    try:
        if importlib.util.find_spec("tbe") is not None:
            return False
    except (ImportError, ValueError):
        pass
    return os.path.isfile("/usr/local/Ascend/ascend-toolkit/set_env.sh")


if _cann_env_needed():
    if not _wrapper.is_file():
        print(
            "CANN env not loaded (tbe missing) and scripts/with_ascend_env.sh "
            "not found; source /usr/local/Ascend/ascend-toolkit/set_env.sh "
            "or set FUNASR_DEVICE=cpu", file=sys.stderr)
        raise SystemExit(2)
    print(f"[validate_asr_batch] loading CANN env via {_wrapper}", flush=True)
    os.execvp("bash", ["bash", str(_wrapper), sys.executable, *sys.argv])

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import funasr_ts  # noqa: E402

SR = 24000
N_ITEMS = 6
TEXTS = [
    "你好",
    "今天天气不错我们去公园散步吧顺便买点咖啡",
    "短句",
    "这是一个稍微长一点的句子用来测试时间戳对齐的稳定性",
    "混合中英 mixed language test 一二三",
    "尾句",
]


def _audio(sec: float, seed: int) -> np.ndarray:
    """确定性合成音（正弦+固定种子噪声）：长度差异最大化 padding 影响。"""
    rng = np.random.default_rng(seed)
    n = int(SR * sec)
    return (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SR)
            + 0.05 * rng.standard_normal(n)).astype(np.float32)


def main() -> int:
    ok = True
    model, dev = funasr_ts.get_model()
    print(f"device selection: FUNASR_DEVICE="
          f"{os.getenv('FUNASR_DEVICE', '<unset>')} -> {dev}")
    print(f"device={dev} input_mode={funasr_ts._input_mode['mode']}"
          f" model={Path(funasr_ts._resolve_model_dir()).name}"
          f" patch={funasr_ts.batch_patch_status()}")

    audios = [_audio(1.0 + 0.5 * i, 1000 + i) for i in range(N_ITEMS)]
    texts = TEXTS[:N_ITEMS]

    # A. 批推 vs 逐条：时间戳逐条完全相等
    t0 = time.perf_counter()
    single = [funasr_ts.timestamp(a, t, SR) for a, t in zip(audios, texts)]
    t_single = time.perf_counter() - t0
    t0 = time.perf_counter()
    batch = funasr_ts._generate_batch(model, audios, texts, SR)
    t_batch = time.perf_counter() - t0
    for i, (s, b) in enumerate(zip(single, batch)):
        ts_s, ts_b = s.get("timestamp"), b.get("timestamp")
        same = ts_s == ts_b
        ok &= same
        detail = ("identical" if same else
                  f"DIFF first3 single={ts_s[:3]} batch={ts_b[:3]}")
        print(f"[{'PASS' if same else 'FAIL'}] item {i}"
              f" ({len(audios[i]) / SR:.1f}s, {len(ts_s)} 字): {detail}")
    print(f"[INFO] C. 批量吞吐: serial {t_single * 1000:.0f}ms"
          f" vs batch {t_batch * 1000:.0f}ms"
          f" (x{t_single / max(t_batch, 1e-9):.2f})，不设门")

    # B. B=1 退化回归
    s1 = funasr_ts.timestamp(audios[0], texts[0], SR)
    b1 = funasr_ts._generate_batch(model, [audios[0]], [texts[0]], SR)[0]
    same = s1.get("timestamp") == b1.get("timestamp")
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] B=1 退化回归")

    # D. 等长批回归（补丁透传路径）：等长满宽批走原批路径（_install_batch_
    # safe_predictor 等长直通）——该路径本身必须与单条一致（无 padding 时
    # 上游批前向无损）。热身形状（全部同长 tone）走的就是这条路径。
    eq_audios = [_audio(2.0, 2000 + i) for i in range(4)]
    eq_texts = TEXTS[:4]
    single_eq = [funasr_ts.timestamp(a, t, SR) for a, t in zip(eq_audios, eq_texts)]
    batch_eq = funasr_ts._generate_batch(model, eq_audios, eq_texts, SR)
    same = all(s.get("timestamp") == b.get("timestamp")
               for s, b in zip(single_eq, batch_eq))
    ok &= same
    print(f"[{'PASS' if same else 'FAIL'}] 等长批 B=4（补丁透传路径）")

    # E. 生产入口 batch_group_timestamp：批 vs 逐 job group_timestamp 逐毫秒
    #    相等（切组/展平/偏移回拼 + 超长 part 硬切 _split_part 路径一并对照
    #    ——两路径共用确定性 _split_groups）。生效档默认取 ASR_MAX_BATCH_SIZE
    #    （与生产预算/pod 钉值一致），可用第一个命令行参数覆盖；结束后恢复，
    #    避免把验证态留给后续调用。
    b_env = os.getenv("ASR_MAX_BATCH_SIZE", "4")
    try:
        gate_b = max(1, int(b_env))
    except ValueError:
        gate_b = 4
    if len(sys.argv) > 1:
        gate_b = max(1, int(sys.argv[1]))
    old_effective = funasr_ts._batch_effective["n"]
    funasr_ts._batch_effective["n"] = gate_b
    patch = funasr_ts.batch_patch_status() or {}
    actual_b = funasr_ts.effective_batch()
    if patch.get("needed") and not patch.get("ok"):
        ok = False
        print("[FAIL] E 节批推安全补丁未安装（effective_batch 强制 1）"
              f" patch={patch}")
    if actual_b != gate_b:
        ok = False
        print(f"[FAIL] E 节实际生效档 B{actual_b} != 期望 B{gate_b}"
              f"（env ASR_MAX_BATCH_SIZE={b_env}）")
    print(f"[INFO] E 节生效档 = B{actual_b}（期望 B{gate_b}，"
          f"env ASR_MAX_BATCH_SIZE={b_env}）")
    seg = _audio(8.0, 3003)
    gap = np.zeros(int(SR * 0.5), dtype=np.float32)
    long_audio = np.concatenate(
        [p for i in range(8) for p in ((seg, gap) if i < 7 else (seg,))])
    long_text = "今天天气不错我们去公园散步吧顺便买点咖啡然后去看电影最后回家休息。" * 3
    items = [
        ([_audio(1.0, 3000), _audio(1.2, 3001)], ["你好", "世界你好"]),
        ([_audio(2.0, 3002)], ["这是一个正常长度的句子"]),
        ([long_audio], [long_text]),   # 67.5s > 30s：触发 _split_part 硬切
    ]
    try:
        batch_jobs = funasr_ts.batch_group_timestamp(items, SR, max_group_sec=30)
        for i, (parts, texts) in enumerate(items):
            single = funasr_ts.group_timestamp(parts, texts, SR, 30)
            same = single.get("timestamp") == batch_jobs[i].get("timestamp")
            ok &= same
            print(f"[{'PASS' if same else 'FAIL'}] E. 生产入口 batch_group_timestamp"
                  f" job {i} ({sum(len(p) for p in parts) / SR:.1f}s,"
                  f" {len(single.get('timestamp') or [])} 字)")
    finally:
        funasr_ts._batch_effective["n"] = old_effective

    print("VALIDATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""POC②：forward_batch 分段计时——钉死全局串行段（64%）坐在哪个算子。

背景
    poc_stream_concurrency 测得多流并发总吞吐封顶 ~1.4x（Amdahl s≈0.64），
    raw≈synth 排除了 Python 预处理。本脚本把 forward_batch 逐段包上
    sync+计时，跑 B=8 真实形状，输出每段毫秒/占比，直接指认串行大段；
    同时打印 LSTM 模块的参数张量数（8=双向 → aclnn 不支持 → acl_op
    fallback → 全局串行嫌疑）。

怎么跑（空闲卡上最干净；同服务卡需保持空闲）
    docker cp 不需要——宿主机挂载目录更新脚本后直接：
    docker run --rm -it \\
      --device /dev/davinci3 --device /dev/davinci_manager \\
      --device /dev/devmm_svm --device /dev/hisi_hdc \\
      -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\
      -v /usr/local/dcmi:/usr/local/dcmi:ro \\
      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
      -v /root/cui:/poc -e PYTHONPATH=/app -e ASCEND_DEVICE_ID=0 -w /app \\
      --entrypoint /app/scripts/with_ascend_env.sh \\
      kokoro-tts-ascend:20261113-r2 \\
      python3 /poc/poc_forward_stages.py --device npu:0

判读
    占比最大的段 = 串行点（结合是否 LSTM 段对照 acl_op 嫌疑）；
    段和 vs 整体 forward_batch 墙钟的差值 = 计时/同步开销（应很小）。
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback

_POC_TEXT = ("华为昇腾处理器面向人工智能训练与推理场景，语音合成服务在"
             "容器队列中弹性扩容，批次吞吐随并发拐点稳定上升。")


def parse_args():
    ap = argparse.ArgumentParser(description="forward_batch 分段计时 POC")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--iters", type=int, default=5, help="每段计时重复次数")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--target-mb", type=int, default=8000,
                    help="单前向库存预算 MB（自动缩批用）")
    return ap.parse_args()


def fail(msg: str) -> None:
    print(f"VERDICT: INCONCLUSIVE — {msg}")
    sys.exit(1)


def main() -> None:
    args = parse_args()
    import torch

    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        fail(f"import torch_npu 失败: {exc}")
    npu = torch.npu
    dev_index = int(args.device.split(":")[1]) if ":" in args.device else 0
    npu.set_device(dev_index)

    # 与 poc_stream_concurrency 同款加载：跳过预算/预占，轻装
    import pipeline

    pipeline._prime_memory = lambda device: None
    voice = pipeline._default_voice
    compute = pipeline.TTSCompute(args.device)

    from batched_model import (_clear_adain_mask, _run_packed_lstm,
                               _set_adain_mask, finalize_audio,
                               forward_batch, install_adain_mask_patch,
                               phonemes_to_ids)

    # 音素串与输入（raw 预构建，排除 Python 噪声）
    try:
        ps_list = pipeline.get_g2p().phonemize(_POC_TEXT)
        ps = ps_list[0] if ps_list else ""
    except Exception as exc:  # noqa: BLE001
        print(f"g2p fallback: {exc}")
        ps = "ə" * 72
    if not ps:
        fail("g2p 产出空音素串")

    # 自动缩批：单前向库存 ≤ target（服务同卡跑时保 10GB 空闲的安全线）
    ids = phonemes_to_ids(compute.model, ps)
    l_ids = len(ids)
    npu.synchronize()
    r0 = npu.memory_reserved()
    out_b1 = compute.synth_batch([(ps, voice, 1.0)], args.dtype)
    npu.synchronize()
    live1 = max((npu.memory_reserved() - r0) / 2**20, 64.0)
    B = max(1, min(args.batch, int(args.target_mb / (1.4 * live1))))
    input_ids = torch.zeros((B, l_ids), dtype=torch.long)   # 0 = PAD_ID
    for i in range(B):
        input_ids[i, :l_ids] = torch.tensor(ids, dtype=torch.long)
    lengths = torch.full((B,), l_ids, dtype=torch.long)
    ref_s = torch.cat([compute.make_ref(ps, voice)] * B, dim=0)
    speeds = torch.ones(B, dtype=torch.float32)
    input_ids_d = input_ids.to(compute.device)
    ref_s_d = ref_s.to(compute.device)
    speeds_d = speeds.to(compute.device)
    lengths_d = lengths.to(compute.device)
    for _ in range(2):   # 最终形状 warmup（库存/算法覆盖）
        out0 = forward_batch(compute.model, input_ids_d, lengths,
                             ref_s_d, speeds_d)
    npu.synchronize()
    samples = int(out0[0].shape[0]) if hasattr(out0[0], "shape") else 0
    print(f"env    : torch={torch.__version__} device={args.device} "
          f" dtype={args.dtype}")
    print(f"script : poc_forward_stages v2（inference_mode 修复版）"
          f"——若输出缺此行，说明宿主机跑的是旧文件")
    print(f"shape  : B={B} tokens={l_ids} live/B1={live1:.0f}MB "
          f"audio/句={samples / 24000.0:.1f}s")

    # LSTM 结构指认：参数张量数（4=单向，8=双向 → aclnn 不支持 → acl_op）
    for name in ("predictor.lstm", "predictor.shared"):
        mod = getattr(compute.model.predictor, name.split(".")[1])
        n_w = len(getattr(mod, "_flat_weights", []))
        print(f"lstm   : {name:<18} bidirectional={mod.bidirectional} "
              f"params={n_w} {'→ aclnn 预期 4，必 fallback acl_op' if n_w != 4 else ''}")

    # 分段计时：完整复制 forward_batch 的算子序列，逐段 sync+计时
    # 必须与原函数同款 inference_mode——否则 autograd 保留 decoder 全分辨率
    # 激活链，B=8 直接吃满 60GB（首跑 OOM 的根因）
    t = {}   # stage -> 累计 ms
    def stage(name, fn):
        npu.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            r = fn()
        npu.synchronize()
        t[name] = t.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        return r

    n = args.iters
    for _ in range(n):
        lengths_cpu = lengths.detach().cpu().long()
        il_d = lengths_d

        def _bert():
            mask = (torch.arange(l_ids, device=il_d.device).unsqueeze(0)
                    >= il_d.unsqueeze(1))
            return stage("bert(plBERT)", lambda: compute.model.bert(
                input_ids_d, attention_mask=(~mask).int()))
        bert_dur = stage("bert+mask", _bert)

        d_en = stage("bert_encoder",
                     lambda: compute.model.bert_encoder(bert_dur).transpose(-1, -2))
        s = ref_s_d[:, 128:]
        d = stage("pred.text_encoder", lambda: compute.model.predictor.text_encoder(
            d_en, s, il_d, il_d.unsqueeze(1) <= torch.arange(
                l_ids, device=il_d.device).unsqueeze(0)))
        x = stage("LSTM①pred.lstm", lambda: _run_packed_lstm(
            compute.model.predictor.lstm, d, lengths_cpu))

        def _dur():
            duration = compute.model.predictor.duration_proj(x)
            mask = (torch.arange(l_ids, device=il_d.device).unsqueeze(0)
                    >= il_d.unsqueeze(1))
            duration = torch.sigmoid(duration).sum(axis=-1).masked_fill(mask, 0.0)
            return torch.round(duration / speeds_d.unsqueeze(1)).clamp(min=1).long(), mask
        pred_dur, text_mask = stage("duration_proj", _dur)

        def _aln():
            frames = pred_dur.masked_fill(text_mask, 0).sum(dim=1)
            frames_cpu = frames.cpu().long()
            T_max = int(frames_cpu.max())
            aln = torch.zeros(B, l_ids, T_max, device=il_d.device)
            for i in range(B):
                Li, Fi = int(lengths_cpu[i]), int(frames_cpu[i])
                idx = torch.repeat_interleave(
                    torch.arange(Li, device=il_d.device), pred_dur[i, :Li])
                aln[i, idx, torch.arange(Fi, device=il_d.device)] = 1.0
            return aln, frames, frames_cpu, T_max
        aln, frames, frames_cpu, T_max = stage("align_build(逐样本循环)", _aln)

        en = stage("en_matmul", lambda: d.transpose(-1, -2) @ aln)
        frame_mask = stage("frame_mask", lambda: (
            torch.arange(T_max, device=il_d.device).unsqueeze(0)
            < frames.unsqueeze(1)).float().unsqueeze(1))
        install_adain_mask_patch()
        _set_adain_mask(frame_mask, all_real=bool((frames_cpu == T_max).all()))
        try:
            x_f = stage("LSTM②shared(长序列)", lambda: _run_packed_lstm(
                compute.model.predictor.shared, en.transpose(-1, -2), frames_cpu))
            F0 = x_f.transpose(-1, -2)

            def _f0n():
                F0_ = F0
                for block in compute.model.predictor.F0:
                    F0_ = block(F0_, s)
                F0_pred = compute.model.predictor.F0_proj(F0_).squeeze(1)
                N_ = x_f.transpose(-1, -2)
                for block in compute.model.predictor.N:
                    N_ = block(N_, s)
                N_pred = compute.model.predictor.N_proj(N_).squeeze(1)
                return F0_pred, N_pred
            F0_pred, N_pred = stage("F0/N AdaIN blocks", _f0n)

            def _pad2():
                pad2 = (torch.arange(2 * T_max, device=il_d.device).unsqueeze(0)
                        >= (2 * frames).unsqueeze(1))
                return (F0_pred.masked_fill(pad2, 0.0),
                        N_pred.masked_fill(pad2, 0.0))
            F0_pred, N_pred = stage("pad_fill", _pad2)

            t_en = stage("text_encoder", lambda: compute.model.text_encoder(
                input_ids_d, il_d, text_mask))
            asr_m = stage("asr_matmul", lambda: t_en @ aln)
            audio = stage("decoder(音频生成,最重段?)", lambda: compute.model.decoder(
                asr_m, F0_pred, N_pred, ref_s_d[:, :128]))
        finally:
            _clear_adain_mask()
        stage("finalize(D2H)", lambda: finalize_audio(audio, frames_cpu,
                                                      audio.shape[-1] // T_max))

    # 整体未分段墙钟对照（sync 开销量化）
    npu.synchronize()
    t0 = time.perf_counter()
    forward_batch(compute.model, input_ids_d, lengths, ref_s_d, speeds_d)
    npu.synchronize()
    full_ms = (time.perf_counter() - t0) * 1000

    total = sum(t.values())
    per_iter = {k: v / n for k, v in t.items()}
    total_iter = sum(per_iter.values())   # 百分比基准 = 单轮合计（v2 修正：
    print(f"\n== 分段耗时（{n} 次均值，B={B}）==")   # 曾误用 n 轮累计导致占比÷5）
    for k in sorted(per_iter, key=per_iter.get, reverse=True):
        print(f"{k:<28} {per_iter[k]:8.1f} ms  {per_iter[k] / total_iter * 100:5.1f}%")
    print(f"{'合计(分段)':<28} {total_iter:8.1f} ms")
    print(f"{'整体 forward_batch 墙钟':<28} {full_ms:8.1f} ms "
          f"(差值={full_ms - total_iter:.1f} ms = 分段 sync 开销)")
    top = max(per_iter, key=per_iter.get)
    print(f"\nVERDICT: 串行大头 = {top}（{per_iter[top] / total_iter * 100:.0f}%）"
          f"——对照上表 LSTM 段与参数张量数定位改法")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        fail("分段计时异常（见堆栈）")

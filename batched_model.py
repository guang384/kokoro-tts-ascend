"""Kokoro KModel 批量前向：padding + mask，一次 NPU/CPU 前向合成 B 个句子。

基于 kokoro 0.9.x 的 KModel.forward_with_tokens 结构改造，与官方单样本
forward 保持一致性的关键点：

1. input_ids 右侧 pad 到 L_max（pad id=0，与 BOS/EOS 同 id），text_mask 屏蔽；
   plBERT 用 attention_mask，TextEncoder/DurationEncoder 内部本就 pack_padded。
2. predictor.lstm 与 F0Ntrain.shared 两处裸调的双向 LSTM 改为 pack_padded，
   否则反向方向会从 padding 零一路污染回真实 token（时长/F0 偏移）。
3. 时长在 pad 位置强制为 0；对齐矩阵逐样本构造后堆叠 [B, L_max, T_max]，
   en/t_en 用一次 bmm 展开到帧域。
4. AdaIN1d 的 InstanceNorm 会把 pad 尾巴统计进去 → masked 统计补丁 + pad 区
   置零（见下方 AdaIN 补丁注释）。
5. 音频裁剪：istft 输出长度 = 帧数 × 定比（无常数项），逐样本精确裁剪。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_ID = 0
_SAMPLE_RATE = 24000          # 与 KOKORO_SAMPLE_RATE 默认一致（淡出窗口用）

# ---------------------------------------------------------------------------
# AdaIN1d 的 InstanceNorm1d 对整条时间轴做统计，批量时 pad 零尾巴会污染
# 均值/方差（F0 音高偏移、音色增益漂移）。这里给 AdaIN1d.forward 打上下文
# 补丁：设置了 mask 时改用逐样本 masked 统计；未设置时走原实现（含 B=1）。
# mask 时间长度随解码器内部上/下采样自动 nearest 对齐。
# ---------------------------------------------------------------------------
_ADAIN_MASK = {"m": None, "all_real": True}  # m: float [B, 1, T0]，1=真实帧

_ADAIN1D_ORIG_FORWARD = None


def _set_adain_mask(mask: torch.Tensor, all_real: bool | None = None):
    """登记 AdaIN mask；all_real 可由调用方（CPU 侧帧数）给出，避免 .all() 同步。"""
    global _ADAIN_MASK
    _ADAIN_MASK = {"m": mask,
                   "all_real": bool(mask.all()) if all_real is None else all_real}


def _clear_adain_mask():
    global _ADAIN_MASK
    _ADAIN_MASK = {"m": None, "all_real": True}


def _patched_adain_forward(self, x, s):
    m = _ADAIN_MASK["m"]
    if m is None or _ADAIN_MASK["all_real"] or x.dim() < 3:
        return _ADAIN1D_ORIG_FORWARD(self, x, s)
    T = x.shape[-1]
    if T != m.shape[-1]:  # 块内上/下采样后自动对齐 mask
        m = F.interpolate(m, size=T, mode="nearest")
    h = self.fc(s)
    h = h.view(h.size(0), h.size(1), 1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    cnt = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
    mean = (x * m).sum(dim=-1, keepdim=True) / cnt
    var = ((x - mean) * m).pow(2).sum(dim=-1, keepdim=True) / cnt
    xn = (x - mean) / torch.sqrt(var + self.norm.eps)
    if self.norm.affine:  # 与 nn.InstanceNorm1d(affine=True) 一致，保留其 weight/bias
        xn = xn * self.norm.weight.view(1, -1, 1) + self.norm.bias.view(1, -1, 1)
    out = (1 + gamma) * xn + beta
    # pad 区置零：否则 (0-mean)/std*γ+β 的非零垃圾会经后续卷积感受野
    # 渗入最后几个真实帧；单样本路径的"序列末端之外"是真零，必须对齐
    return out * m


def install_adain_mask_patch():
    """幂等安装。所有 istftnet.AdaIN1d 实例（预测器 F0/N 与解码器）生效。"""
    from kokoro import istftnet

    global _ADAIN1D_ORIG_FORWARD
    if _ADAIN1D_ORIG_FORWARD is not None:
        return
    _ADAIN1D_ORIG_FORWARD = istftnet.AdaIN1d.forward
    istftnet.AdaIN1d.forward = _patched_adain_forward


def phonemes_to_ids(model, ps: str) -> list:
    """与 KModel.forward 相同的音素 -> id 映射（保留 id=0，只滤 None），加 BOS/EOS。"""
    ids = list(filter(lambda i: i is not None, map(model.vocab.get, ps)))
    assert len(ids) + 2 <= model.context_length, \
        f"phonemes too long: {len(ids) + 2} > {model.context_length}"
    return [PAD_ID, *ids, PAD_ID]


def _run_packed_lstm(lstm: nn.LSTM, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """x: [B, T, C]，按 lengths pack 过 LSTM 再展开，pad 区为 0（与单样本等价）。"""
    packed = nn.utils.rnn.pack_padded_sequence(
        x, lengths.detach().cpu().long(), batch_first=True, enforce_sorted=False)
    lstm.flatten_parameters()
    out, _ = lstm(packed)
    out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    if out.shape[1] < x.shape[1]:  # pad_packed 展开到 max(lengths)，兜底补齐
        pad = torch.zeros(out.shape[0], x.shape[1] - out.shape[1], out.shape[2],
                          device=out.device, dtype=out.dtype)
        out = torch.cat([out, pad], dim=1)
    return out


@torch.inference_mode()  # 比 no_grad 更省：禁用张量版本计数；输出只走 numpy 编码，无自动求导需求
def forward_batch(model, input_ids: torch.LongTensor, input_lengths: torch.LongTensor,
                  ref_s: torch.FloatTensor, speeds: torch.FloatTensor) -> list:
    """批量合成。

    Args:
        model: KModel（已 .to(device).eval()）
        input_ids: [B, L_max] 右侧 pad（PAD_ID）
        input_lengths: [B] 每样本真实长度（含 BOS/EOS）
        ref_s: [B, 256] 每样本风格向量（voice_pack[len(ps)-1]）
        speeds: [B] 每样本语速
    Returns:
        [np.ndarray int16] 每样本 PCM16 音频（24kHz，fade/量化在卡上完成，
        已裁剪到自身帧数——ship_batch 直写 SHM 零数值处理）
    """
    B, L = input_ids.shape
    device = input_ids.device
    lengths_cpu = input_lengths.detach().cpu().long()   # 先算 CPU 侧（原 H2D 后再 D2H）
    input_lengths = input_lengths.to(device)

    # ---- 文本侧：plBERT + duration predictor（与单样本路径一致，LSTM 走 pack）----
    text_mask = torch.arange(L, device=device).unsqueeze(0) >= input_lengths.unsqueeze(1)  # True=pad
    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)                    # [B,128,L]
    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)      # [B,L,128]
    x = _run_packed_lstm(model.predictor.lstm, d, lengths_cpu)  # 复用 CPU lengths，省一次 D2H 同步

    duration = model.predictor.duration_proj(x)                              # [B,L,max_dur]
    duration = torch.sigmoid(duration).sum(axis=-1).masked_fill(text_mask, 0.0)
    pred_dur = torch.round(duration / speeds.unsqueeze(1)).clamp(min=1).long()  # [B,L]

    frames = pred_dur.masked_fill(text_mask, 0).sum(dim=1)                   # [B] 每样本总帧
    frames_cpu = frames.cpu().long()
    T_max = int(frames_cpu.max())

    # ---- 对齐矩阵：逐样本构造 [L_i, F_i]，堆叠为 [B, L_max, T_max] ----
    # 工厂函数不进 autocast（恒默认 fp32）：fp16 前向里 aln 保持 fp32，
    # 随后的 matmul 由 autocast 统一选型——无需显式 dtype
    aln = torch.zeros(B, L, T_max, device=device)
    for i in range(B):
        Li, Fi = int(lengths_cpu[i]), int(frames_cpu[i])
        idx = torch.repeat_interleave(torch.arange(Li, device=device), pred_dur[i, :Li])
        aln[i, idx, torch.arange(Fi, device=device)] = 1.0

    en = d.transpose(-1, -2) @ aln                                           # [B,128,T_max]

    # 帧级真实 mask（1=真实帧）：AdaIN masked 统计用
    frame_mask = (torch.arange(T_max, device=device).unsqueeze(0) < frames.unsqueeze(1))
    frame_mask = frame_mask.float().unsqueeze(1)                             # [B,1,T_max]

    if _ADAIN1D_ORIG_FORWARD is None:   # 补丁幂等，但每批重复调用无谓
        install_adain_mask_patch()
    # all_real 由 CPU 侧帧数判定（原 mask.all() 每批一次设备→主机同步）
    _set_adain_mask(frame_mask,
                    all_real=bool((frames_cpu == T_max).all()))
    try:
        # ---- F0 / N：shared LSTM 走 pack，pad 区为 0；AdaIN 用 masked 统计 ----
        x_f = _run_packed_lstm(model.predictor.shared, en.transpose(-1, -2), frames_cpu)  # [B,T,64]
        F0 = x_f.transpose(-1, -2)
        for block in model.predictor.F0:
            F0 = block(F0, s)
        F0_pred = model.predictor.F0_proj(F0).squeeze(1)                     # [B,2T]
        N = x_f.transpose(-1, -2)
        for block in model.predictor.N:
            N = block(N, s)
        N_pred = model.predictor.N_proj(N).squeeze(1)                        # [B,2T]
        # pad 帧 F0/N 置零：与单样本路径"末尾之后是真零"的边缘行为一致，
        # 否则 beta 偏置等非零 pad 值会经解码器卷积感受野渗入最后几个真实帧
        pad2 = torch.arange(2 * T_max, device=device).unsqueeze(0) >= (2 * frames).unsqueeze(1)
        F0_pred = F0_pred.masked_fill(pad2, 0.0)
        N_pred = N_pred.masked_fill(pad2, 0.0)

        # ---- 文本编码 + 帧域展开 + 解码（解码器内部各 AdaIN 同样 masked）----
        t_en = model.text_encoder(input_ids, input_lengths, text_mask)       # [B,128,L]
        asr = t_en @ aln                                                     # [B,128,T_max]
        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128])          # [B,1,S]
    finally:
        _clear_adain_mask()

    # ---- 逐样本裁剪：istft 输出长度与帧数成正比（无常数项），精确切分 ----
    ratio = audio.shape[-1] // T_max
    return finalize_audio(audio, frames_cpu, ratio)

_FADE_RAMPS: dict = {}   # (device, fade) → 卡上 ramp（值来自 np.linspace，与旧 CPU 路径逐位一致）


def _fade_ramp(device, fade: int):
    """5ms 淡出斜坡（1→0，方向事故史见 git）；按 (设备, 长度) 缓存。"""
    key = (str(device), fade)
    t = _FADE_RAMPS.get(key)
    if t is None:
        t = torch.from_numpy(
            np.linspace(1.0, 0.0, fade, dtype=np.float32)).to(device)
        _FADE_RAMPS[key] = t
    return t


def finalize_audio(audio: torch.Tensor, frames_cpu: torch.Tensor,
                   ratio: int) -> list:
    """fade + PCM16 量化在卡上完成 → 整批一次 D2H 直出 int16 → 按帧裁剪。

    数值顺序与旧 CPU 路径逐段一致（等价性由测试逐字节锁定）：
      fade(float 域，仅每句真实末尾 fade 个样本) → nan_to_num → clip(±1)
      → ×32767 → round（np.rint/torch.round 同为 half-to-even）→ int16
    收益：D2H 字节数减半（fp32→int16），CPU 侧 5 遍全量扫描归零，
    ship_batch 只剩按帧数的纯 memcpy 裁剪直写 SHM。
    """
    B = audio.shape[0]
    fade = int(0.005 * _SAMPLE_RATE)
    audio = audio.detach().float()
    ramp = _fade_ramp(audio.device, fade)
    for i in range(B):
        end = int(frames_cpu[i]) * ratio
        if end > fade:   # 旧路径：短于 fade 窗的句不淡出
            audio[i, 0, end - fade:end] *= ramp
    audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = (audio.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    audio_cpu = audio.cpu()   # 整批一次 D2H（int16，字节数减半）
    out = []
    for i in range(B):
        out.append(audio_cpu[i, 0, :int(frames_cpu[i]) * ratio].numpy())
    return out

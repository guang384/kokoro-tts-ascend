"""gateway 音频编码：WAV/FLAC/PCM 直出 + ffmpeg 系封装 + chunk 拼接。"""
import io
import subprocess
from typing import List

import numpy as np
import soundfile as sf

import gateway_state as gst


def _concat(parts: List[np.ndarray]) -> np.ndarray:
    """拼接句音频（PCM16），句间补 60ms 静音；恒定返回自有数组。

    单片快捷返回（单句请求常见）——入参已是 copy 的自有数组，无悬垂风险。"""
    if not parts:
        return np.zeros(0, dtype="<i2")
    if len(parts) == 1:
        return parts[0]
    gap = np.zeros(int(gst._sample_rate * gst._CHUNK_GAP_S), dtype="<i2")
    out = []
    for i, p in enumerate(parts):
        if i:
            out.append(gap)
        out.append(p)
    return np.concatenate(out)


def _encode_wav(audio: np.ndarray) -> bytes:
    """PCM16 int16 直写（soundfile 对 int16 输入无损直写，不再经 float）。"""
    buf = io.BytesIO()
    sf.write(buf, audio, gst._sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _encode_flac(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, gst._sample_rate, subtype="PCM_16", format="FLAC")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    return audio.tobytes()   # 已是 PCM16 小端


_FFMPEG_ARGS = {
    "mp3": ("-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"),
    "opus": ("-c:a", "libopus", "-b:a", "96k", "-f", "ogg"),
    "aac": ("-c:a", "aac", "-b:a", "128k", "-f", "adts"),
}


def _encode_ffmpeg(audio: np.ndarray, fmt: str) -> bytes:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "s16le", "-ar", str(gst._sample_rate), "-ac", "1", "-i", "pipe:0",
        *_FFMPEG_ARGS[fmt], "pipe:1",
    ]
    # 超时必设：挂死的 ffmpeg 会永久占住 run_in_threadpool 的线程槽，
    # 反复挂死即耗尽线程池、拖垮所有编码请求（run 超时后自动 kill 子进程）
    try:
        proc = subprocess.run(cmd, input=_encode_pcm(audio),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=gst._ffmpeg_timeout_s)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg {fmt} 编码超时(>{gst._ffmpeg_timeout_s:.0f}s)，已终止")
    except FileNotFoundError:
        # 缺二进制：裸穿会是 WinError 2 / ENOENT 乱码——给运维可读的原因
        raise RuntimeError(f"ffmpeg 未安装，无法编码 {fmt}（镜像内已内置；裸机部署需安装 ffmpeg）")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg {fmt} 编码失败: {proc.stderr.decode(errors='ignore')[:300]}")
    return proc.stdout

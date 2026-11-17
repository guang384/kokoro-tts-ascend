"""compute（兼容 facade）：两个推理角色入口 + 全部历史符号的聚合视图。

supervisor 经 compute.tts_main / compute.asr_main spawn 工人；测试与脚本
仍从本模块 import。实现分布：worker_common（共享状态/脚手架）、
tts_worker（KModel 前向 + SHM 写回）、asr_worker（FunASR 对齐）。
"""
from asr_worker import (_asr_snapshot, _asr_state, _load_asr, _read_job_audio,
                        _run_asr_batch, _sample_asr_mem, asr_main)
from tts_worker import (_WIN_SHM_KEEPALIVE, _load_compute, _sample_npu_mem,
                        _tts_snapshot, ship_batch, tts_main)
from worker_common import (SAMPLE_RATE, _ASR_MAX_BATCH, _MAX_BATCH, _busy,
                           _looks_like_oom, _reply_chunk_ts,
                           _reply_chunk_ts_safe, _stats, _with_busy,
                           group_by_request)

__all__ = [
    "SAMPLE_RATE", "_ASR_MAX_BATCH", "_MAX_BATCH", "_WIN_SHM_KEEPALIVE",
    "_asr_snapshot", "_asr_state", "_busy", "_load_asr", "_load_compute",
    "_looks_like_oom", "_read_job_audio", "_reply_chunk_ts",
    "_reply_chunk_ts_safe", "_run_asr_batch", "_sample_asr_mem",
    "_sample_npu_mem", "_stats", "_tts_snapshot", "_with_busy", "asr_main",
    "group_by_request", "ship_batch", "tts_main",
]

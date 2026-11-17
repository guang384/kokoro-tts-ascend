# =====================================================================
# 基础镜像：华为官方 torch_npu 镜像
#   torch 2.14.0 + torch_npu + CANN 9.2.0 已在镜像内严格配套，免手动安装
#   Atlas 800T A2 为 aarch64(鲲鹏)，构建/拉取注意 --platform=arm64
#   如需换版本，改 BASE_IMAGE 即可（tag 规则: torch-npu:<torch>-cann<cann>-910b-ubuntu22.04-py<py>）
# =====================================================================
ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/torch-npu:2.14.0-cann9.2.0-910b-ubuntu22.04-py3.12
# 构建期 pip 走国内源；模型走 hf-mirror
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG HF_ENDPOINT_BUILD=https://hf-mirror.com
# 镜像版本号：构建时传 --build-arg IMAGE_VERSION=$(date +%Y%m%d)-rN，同步烧进镜像与 /health
ARG IMAGE_VERSION=dev

FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL
ARG HF_ENDPOINT_BUILD
ARG IMAGE_VERSION

ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    HF_ENDPOINT=${HF_ENDPOINT_BUILD} \
    HF_HUB_OFFLINE=1 \
    KOKORO_MODEL_ID=hexgrad/Kokoro-82M-v1.1-zh \
    KOKORO_SAMPLE_RATE=24000 \
    DEVICE= \
    KOKORO_VOICE=zf_001 \
    TTS_MAX_INPUT_CHARS=2000 \
    MAX_INFLIGHT_SENTENCES=512 \
    MAX_PENDING_REQUESTS=200 \
    WORKER_MAILBOX_SIZE=1024 \
    GATEWAY_WORKERS=2 \
    G2P_WORKERS=2 \
    TTS_WORKERS=4 \
    ASR_WORKERS=4 \
    WAIT_TIMEOUT_S=120 \
    TTS_BATCH_BUCKET_CHARS=64 \
    TTS_SPEED_FLOOR=0.5 \
    KOKORO_EN_G2P=auto \
    FUNASR_DEVICE=auto \
    LOG_LEVEL=INFO \
    TTS_CHUNK_CHARS=120 \
    TTS_MAX_BATCH_SIZE=8 \
    ASR_MAX_BATCH_SIZE=8 \
    PORT=5174 \
    FUNASR_MODEL_DIR=/opt/funasr-models/timestamp-v1 \
    ASCEND_DEVICE_ID= \
    PYTHONPATH=/app

# OCI 元数据（版本号在末层烧入——IMAGE_VERSION 是 build-arg，在其使用点
# 之后才参与缓存键，放末层使改 tag 不再重建 pip/模型下载层）
LABEL org.opencontainers.image.title="kokoro-tts-ascend"

RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng ffmpeg curl redis-server \
    && rm -rf /var/lib/apt/lists/*

# pypinyin 固定 0.55.0：请求级多音字覆盖依赖其内部结构
# （PHRASES_DICT / core.seg / mmseg），升级前必须重跑 tests/test_g2p_overrides.py
RUN pip install --no-cache-dir \
        "kokoro==0.9.4" "misaki[zh]==0.9.4" \
        "pypinyin==0.55.0" \
        "soundfile>=0.12" "numpy>=1.26" \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" \
        "huggingface_hub>=0.23" "httpx>=0.27" \
        "phonemizer" "espeakng-loader" \
        "funasr==1.4.21" "kaldi-native-fbank" "prometheus-client>=0.20" "redis>=5"
# funasr 钉 1.4.21：批推安全补丁依赖其内部结构（AutoModel.model.predictor /
# get_upsample_timestamp / mask 形状 / frontend_conf 合并）。升级前重跑
# tests/validate_asr_batch.py，并确认 _batch_patch=ok（否则运行时自动降单条）。

# 英文 G2P：pip 装本地 wheel（en_core_web_sm 3.8.x，配套镜像内 spacy 3.8）。
# 不能用 `python -m spacy download`：它会 import spacy→thinc→torch，触发 oneDNN
# xbyak_aarch64 的 CPU 探测，QEMU 交叉构建下段错误/挂死（同下方 FunASR 注释之坑）；
# pip 装 wheel 只做元数据安装，QEMU 安全。运行期真机 arm64 无此问题。
# 构建期硬失败：装不上就断构建，不出静默降级的镜像（运行期仍保留
# spacy 缺失时降级 espeak-ng 的兜底，两码事）。
# wheel 由构建机预下载到 wheels/（不在构建期连 GitHub——间歇性连不通会
# 直接断构建；wheel 与官方安装的 RECORD 逐文件 sha256 对拍过）：
#   curl -fL -o wheels/en_core_web_sm-3.8.0-py3-none-any.whl \
#     https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
COPY wheels/en_core_web_sm-3.8.0-py3-none-any.whl /tmp/
RUN pip install --no-cache-dir /tmp/en_core_web_sm-3.8.0-py3-none-any.whl \
    && python -c "from importlib.metadata import version; assert version('en_core_web_sm') == '3.8.0'" \
    && rm -f /tmp/en_core_web_sm-3.8.0-py3-none-any.whl

# 预下载模型权重 + 103 个音色进镜像（此后运行期完全离线）。
# hf-mirror 偶发瞬断/挂起：单轮 15 分钟兜底超时 + 最多 5 轮重试；
# 全部失败仍硬失败断构建（不静默出缺权重的镜像）
RUN set -eu; \
    ok=0; \
    for i in 1 2 3 4 5; do \
      if timeout 900 env HF_HUB_OFFLINE=0 python -c \
           "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['KOKORO_MODEL_ID'])"; then \
        ok=1; break; \
      fi; \
      echo "[build] snapshot_download 第 $i/5 轮失败，5s 后重试..."; \
      sleep 5; \
    done; \
    test "$ok" = 1

# FunASR 字级时间戳模型（/tts 的字级时间戳用，约 160MB）
# 用纯 HTTPS 直链下载而非 modelscope snapshot_download：modelscope 的 import 链
# 会拉起 torch/oneDNN(xbyak_aarch64) 做 CPU 探测，在 QEMU 交叉构建下段错误/挂死
RUN set -eux; \
    BASE=https://modelscope.cn/models/iic/speech_timestamp_prediction-v1-16k-offline/resolve/master; \
    D=/opt/funasr-models/timestamp-v1; \
    mkdir -p "$D/example"; \
    for f in .gitattributes README.md am.mvn config.yaml configuration.json \
             example/asr_example.wav example/text.txt model.pt seg_dict tokens.json; do \
      curl -fsSL --retry 3 --retry-delay 2 -o "$D/$f" "$BASE/$f"; \
    done; \
    test "$(stat -c%s "$D/model.pt")" -gt 150000000

WORKDIR /app
COPY gateway.py gateway_state.py gateway_agg.py gateway_obs.py audio_enc.py pipeline.py npu_ops.py batched_model.py funasr_ts.py supervisor.py g2p.py compute.py worker_common.py tts_worker.py asr_worker.py proc_common.py config.py redis_transport.py heartbeat.py protocol.py ./
# 测试与运维工具随镜像分发（生产机开箱即用：离线回归 / 真机冒烟 / 显存核算）。
# pytest 为纯轮子安装，QEMU 交叉构建安全（坑仅在 import torch/thinc 的包）；
# httpx 已含在上方运行依赖（test_tts_api / benchmark 用）
COPY tests/ tests/
COPY conftest.py pytest.ini ./
# 默认值/打包一致性测试需要读取这些源文件
# （tests/test_defaults_consistency.py / test_packaging_consistency.py）；
# 随镜像分发也让运维能直接核对镜像构建配置。
COPY Dockerfile Dockerfile.cpu Dockerfile.npuprep requirements.txt ./
COPY scripts/ scripts/
RUN ln -sf /app/scripts/benchmark.py /opt/benchmark.py \
    && chmod +x /app/scripts/with_ascend_env.sh
RUN pip install --no-cache-dir pytest "fakeredis[lua]"

# 版本号烧末层：IMAGE_VERSION 在此之前不参与任何层（上面全部可缓存）——
# 改 tag（如 20261115-r6→-r7）只失效这两条元数据层，代码改动构建分钟级
ENV IMAGE_VERSION=${IMAGE_VERSION}
LABEL org.opencontainers.image.version=${IMAGE_VERSION}

EXPOSE 5174
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=5 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# GATEWAY_WORKERS=N 即单容器 N 个实例共享同一张 NPU；端口由 PORT 控制
# timeout-keep-alive 0：响应后立即回收连接，客户端下个请求重新 accept 分流——
#   多 worker 分发是按 TCP 连接而非按请求，长连接会把请求粘死在单个 worker 上，
#   设 0 等效逐请求均衡（TTS 响应大，内网握手开销可忽略）
# 多 worker 启动器：GATEWAY_WORKERS 个独立 uvicorn 进程各自以 SO_REUSEPORT
# 绑定同一端口，内核按四元组哈希均匀分发新连接（连接层统一队列/分发）——
# 突发流量下不再出现共享 accept 粘性导致的个别 worker 撞 429 闸门；
# 每个 worker 注入 WORKER_IDX，日志带 [gw-N] 标签；退出自动拉起（见 supervisor.py）
CMD ["python3", "supervisor.py"]

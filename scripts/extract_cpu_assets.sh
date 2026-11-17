#!/usr/bin/env bash
# 从现有 arm64 生产镜像（或 docker save 的 tar.gz）提取 CPU 端到端验证所需
# 模型资产：Kokoro 权重+音色（/opt/hf-cache）与 FunASR 时间戳模型
# （/opt/funasr-models）。docker cp 不执行镜像内任何程序，x86 宿主机可直接
# 从 arm64 镜像取文件。
#
# 用法：
#   bash scripts/extract_cpu_assets.sh --image <日期tag> [--out DIR] [--tar FILE]
# 必须显式指定 --image（或 IMAGE env）：不再默认 latest，避免从旧镜像取资产。
set -euo pipefail

IMAGE="${IMAGE:-}"
OUT="cpu-e2e-assets"
TAR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --image) IMAGE="${2:?}"; shift 2 ;;
        --out)   OUT="${2:?}"; shift 2 ;;
        --tar)   TAR="${2:?}"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$IMAGE" ]; then
    echo "必须用 --image 指定带模型资产的日期镜像（如 kokoro-tts-ascend:20261117-r8）" >&2
    exit 2
fi

if [ -n "$TAR" ]; then
    echo "[assets] docker load -i $TAR"
    docker load -i "$TAR"
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[assets] 镜像不存在: $IMAGE" >&2
    echo "  先 build/pull，或用 --tar 载入 docker save 归档（AGENTS.md 的导出流程）。" >&2
    exit 3
fi

mkdir -p "$OUT/hf-cache" "$OUT/funasr-models"
CID="kokoro-cpu-assets-$$"
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

# 只创建容器（不启动），即可 docker cp 取其文件系统。
docker create --name "$CID" "$IMAGE" >/dev/null
echo "[assets] 提取 /opt/hf-cache ..."
docker cp "$CID:/opt/hf-cache/." "$OUT/hf-cache/"
echo "[assets] 提取 /opt/funasr-models ..."
docker cp "$CID:/opt/funasr-models/." "$OUT/funasr-models/"

VOICE=$(find "$OUT/hf-cache" -path '*Kokoro-82M-v1.1-zh*' -name 'zf_001.pt' -print -quit || true)
MODEL="$OUT/funasr-models/timestamp-v1/model.pt"
[ -n "$VOICE" ] || { echo "[assets] 缺 Kokoro 音色 voices/zf_001.pt" >&2; exit 4; }
[ -f "$MODEL" ] || { echo "[assets] 缺 FunASR model.pt" >&2; exit 4; }
SIZE=$(stat -c%s "$MODEL")
[ "$SIZE" -gt 150000000 ] || { echo "[assets] FunASR model.pt 过小: $SIZE" >&2; exit 4; }

echo "[assets] OK"
echo "  HF cache : $OUT/hf-cache   ($(du -sh "$OUT/hf-cache" | cut -f1))"
echo "  FunASR   : $OUT/funasr-models ($(du -sh "$OUT/funasr-models" | cut -f1))"
echo "  voice    : $VOICE"
echo "  model.pt : $SIZE bytes"
echo
echo "运行 E2E："
echo "  python3 scripts/cpu_e2e.py --assets $OUT"

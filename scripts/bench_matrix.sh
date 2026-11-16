#!/usr/bin/env bash
# 配置矩阵基准 runbook——在真机 root shell 执行（建议于 /root/cui 下，结果落同目录）。
# 用法: bash bench_matrix.sh [CARD] [IMAGE]
#   CARD  默认 0（服务容器实际挂载的卡号，与现网一致）
#   IMAGE 默认 kokoro-tts-ascend:20261114-r3（pad 仪表在本版 forward done 行）
#
# 三组配置：
#   baseline_4xB8 : 4 tts × B8 + 4 asr     （现状基线）
#   A_1xB32       : 1 tts × B32 + 4 asr   （单流大批，277x 实时候选）
#   B_2xB16       : 2 tts × B16 + 4 asr   （双故障域折中）
#
# 注意：
#   - 脚本会按配置反复重启 kokoro-tts 容器——执行期间保持业务无流量
#   - 每组配置自动等 /health ready（capacity_check 不过会报 fail 并继续下一组）
#   - 每组采集：/v1 与 /tts × c=1/4/8/16 基准、forward/tts/v1 done 日志、
#     /metrics、npu-smi——全部落在 $OUT/
set -u
CARD=${1:-0}
IMAGE=${2:-kokoro-tts-ascend:20261114-r3}
NAME=kokoro-tts
OUT=bench_$(date +%m%d-%H%M)
mkdir -p "$OUT"

run_config () {
  local label=$1 workers=$2 maxb=$3 asr=$4
  echo "==== [$label] TTS_WORKERS=$workers TTS_MAX_BATCH_SIZE=$maxb ASR_WORKERS=$asr ===="
  docker rm -f $NAME >/dev/null 2>&1 || true
  docker run -d --name $NAME --restart unless-stopped \
    --shm-size 2g --log-opt max-size=50m --log-opt max-file=3 \
    -p 5174:5174 \
    -e TTS_WORKERS=$workers -e TTS_MAX_BATCH_SIZE=$maxb \
    -e ASR_WORKERS=$asr -e GATEWAY_WORKERS=2 -e ASCEND_DEVICE_ID=$CARD \
    --device /dev/davinci$CARD --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    "$IMAGE" || { echo "!! 容器启动失败：$label"; return 1; }

  local ok=0
  for _ in $(seq 1 90); do
    if curl -sf localhost:5174/health | grep -q '"ready":[ ]*true'; then ok=1; break; fi
    sleep 2
  done
  if [ "$ok" -ne 1 ]; then
    echo "!! health 180s 未就绪（capacity_check 可能拒绝该配置）：$label"
    docker logs $NAME 2>&1 | tail -30 | tee "$OUT/${label}_fail.txt"
    return 1
  fi

  # 全舰队就绪：/health ready=true 只代表"每角色 ≥1 ready"（gateway.py:860），
  # 压测必须等满编（kokoro_worker_ready 计数 == 配置的 worker 数），否则
  # 基线会在半编状态被测、三组对比失真。
  local rt=0 ra=0
  for _ in $(seq 1 120); do
    local m
    m=$(curl -s localhost:5174/metrics)
    # prometheus_client 输出浮点值（1.0）——awk 对 1/1.0 任何格式免疫
    rt=$(echo "$m" | awk '/^kokoro_worker_ready\{.*role="tts"/ {s += $NF} END {printf "%d", s + 0}')
    ra=$(echo "$m" | awk '/^kokoro_worker_ready\{.*role="asr"/ {s += $NF} END {printf "%d", s + 0}')
    if [ "$rt" -ge "$workers" ] && [ "$ra" -ge "$asr" ]; then ok=2; break; fi
    sleep 2
  done
  if [ "$ok" -lt 2 ]; then
    echo "!! 全舰队未就绪（tts ${rt:-0}/$workers asr ${ra:-0}/$asr）：$label"
    docker logs $NAME 2>&1 | tail -30 | tee "$OUT/${label}_fail.txt"
    return 1
  fi
  sleep 5   # 心跳/指标稳定缓冲
  echo "[$label] 全舰队就绪：tts $rt/$workers asr $ra/$asr"

  for ep in speech tts; do
    for c in 1 4 8 16; do
      echo "---- $label $ep c=$c ----"
      docker exec $NAME python3 /app/scripts/benchmark.py \
        --url http://127.0.0.1:5174 --endpoint $ep \
        -c $c -n 32 --random 64 --voice zf_001 \
        | tee "$OUT/${label}_${ep}_c${c}.txt"
    done
  done

  docker logs $NAME 2>&1 | grep -E "req=|chunk=|bid=" \
    | tail -600 > "$OUT/${label}_logs.txt"
  curl -s localhost:5174/metrics > "$OUT/${label}_metrics.txt"
  npu-smi info > "$OUT/${label}_smi.txt" 2>/dev/null || true
  echo "[$label] 完成 → $OUT/${label}_*"
}

run_config baseline_4xB8 4 8  4
run_config A_1xB32        1 32 4
run_config B_2xB16        2 16 4

echo
echo "全部完成。结果目录：$OUT/（打包回传：tar czf $OUT.tar.gz $OUT）"

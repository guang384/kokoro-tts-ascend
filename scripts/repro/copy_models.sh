#!/bin/sh
# 从镜像拷出本地复现所需权重（kokoro 三件套 + FunASR timestamp-v1 模型文件）
set -e
SNAP=/opt/funasr-models/timestamp-v1
KF=/opt/hf-cache/hub/models--hexgrad--Kokoro-82M-v1.1-zh/snapshots
cp -L "$SNAP/model.pt" "$SNAP/config.yaml" "$SNAP/configuration.json" \
      "$SNAP/tokens.json" "$SNAP/seg_dict" "$SNAP/am.mvn" /out/
cp -L "$KF"/*/config.json /out/kokoro_cfg.json
cp -L "$KF"/*/kokoro-v1_1-zh.pth /out/kokoro.pth
cp -L "$KF"/*/voices/zf_001.pt /out/zf_001.pt
ls -la /out/

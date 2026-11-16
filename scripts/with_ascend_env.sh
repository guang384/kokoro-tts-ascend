#!/usr/bin/env bash
# 复刻镜像 ENTRYPOINT 的 CANN 环境，供 docker exec 等不经过 ENTRYPOINT 的命令使用。
# 用法：bash scripts/with_ascend_env.sh python tests/validate_asr_batch.py
set -e
export _KOKORO_CANN_REEXEC=1
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh ]; then
    source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
fi
exec "$@"

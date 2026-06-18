#!/usr/bin/env bash
set -euo pipefail

ACCELSIM_ROOT="/workspace/accel-sim-framework"

cd "${ACCELSIM_ROOT}"

export OPENCL_REMOTE_GPU_HOST="${OPENCL_REMOTE_GPU_HOST:-}"
source ./gpu-simulator/setup_environment.sh

python3 ./util/tracer_nvbit/run_hw_trace.py \
    -B synthetic_large3_n4 \
    -D 0

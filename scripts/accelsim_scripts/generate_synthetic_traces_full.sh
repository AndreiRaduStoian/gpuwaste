#!/usr/bin/env bash
set -euo pipefail

# Paths
ACCELSIM_ROOT="/workspace/accel-sim-framework"
BENCH_DIR="/workspace/gpuwaste/benchmarks/cu"
EXE="${BENCH_DIR}/synthetic_kernels_dc"
TRACER_SO="${ACCELSIM_ROOT}/util/tracer_nvbit/tracer_tool/tracer_tool.so"
POSTPROC="${ACCELSIM_ROOT}/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"

# Output location
OUT_ROOT="/workspace/gpuwaste/accelsim_traces/synthetic/12.9"

# CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: include CUDA source line info in traces if binary was built with -lineinfo.
export TRACE_LINEINFO=0

# Check inputs
if [[ ! -x "${EXE}" ]]; then
    echo "ERROR: executable not found or not executable: ${EXE}"
    echo "Compile it first, e.g.:"
    echo "  cd ${BENCH_DIR}"
    echo "  nvcc -O3 -lineinfo -arch=sm_86 synthetic_kernels.cu -o synthetic_kernels_dc"
    exit 1
fi

if [[ ! -f "${TRACER_SO}" ]]; then
    echo "ERROR: tracer shared library not found: ${TRACER_SO}"
    exit 1
fi

if [[ ! -x "${POSTPROC}" ]]; then
    echo "ERROR: post-processing tool not found or not executable: ${POSTPROC}"
    exit 1
fi

mkdir -p "${OUT_ROOT}"

run_trace() {
    local kernel="$1"
    local n="$2"
    local iterations="$3"
    local block_size="$4"
    local name_filter="$5"

    local tag="N${n}_I${iterations}_B${block_size}"
    local out_dir="${OUT_ROOT}/${kernel}/${tag}"

    echo
    echo "============================================================"
    echo "Tracing kernel: ${kernel}"
    echo "N=${n}, iterations=${iterations}, block_size=${block_size}"
    echo "Output: ${out_dir}"
    echo "Kernel filter: ${name_filter}"
    echo "============================================================"

    cd "${BENCH_DIR}"

    # Remove old local trace directory produced by NVBit.
    rm -rf traces

    # Trace only kernels whose name matches the expected synthetic kernel name.
    # This avoids tracing CUDA runtime/helper kernels such as memset kernels.
    export DYNAMIC_KERNEL_RANGE="2@${name_filter}"

    LD_PRELOAD="${TRACER_SO}" \
        "${EXE}" "${kernel}" "${n}" "${iterations}" "${block_size}"

    if [[ ! -d traces ]]; then
        echo "ERROR: tracer did not produce a traces directory for ${kernel}"
        exit 1
    fi

    local kernelslist
    kernelslist="$(find traces -maxdepth 1 -name 'kernelslist_ctx_*' | head -n 1 || true)"

    if [[ -z "${kernelslist}" ]]; then
        echo "ERROR: no kernelslist_ctx_* file found for ${kernel}"
        echo "Files in traces/:"
        find traces -maxdepth 2 -type f
        exit 1
    fi

    echo "Post-processing ${kernelslist}"
    "${POSTPROC}" "${kernelslist}"

    if [[ ! -f traces/kernelslist.g ]]; then
        echo "ERROR: post-processing did not produce traces/kernelslist.g for ${kernel}"
        echo "Files in traces/:"
        find traces -maxdepth 2 -type f
        exit 1
    fi

    local traceg_count
    traceg_count="$(find traces -maxdepth 1 \( -name '*.traceg' -o -name '*.traceg.xz' \) | wc -l)"

    if [[ "${traceg_count}" -eq 0 ]]; then
        echo "ERROR: no .traceg files produced for ${kernel}"
        exit 1
    fi

    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
    mv traces "${out_dir}/"

    echo "Saved processed trace:"
    echo "  ${out_dir}/traces/kernelslist.g"
    echo "Traceg files:"
    find "${out_dir}/traces" -maxdepth 1 \( -name '*.traceg' -o -name '*.traceg.xz' \) -print

    echo "Tracer stats:"
    find "${out_dir}/traces" -maxdepth 1 -name 'stats_ctx_*' -exec cat {} \;
}

# Kernel filters are regexes matched against the CUDA kernel name.
# They are intentionally broad enough to match mangled CUDA names.

run_trace "vectoradd" 67108864 1    256 ".*vector.*"
run_trace "compute"   16777216 1000 256 ".*compute.*,.*fma.*"
run_trace "pointer"   16777216 100  256 ".*pointer.*,.*chase.*"
run_trace "shared"    16777216 100  256 ".*shared.*,.*barrier.*"
run_trace "sfu"       16777216 100  256 ".*sfu.*"

echo
echo "All traces generated under:"
echo "  ${OUT_ROOT}"

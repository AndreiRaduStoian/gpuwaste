#!/usr/bin/env bash
set -euo pipefail

ACCELSIM_ROOT="/workspace/accel-sim-framework"
TRACE_ROOT="/workspace/gpuwaste/accelsim_traces/synthetic_large3_n2/12.9/"
BENCHMARK="synthetic_large3_n2"
BASE_CONFIG="RTX3070-SASS"
SIM_BIN_DIR="${ACCELSIM_ROOT}/gpu-simulator/bin/release"

PARAM="${1:-}"

if [[ -z "${PARAM}" ]]; then
    echo "Usage: $0 <sm|sp|sfu|core|mem>"
    exit 1
fi

case "${PARAM}" in
    sm)   PREFIX="SM" ;;
    sp)   PREFIX="SP" ;;
    sfu)  PREFIX="SFU" ;;
    core) PREFIX="CORE" ;;
    mem)  PREFIX="MEM" ;;
    *)
        echo "ERROR: unknown parameter '${PARAM}'"
        echo "Use one of: sm sp sfu core mem"
        exit 1
        ;;
esac

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="${PARAM}_n2_${TIMESTAMP}"
OUT_ROOT="/workspace/gpuwaste/results/accelsim_${RUN_TAG}"

mkdir -p "${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}/stats"

MANIFEST="${OUT_ROOT}/manifest.csv"
echo "run_name,config,param,scale" > "${MANIFEST}"

cd "${ACCELSIM_ROOT}"

set +u
source ./gpu-simulator/setup_environment.sh
set -u

wait_for_run() {
    local run_name="$1"
    local status_log="${OUT_ROOT}/logs/${run_name}_status.log"

    echo "Waiting for ${run_name}..."

    while true; do
        ./util/job_launching/job_status.py -N "${run_name}" > "${status_log}" 2>&1 || true

        cat "${status_log}"

        if grep -q "FAILED\|ERROR\|Failed/Error:[1-9]" "${status_log}"; then
            echo "ERROR: ${run_name} has failed jobs."
            exit 1
        fi

        # synthetic_large3 has 3 jobs. When all three are COMPLETE_NO_OTHER_INFO,
        # the run is done.
        complete_count="$(grep -c "COMPLETE_NO_OTHER_INFO" "${status_log}" || true)"
        running_count="$(grep -c "RUNNING" "${status_log}" || true)"
        waiting_count="$(grep -c "WAITING" "${status_log}" || true)"

        echo "Status counts: complete=${complete_count}, running=${running_count}, waiting=${waiting_count}"

        if [[ "${complete_count}" -ge 3 && "${running_count}" -eq 0 && "${waiting_count}" -eq 0 ]]; then
            echo "${run_name} finished."
            break
        fi

        sleep 5
    done
}

run_one() {
    local scale="$1"
    local suffix="$2"

    local config="${BASE_CONFIG}${suffix}"
    local scale_tag="${scale//./}"
    local run_name="sweep_${RUN_TAG}_${PARAM}${scale_tag}"

    echo
    echo "============================================================"
    echo "Running ${run_name}"
    echo "Config: ${config}"
    echo "Scale: ${scale}"
    echo "============================================================"

    echo "${run_name},${config},${PARAM},${scale}" >> "${MANIFEST}"

    ./util/job_launching/run_simulations.py \
        -B "${BENCHMARK}" \
        -C "${config}" \
        -T "${TRACE_ROOT}" \
        -s "${SIM_BIN_DIR}" \
        -N "${run_name}" \
        > "${OUT_ROOT}/logs/${run_name}_launch.log" 2>&1

    echo "Launch log:"
    cat "${OUT_ROOT}/logs/${run_name}_launch.log"

    wait_for_run "${run_name}"

    ./util/job_launching/get_stats.py -N "${run_name}" \
        > "${OUT_ROOT}/stats/${run_name}_stats.csv"

    echo "Saved stats:"
    echo "  ${OUT_ROOT}/stats/${run_name}_stats.csv"
}

run_one "1.0" ""
run_one "0.75" "-${PREFIX}075"
run_one "0.5" "-${PREFIX}050"
run_one "0.25" "-${PREFIX}025"

echo
echo "============================================================"
echo "Finished ${PARAM} sweep."
echo "Results saved in:"
echo "  ${OUT_ROOT}"
echo "Manifest:"
echo "  ${MANIFEST}"
echo "============================================================"

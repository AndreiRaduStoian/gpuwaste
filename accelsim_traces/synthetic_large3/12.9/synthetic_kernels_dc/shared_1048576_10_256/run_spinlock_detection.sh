set -e

export DYNAMIC_KERNEL_RANGE=""

export CUDA_VERSION="12.9"; export CUDA_VISIBLE_DEVICES="0" ; 
rm -f spinlock_detection/*
export TRACES_FOLDER=/workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_1048576_10_256; SPINLOCK_PHASE=0 CUDA_INJECTION64_PATH=/workspace/accel-sim-framework/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so /workspace/gpuwaste/benchmarks/cu/synthetic_kernels_dc shared 1048576 10 256 ;  SPINLOCK_PHASE=1 CUDA_INJECTION64_PATH=/workspace/accel-sim-framework/util/tracer_nvbit/others/spinlock_tool/spinlock_tool.so /workspace/gpuwaste/benchmarks/cu/synthetic_kernels_dc shared 1048576 10 256 ; 
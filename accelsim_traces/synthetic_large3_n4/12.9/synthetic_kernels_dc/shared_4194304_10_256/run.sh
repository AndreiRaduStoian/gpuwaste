set -e

export DYNAMIC_KERNEL_RANGE=""

export CUDA_VERSION="12.9"; export CUDA_VISIBLE_DEVICES="0" ; 
rm -f traces/*
export TRACES_FOLDER=/workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_4194304_10_256; ENABLE_SPINLOCK_FAST_FORWARD=0 SPINLOCK_ITER_TO_KEEP=1 CUDA_INJECTION64_PATH=/workspace/accel-sim-framework/util/tracer_nvbit/tracer_tool/tracer_tool.so /workspace/gpuwaste/benchmarks/cu/synthetic_kernels_dc shared 4194304 10 256 ; /workspace/accel-sim-framework/util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing /workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_4194304_10_256/traces ; rm -f /workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_4194304_10_256/traces/*.trace ; rm -f /workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_4194304_10_256/traces/*.trace.xz ; rm -f /workspace/accel-sim-framework/hw_run/traces/device-0/12.9/synthetic_kernels_dc/shared_4194304_10_256/traces/kernelslist 
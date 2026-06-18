from roofline import (
    MachineConfig,
    scale_core_frequency,
    scale_memory_bandwidth,
    scale_sm_count,
    plot_rooflines,
    predict_kernel_runtime_seconds,
    make_kernel_point,
)


RTX_3070 = MachineConfig(
    name="RTX 3070",
    # from NVIDIA specs
    peak_ops_per_sec=20.31e12,
    memory_bandwidth_bytes_per_sec=448e9,
)


vectoradd = make_kernel_point(
    name="vectoradd",
    total_ops=67_108_864,
    total_bytes=805_306_368,
    runtime_ms=1.74864,
)

compute = make_kernel_point(
    name="compute",
    total_ops=100_696_932_032,
    total_bytes=67_108_864,
    runtime_ms=7.092224,
)

pointer = make_kernel_point(
    name="pointer",
    total_ops=16_777_216 * 100,
    total_bytes=(16_777_216 * 100 * 4) + (16_777_216 * 4),
    runtime_ms=1.19808,
)

shared = make_kernel_point(
    name="shared",
    total_ops=16_777_216 * 100 * 5,
    total_bytes=16_777_216 * 8,
    runtime_ms=4.903712,
)

sfu = make_kernel_point(
    name="sfu",
    total_ops=3_355_443_200,      # only normal FP32 mul/add ops
    total_bytes=67_108_864,
    runtime_ms=8.600416,
)

kernels = [
    vectoradd,
    compute,
    pointer,
    shared,
    sfu,
]


machines = [
    RTX_3070,
    # scale_core_frequency(RTX_3070, 0.5, name="RTX 3070 core x0.5"),
    # scale_memory_bandwidth(RTX_3070, 0.5, name="RTX 3070 mem x0.5"),
    # scale_sm_count(RTX_3070, 0.5, name="RTX 3070 SM x0.5"),
]


plot_rooflines(
    machines=machines,
    kernels=kernels,
    title="Synthetic kernels on Roofline model",
    output_file="synthetic_roofline.png",
)


for kernel_name, ops, bytes_moved, runtime_ms in [
    ("vectoradd", 67_108_864, 805_306_368, 1.74864),
    ("compute", 100_696_932_032, 67_108_864, 7.092224),
    ("pointer", 16_777_216 * 100 * 4, (16_777_216 * 100 * 4) + (16_777_216 * 4), 1.19808),
    ("shared", 16_777_216 * 100 * 5, 16_777_216 * 8, 4.903712),
    ("sfu", 3_355_443_200, 67_108_864, 8.600416),
]:
    predicted_s = predict_kernel_runtime_seconds(
        RTX_3070,
        total_ops=ops,
        total_bytes=bytes_moved,
    )

    print(kernel_name)
    print(f"  measured runtime:  {runtime_ms:.6f} ms")
    print(f"  roofline runtime:  {predicted_s * 1000:.6f} ms")
    print(f"  measured / model:  {runtime_ms / (predicted_s * 1000):.2f}x")
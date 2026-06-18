from roofline import (
    MachineConfig,
    scale_core_frequency,
    scale_memory_bandwidth,
    scale_sm_count,
    predict_kernel_runtime_seconds,
)

rtx3070 = MachineConfig(
    name="RTX 3070",
    peak_ops_per_sec=20.3e12,          # FP32 ops/s, approx RTX 3070
    memory_bandwidth_bytes_per_sec=448e9,
)

kernels = {
    "vectoradd": {
        "ops": 67_108_864,
        "bytes": 805_306_368,
        "measured_ms": 1.748640,
    },
    "compute": {
        "ops": 100_696_932_032,
        "bytes": 67_108_864,
        "measured_ms": 7.092224,
    },
    "pointer": {
        "ops": 16_777_216 * 100,
        "bytes": (16_777_216 * 100 * 4) + (16_777_216 * 4),
        "measured_ms": 1.198080,
    },
    "shared": {
        "ops": 16_777_216 * 100 * 5,
        "bytes": 16_777_216 * 8,
        "measured_ms": 4.903712,
    },
    "sfu": {
        "ops": 3_355_443_200,
        "bytes": 67_108_864,
        "measured_ms": 8.600416,
    },
}

scales = [1.0, 0.75, 0.5, 0.25]

scenarios = {
    "compute_scaling": lambda s: scale_core_frequency(
        rtx3070, s, name=f"compute x{s}"
    ),
    "memory_scaling": lambda s: scale_memory_bandwidth(
        rtx3070, s, name=f"memory x{s}"
    ),
    "combined_scaling": lambda s: MachineConfig(
        name=f"combined x{s}",
        peak_ops_per_sec=rtx3070.peak_ops_per_sec * s,
        memory_bandwidth_bytes_per_sec=rtx3070.memory_bandwidth_bytes_per_sec * s,
    ),
}

for scenario_name, make_machine in scenarios.items():
    print()
    print(scenario_name)
    print("kernel,scale,predicted_ms,slowdown")

    baseline_predictions = {}

    for kernel_name, k in kernels.items():
        baseline_s = predict_kernel_runtime_seconds(
            rtx3070,
            total_ops=k["ops"],
            total_bytes=k["bytes"],
        )
        baseline_predictions[kernel_name] = baseline_s

    for kernel_name, k in kernels.items():
        for scale in scales:
            machine = make_machine(scale)

            predicted_s = predict_kernel_runtime_seconds(
                machine,
                total_ops=k["ops"],
                total_bytes=k["bytes"],
            )

            slowdown = predicted_s / baseline_predictions[kernel_name]

            print(
                f"{kernel_name},{scale},{predicted_s * 1000:.6f},{slowdown:.2f}"
            )
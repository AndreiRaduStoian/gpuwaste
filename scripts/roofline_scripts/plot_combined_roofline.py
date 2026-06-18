from roofline import (
    MachineConfig,
    make_kernel_point,
    scale_core_frequency,
    scale_memory_bandwidth,
    plot_rooflines,
)


# RTX 3070 approximate specs
# FP32 peak: ~20.3 TFLOP/s
# Memory bandwidth: ~448 GB/s
BASE_MACHINE = MachineConfig(
    name="RTX 3070 baseline",
    peak_ops_per_sec=20.3e12,
    memory_bandwidth_bytes_per_sec=448e9,
)


kernels = [
    make_kernel_point(
        name="vectoradd",
        total_ops=67_108_864,
        total_bytes=805_306_368,
        runtime_ms=1.74864,
    ),
    make_kernel_point(
        name="compute",
        total_ops=100_696_932_032,
        total_bytes=67_108_864,
        runtime_ms=7.092224,
    ),
    make_kernel_point(
        name="pointer",
        total_ops=16_777_216 * 100,
        total_bytes=(16_777_216 * 100 * 4) + (16_777_216 * 4),
        runtime_ms=1.19808,
    ),
    make_kernel_point(
        name="shared",
        total_ops=16_777_216 * 100 * 5,
        total_bytes=16_777_216 * 8,
        runtime_ms=4.903712,
    ),
    make_kernel_point(
        name="sfu",
        total_ops=3_355_443_200,
        total_bytes=67_108_864,
        runtime_ms=8.600416,
    ),
]


SCALES = [1.0, 0.75, 0.5, 0.25]


def make_compute_scaled_machines():
    return [
        scale_core_frequency(
            BASE_MACHINE,
            scale=s,
            name=f"Compute x{s}",
        )
        for s in SCALES
    ]


def make_memory_scaled_machines():
    return [
        scale_memory_bandwidth(
            BASE_MACHINE,
            scale=s,
            name=f"Memory BW x{s}",
        )
        for s in SCALES
    ]


def make_combined_scaled_machines():
    machines = []

    for s in SCALES:
        machine = MachineConfig(
            name=f"Compute+Memory x{s}",
            peak_ops_per_sec=BASE_MACHINE.peak_ops_per_sec * s,
            memory_bandwidth_bytes_per_sec=BASE_MACHINE.memory_bandwidth_bytes_per_sec * s,
        )
        machines.append(machine)

    return machines


def main():
    plot_rooflines(
        machines=make_compute_scaled_machines(),
        kernels=kernels,
        title="Roofline under compute throughput scaling",
        output_file="roofline_compute_scaling.png",
    )

    plot_rooflines(
        machines=make_memory_scaled_machines(),
        kernels=kernels,
        title="Roofline under memory bandwidth scaling",
        output_file="roofline_memory_scaling.png",
    )

    plot_rooflines(
        machines=make_combined_scaled_machines(),
        kernels=kernels,
        title="Roofline under combined compute and memory scaling",
        output_file="roofline_combined_scaling.png",
    )


if __name__ == "__main__":
    main()
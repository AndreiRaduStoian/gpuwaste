from roofline import (
    MachineConfig,
    KernelPoint,
    scale_core_frequency,
    scale_memory_bandwidth,
    scale_sm_count,
    plot_rooflines,
    predict_kernel_runtime_seconds,
)

# placeholders
base = MachineConfig(
    name="Base GPU",
    peak_ops_per_sec=20e12,
    memory_bandwidth_bytes_per_sec=448e9
)

machines = [
    base,
    scale_core_frequency(base, 0.5, name="Core freq 50%"),
    scale_memory_bandwidth(base, 0.5, name="Memory BW 50%"),
    scale_sm_count(base, 0.5, name="SM count 50%"),
]

kernels = [
    KernelPoint("VectorAdd", operational_intensity=1 / 12),
    KernelPoint("BFS", operational_intensity=0.05),
    KernelPoint("Hotspot", operational_intensity=0.5),
    KernelPoint("LUD", operational_intensity=4.0),
    KernelPoint("ComputeLoop", operational_intensity=100.0),
]

plot_rooflines(
    machines=machines,
    kernels=kernels,
    title="Roofline shrinking scenarios",
    output_file="roofline_shrinking.png",
)

runtime = predict_kernel_runtime_seconds(
    machine=base,
    total_ops=1e12,
    total_bytes=200e9,
)

print(f"Predicted runtime: {runtime:.6f} s")

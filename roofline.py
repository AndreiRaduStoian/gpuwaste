# roofline.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MachineConfig:
    name: str

    peak_ops_per_sec: float

    memory_bandwidth_bytes_per_sec: float


@dataclass(frozen=True)
class KernelPoint:
    name: str

    operational_intensity: float

    measured_ops_per_sec: Optional[float] = None


def roofline_performance(machine: MachineConfig,operational_intensity: np.ndarray) -> np.ndarray:
    memory_bound = machine.memory_bandwidth_bytes_per_sec * operational_intensity
    compute_bound = machine.peak_ops_per_sec
    return np.minimum(compute_bound, memory_bound)


def ridge_point(machine: MachineConfig) -> float:
    """
    Operational intensity needed to become compute-bound.
    """
    return machine.peak_ops_per_sec / machine.memory_bandwidth_bytes_per_sec


def scale_core_frequency(machine: MachineConfig, scale: float, name: Optional[str] = None) -> MachineConfig:

    return MachineConfig(
        name=name or f"{machine.name} core x{scale}",
        peak_ops_per_sec=machine.peak_ops_per_sec * scale,
        memory_bandwidth_bytes_per_sec=machine.memory_bandwidth_bytes_per_sec,
    )


def scale_memory_bandwidth(
    machine: MachineConfig,
    scale: float,
    name: Optional[str] = None,
) -> MachineConfig:
    """
    Memory bandwidth scaling changes the sloped memory roof.
    Compute peak is kept fixed.
    """
    return MachineConfig(
        name=name or f"{machine.name} mem x{scale}",
        peak_ops_per_sec=machine.peak_ops_per_sec,
        memory_bandwidth_bytes_per_sec=machine.memory_bandwidth_bytes_per_sec * scale,
    )


def scale_sm_count(
    machine: MachineConfig,
    scale: float,
    memory_scales_too: bool = False,
    name: Optional[str] = None,
) -> MachineConfig:

    mem_scale = scale if memory_scales_too else 1.0

    return MachineConfig(
        name=name or f"{machine.name} SM x{scale}",
        peak_ops_per_sec=machine.peak_ops_per_sec * scale,
        memory_bandwidth_bytes_per_sec=machine.memory_bandwidth_bytes_per_sec * mem_scale,
    )


def plot_rooflines(
    machines: List[MachineConfig],
    kernels: Optional[List[KernelPoint]] = None,
    title: str = "Roofline Model",
    output_file: Optional[str] = None,
):
    min_oi = 1e-3
    max_oi = 1e3

    if kernels:
        kernel_ois = [k.operational_intensity for k in kernels]
        min_oi = min(min_oi, min(kernel_ois) / 10)
        max_oi = max(max_oi, max(kernel_ois) * 10)

    x = np.logspace(np.log10(min_oi), np.log10(max_oi), 500)

    plt.figure(figsize=(8, 6))

    for machine in machines:
        y = roofline_performance(machine, x)
        plt.plot(x, y, label=f"{machine.name}")

        rp = ridge_point(machine)
        plt.axvline(rp, linestyle="--", linewidth=0.8)

    if kernels:
        for kernel in kernels:
            if kernel.measured_ops_per_sec is not None:
                y = kernel.measured_ops_per_sec
            else:
                y = roofline_performance(
                    machines[0],
                    np.array([kernel.operational_intensity]),
                )[0]

            plt.scatter(kernel.operational_intensity, y)
            plt.text(
                kernel.operational_intensity,
                y,
                f"  {kernel.name}",
                fontsize=9,
                verticalalignment="center",
            )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Operational intensity [ops / byte]")
    plt.ylabel("Performance [ops / second]")
    plt.title(title)
    plt.grid(True, which="both", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=200)

    plt.show()


def predict_kernel_runtime_seconds(
    machine: MachineConfig,
    total_ops: float,
    total_bytes: float,
) -> float:

    if total_bytes <= 0:
        operational_intensity = float("inf")
        perf = machine.peak_ops_per_sec
    else:
        operational_intensity = total_ops / total_bytes
        perf = min(
            machine.peak_ops_per_sec,
            machine.memory_bandwidth_bytes_per_sec * operational_intensity,
        )

    return total_ops / perf




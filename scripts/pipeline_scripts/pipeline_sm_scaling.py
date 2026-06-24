from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from hardware import RTX_3070_CALIBRATION_HARDWARE, CoreHardwareConfig, InstructionTiming
from pipelinesim import PipelineSimulator
from schedulers import RRScheduler
from software import (
    cuda_launch_to_pipeline_config,
    make_compute_fma_idg,
    make_vector_add_idg,
    make_pointer_chase_idg,
    make_shared_barrier_idg,
    make_sfu_idg,
)

OUT = Path("pipeline_sm_throughput_results")
OUT.mkdir(exist_ok=True)

N = 2_097_152
ITERATIONS = 10
BLOCK_SIZE = 256
SM_COUNT = 46
RESIDENT_BLOCKS_PER_SM = 6
CLOCK_CYCLES_PER_US = 1815.0

KERNELS = ["compute", "sfu"]
FAMILIES = {
    "SP": ("fma_f32", "add_f32"),
    "SFU": ("sfu",),
}
SCALES = [1.0, 0.5, 0.25]

LINESTYLES = {
    "compute": "-",
    "vectoradd": "--",
    "pointer": ":",
    "shared": "-.",
    "sfu": (0, (3, 1, 1, 1)),
}
MARKERS = {
    "compute": "o",
    "vectoradd": "s",
    "pointer": "^",
    "shared": "D",
    "sfu": "x",
}


def round_to_issue_quantum(x, issue_limit_ipc):
    q = 1.0 / issue_limit_ipc
    return max(q, round(x / q) * q)


def scaled_hardware(family, scale):
    base = RTX_3070_CALIBRATION_HARDWARE
    scaled_ops = FAMILIES[family]
    timings = {}

    for op, t in base.timings.items():
        lam = t.lambda_cpi
        if op in scaled_ops:
            lam = round_to_issue_quantum(lam / scale, base.issue_limit_ipc)
        timings[op] = InstructionTiming(t.subsystem, lam, t.completion_latency)

    return CoreHardwareConfig(base.subsystems, base.issue_limit_ipc, timings)


def make_idg(kernel):
    if kernel == "compute":
        return make_compute_fma_idg(ITERATIONS)
    if kernel == "vectoradd":
        return make_vector_add_idg(ITERATIONS)
    if kernel == "pointer":
        return make_pointer_chase_idg(ITERATIONS)
    if kernel == "shared":
        return make_shared_barrier_idg(ITERATIONS)
    if kernel == "sfu":
        return make_sfu_idg(ITERATIONS)



def run_one(kernel, hw):
    grid_size, exe = cuda_launch_to_pipeline_config(
        n=N,
        block_size=BLOCK_SIZE,
        resident_blocks_per_sm=RESIDENT_BLOCKS_PER_SM,
    )
    idg = make_idg(kernel)
    result = PipelineSimulator(hw, scheduler=RRScheduler()).run_idg(kernel, idg, exe, tracing=False)

    waves = math.ceil(grid_size / (SM_COUNT * RESIDENT_BLOCKS_PER_SM))
    gpu_cycles = result.cycles * waves
    runtime_ms = (gpu_cycles / CLOCK_CYCLES_PER_US) / 1000.0

    return grid_size, waves, result.cycles, gpu_cycles, runtime_ms


def write_tex(rows):
    path = OUT / "proportional_slowdown_table.tex"
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Kernel & SP 0.50 & SP 0.25 & SFU 0.50 & SFU 0.25 \\\\\n")
        f.write("\\midrule\n")
        for kernel in KERNELS:
            vals = []
            for family, scale in [("SP", 0.5), ("SP", 0.25), ("SFU", 0.5), ("SFU", 0.25)]:
                x = [r for r in rows if r["kernel"] == kernel and r["family"] == family and r["scale"] == scale][0]
                vals.append(x["proportional_slowdown"])
            f.write(f"{kernel} & {vals[0]:.2f} & {vals[1]:.2f} & {vals[2]:.2f} & {vals[3]:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print("Saved:", path)


def plot_slowdowns(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)

    for ax, family in zip(axes, ["SP", "SFU"]):
        for kernel in KERNELS:
            kr = sorted(
                [r for r in rows if r["family"] == family and r["kernel"] == kernel],
                key=lambda r: -r["scale"],
            )
            ax.plot(
                [r["scale"] for r in kr],
                [r["slowdown"] for r in kr],
                marker=MARKERS[kernel],
                linestyle=LINESTYLES[kernel],
                linewidth=2.0,
                markersize=5,
                label=kernel,
            )

        ax.plot(SCALES, [1.0 / x for x in SCALES], color="black", linestyle="--", linewidth=1.0, label="ideal bound")
        ax.set_title(f"{family} throughput scaling")
        ax.set_xlabel("throughput scale")
        ax.set_xticks(SCALES)
        ax.set_xlim(1.05, 0.20)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("runtime slowdown")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0.13, 1, 1])

    path = OUT / "sm_throughput_slowdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def main():
    rows = []
    base_runtime = {}

    for family in FAMILIES:
        for scale in SCALES:
            hw = scaled_hardware(family, scale)
            for kernel in KERNELS:
                grid_size, waves, sm_cycles, gpu_cycles, runtime_ms = run_one(kernel, hw)

                if scale == 1.0:
                    base_runtime[(family, kernel)] = runtime_ms

                slowdown = runtime_ms / base_runtime[(family, kernel)]
                prop = scale * slowdown

                rows.append({
                    "family": family,
                    "kernel": kernel,
                    "n": N,
                    "iterations": ITERATIONS,
                    "block_size": BLOCK_SIZE,
                    "scale": scale,
                    "grid_size": grid_size,
                    "waves": waves,
                    "single_sm_wave_cycles": sm_cycles,
                    "gpu_cycles": gpu_cycles,
                    "runtime_ms": runtime_ms,
                    "slowdown": slowdown,
                    "proportional_slowdown": prop,
                })

    csv_path = OUT / "sm_throughput_scaling.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved:", csv_path)

    write_tex(rows)
    plot_slowdowns(rows)


if __name__ == "__main__":
    main()

from pipelinesim import PipelineSimulator
from schedulers import RRScheduler
from software import make_fig1_toy_idg, make_iterative_barrier_idg
from hardware import ExecutionConfig
from hardware import PAPER_TOY_HARDWARE, FERMI_BARRIER_HARDWARE, PASCAL_BARRIER_HARDWARE

import math
import csv


def write_results_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "hardware_threads_per_group", "concurrent_groups_per_core", "occupancy_hardware_threads", "instructions_per_hardware_thread", "cycles", "hardware_threads_per_cycle", "instructions_per_cycle"])
        for r in results:
            e = r.execution_config
            writer.writerow([r.kernel_name, e.hardware_threads_per_group, e.concurrent_groups_per_core, e.occupancy_hardware_threads, r.instruction_count_per_hardware_thread, r.cycles, r.hardware_threads_per_cycle, r.instructions_per_cycle])


def write_trace_csv(result, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "hardware_thread_id", "instr_id", "op", "subsystem", "issue_time", "complete_time", "deps", "raw"])
        for event in result.trace:
            writer.writerow([event.group_id, event.hardware_thread_id, event.instr_id, event.op, event.subsystem, event.issue_time, event.complete_time, " ".join(event.deps), event.raw])


def print_barrier_sweep(rows):
    if not rows:
        return

    base = rows[0]["warps_per_cycle"]

    header = (
        f"{'gpu':<8} {'wg_size':>8} {'w/g':>5} {'groups':>6} "
        f"{'occ':>5} {'cycles':>12} {'WPC':>10} {'norm':>10}"
    )

    print(header)
    print("-" * len(header))

    for r in rows:
        norm = 100.0 * r["warps_per_cycle"] / base if base else math.inf

        print(
            f"{r['gpu']:<8} "
            f"{r['work_group_size']:>8} "
            f"{r['warps_per_group']:>5} "
            f"{r['groups']:>6} "
            f"{r['occupancy']:>5} "
            f"{r['cycles']:>12.2f} "
            f"{r['warps_per_cycle']:>10.5f} "
            f"{norm:>10.2f}"
        )

def print_results(results):
    header = f"{'kernel':<24} {'ht/g':>5} {'groups':>6} {'occ':>5} {'instr/ht':>8} {'cycles':>10} {'HT/cyc':>10} {'instr/cyc':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        e = r.execution_config
        print(f"{r.kernel_name:<24} {e.hardware_threads_per_group:>5} {e.concurrent_groups_per_core:>6} {e.occupancy_hardware_threads:>5} {r.instruction_count_per_hardware_thread:>8} {r.cycles:>10.2f} {r.hardware_threads_per_cycle:>10.4f} {r.instructions_per_cycle:>10.4f}")

# fig1 test
def sweep_occupancy(kernel_name, idg, hardware, hardware_threads_per_group_values, concurrent_groups_per_core_values, scheduler_factory=RRScheduler):
    results = []
    for htg in hardware_threads_per_group_values:
        for groups in concurrent_groups_per_core_values:
            exe = ExecutionConfig(htg, groups)
            sim = PipelineSimulator(hardware, scheduler=scheduler_factory())
            results.append(sim.run_idg(kernel_name, idg, exe))
    return results

# fig16 test
def sweep_iterative_barrier_sampled(
    gpu_name,
    hardware,
    max_occupancy,
    iterations=256,
    work_group_sizes=(32, 64, 128, 256, 512, 1024),
    occupancy_values=None,
    scheduler_factory=RRScheduler,
):
    if occupancy_values is None:
        occupancy_values = [1, 2, 4, 8, 16, 32]
        if max_occupancy not in occupancy_values:
            occupancy_values.append(max_occupancy)
        occupancy_values = [x for x in occupancy_values if x <= max_occupancy]

    idg = make_iterative_barrier_idg(iterations=iterations)
    rows = []

    for occupancy in occupancy_values:
        for work_group_size in work_group_sizes:
            warps_per_group = work_group_size // 32

            if occupancy % warps_per_group != 0:
                continue

            groups = occupancy // warps_per_group

            if groups < 1:
                continue

            exe = ExecutionConfig(
                hardware_threads_per_group=warps_per_group,
                concurrent_groups_per_core=groups,
            )

            print(
                "START RUN:",
                "occ=", occupancy,
                "wg_size=", work_group_size,
                "warps/group=", warps_per_group,
                "groups=", groups,
            )

            sim = PipelineSimulator(hardware, scheduler=scheduler_factory())
            result = sim.run_idg(f"{gpu_name}_iter_barrier", idg, exe, tracing=False)

            compute_instr_per_warp = sum(1 for instr in idg.values() if instr.op == "compute")
            ipc_alu = occupancy * compute_instr_per_warp / result.cycles

            rows.append({
                "gpu": gpu_name,
                "work_group_size": work_group_size,
                "warps_per_group": warps_per_group,
                "groups": groups,
                "occupancy": occupancy,
                "cycles": result.cycles,
                "warps_per_cycle": result.hardware_threads_per_cycle,
                "instructions_per_cycle": result.instructions_per_cycle,
                "alu_instructions_per_cycle": ipc_alu
            })

    rows.sort(key=lambda r: (r["occupancy"], r["work_group_size"]))
    return rows


def fig1_occupancy_sweep():
    return sweep_occupancy(
        "fig1_toy",
        make_fig1_toy_idg(),
        PAPER_TOY_HARDWARE,
        [1],
        [1, 2, 4, 10],
    )

def fig16_iterative_barrier_sweep():
    fermi_rows = sweep_iterative_barrier_sampled(
        "Fermi",
        FERMI_BARRIER_HARDWARE,
        max_occupancy=48,
        iterations=256,
    )

    pascal_rows = sweep_iterative_barrier_sampled(
        "Pascal",
        PASCAL_BARRIER_HARDWARE,
        max_occupancy=64,
        iterations=256,
    )

    return fermi_rows, pascal_rows


if __name__ == "__main__":
    fig1_results = fig1_occupancy_sweep()
    print_results(fig1_results)
    #write_results_csv(fig1_results, "fig1_toy_results.csv")

    fermi_rows, pascal_rows = fig16_iterative_barrier_sweep()
    print_barrier_sweep(fermi_rows)
    print_barrier_sweep(pascal_rows)
    #write_results_csv(fermi_rows, "fermi_iter_barrier_results.csv")
    #write_results_csv(pascal_rows, "pascal_iter_barrier_results.csv")
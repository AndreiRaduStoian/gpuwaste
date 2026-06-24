import csv
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from hardware import CoreHardwareConfig, InstructionTiming, RTX_3070_CALIBRATION_HARDWARE
from pipelinesim import PipelineSimulator
from schedulers import RRScheduler
from software import (
    Instruction,
    cuda_launch_to_pipeline_config,
    make_compute_fma_idg,
    make_shared_barrier_idg,
    make_vector_add_idg,
)


# Used 1132 MHz for accelsim comparison
# Used 1815 MHzf or native comparison
BASE_CORE_CLOCK_CYCLES_PER_US = 1132.0
BASE_DRAM_CLOCK_RATIO = 1.0
BASE_SM_COUNT = 46
BASE_SP_UNITS = 4
BASE_SFU_UNITS = 4
WARP_SIZE = 32
MAX_WARPS_PER_SM = 48


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def resident_blocks_for(block_size: int, max_warps_per_sm: int = MAX_WARPS_PER_SM) -> int:
    warps_per_block = ceil_div(block_size, WARP_SIZE)
    return max(1, max_warps_per_sm // warps_per_block)


@dataclass(frozen=True)
class KernelCase:
    name: str
    n: int
    iterations: int
    block_size: int


@dataclass(frozen=True)
class HardwareCase:
    name: str
    family: str
    fraction: float
    sm_count: int
    resident_blocks_per_sm: int
    clock_cycles_per_us: float
    hardware: CoreHardwareConfig


def copy_timing(t: InstructionTiming) -> InstructionTiming:
    return InstructionTiming(t.subsystem, t.lambda_cpi, t.completion_latency)


def scaled_timing(t: InstructionTiming, lambda_scale: float = 1.0, latency_scale: float = 1.0) -> InstructionTiming:
    return InstructionTiming(
        t.subsystem,
        t.lambda_cpi * lambda_scale,
        t.completion_latency * latency_scale,
    )


def make_hardware(timings: Dict[str, InstructionTiming], issue_limit_ipc: Optional[float] = None) -> CoreHardwareConfig:
    base = RTX_3070_CALIBRATION_HARDWARE
    return CoreHardwareConfig(
        subsystems=base.subsystems,
        issue_limit_ipc=base.issue_limit_ipc if issue_limit_ipc is None else issue_limit_ipc,
        timings=timings,
    )


def base_timings() -> Dict[str, InstructionTiming]:
    return {op: copy_timing(t) for op, t in RTX_3070_CALIBRATION_HARDWARE.timings.items()}


def sp_scaled_hardware(scale: float) -> CoreHardwareConfig:
    timings = base_timings()
    factor = BASE_SP_UNITS / (BASE_SP_UNITS * scale)
    for op in ("fma_f32", "add_f32"):
        timings[op] = scaled_timing(timings[op], lambda_scale=factor)
    return make_hardware(timings)


def sfu_scaled_hardware(scale: float) -> CoreHardwareConfig:
    timings = base_timings()
    factor = BASE_SFU_UNITS / (BASE_SFU_UNITS * scale)
    timings["sfu"] = scaled_timing(timings["sfu"], lambda_scale=factor)
    return make_hardware(timings)


def mem_scaled_hardware(scale: float) -> CoreHardwareConfig:
    timings = base_timings()
    factor = 1.0 / scale
    timings["global_mem"] = scaled_timing(timings["global_mem"], lambda_scale=factor, latency_scale=factor)
    return make_hardware(timings)


def core_scaled_hardware(scale: float) -> CoreHardwareConfig:
    timings = base_timings()
    # DRAM clock is fixed in these Accel-Sim cases. Expressing memory in core
    # cycles therefore shrinks global_mem cycles with core clock, making its
    # real-time cost approximately constant while ALU/local/barrier/SFU slow down
    # through clock_cycles_per_us.
    timings["global_mem"] = scaled_timing(timings["global_mem"], lambda_scale=scale, latency_scale=scale)
    return make_hardware(timings)


def make_hardware_cases(block_size: int) -> list[HardwareCase]:
    rb = resident_blocks_for(block_size)
    return [
        HardwareCase("RTX3070", "BASE", 1.00, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, RTX_3070_CALIBRATION_HARDWARE),

        HardwareCase("SM075", "SM", 0.75, 35, rb, BASE_CORE_CLOCK_CYCLES_PER_US, RTX_3070_CALIBRATION_HARDWARE),
        HardwareCase("SM050", "SM", 0.50, 23, rb, BASE_CORE_CLOCK_CYCLES_PER_US, RTX_3070_CALIBRATION_HARDWARE),
        HardwareCase("SM025", "SM", 0.25, 12, rb, BASE_CORE_CLOCK_CYCLES_PER_US, RTX_3070_CALIBRATION_HARDWARE),

        HardwareCase("SP075", "SP", 0.75, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sp_scaled_hardware(0.75)),
        HardwareCase("SP050", "SP", 0.50, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sp_scaled_hardware(0.50)),
        HardwareCase("SP025", "SP", 0.25, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sp_scaled_hardware(0.25)),

        HardwareCase("SFU075", "SFU", 0.75, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sfu_scaled_hardware(0.75)),
        HardwareCase("SFU050", "SFU", 0.50, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sfu_scaled_hardware(0.50)),
        HardwareCase("SFU025", "SFU", 0.25, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, sfu_scaled_hardware(0.25)),

        HardwareCase("CORE075", "CORE", 0.75, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US * 0.75, core_scaled_hardware(0.75)),
        HardwareCase("CORE050", "CORE", 0.50, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US * 0.50, core_scaled_hardware(0.50)),
        HardwareCase("CORE025", "CORE", 0.25, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US * 0.25, core_scaled_hardware(0.25)),

        HardwareCase("MEM075", "MEM", 0.75, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, mem_scaled_hardware(0.75)),
        HardwareCase("MEM050", "MEM", 0.50, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, mem_scaled_hardware(0.50)),
        HardwareCase("MEM025", "MEM", 0.25, 46, rb, BASE_CORE_CLOCK_CYCLES_PER_US, mem_scaled_hardware(0.25)),
    ]


def make_pointer_chase_idg(iterations: int) -> dict[str, Instruction]:
    idg = {}
    prev = None
    for i in range(iterations):
        instr_id = f"it{i:04d}_ld_next"
        deps = (prev,) if prev else ()
        idg[instr_id] = Instruction(instr_id, "global_mem", deps=deps)
        prev = instr_id
    idg["st_out"] = Instruction("st_out", "global_mem", deps=(prev,))
    return idg


def make_sfu_idg(iterations: int) -> dict[str, Instruction]:
    idg = {}
    prev = None
    for i in range(iterations):
        sin_id = f"it{i:04d}_sinf"
        cos_id = f"it{i:04d}_cosf"
        mul_id = f"it{i:04d}_x2"
        add_id = f"it{i:04d}_add1"
        rsq_id = f"it{i:04d}_rsqrtf"

        idg[sin_id] = Instruction(sin_id, "sfu", deps=(prev,) if prev else ())
        idg[cos_id] = Instruction(cos_id, "sfu", deps=(sin_id,))
        idg[mul_id] = Instruction(mul_id, "fma_f32", deps=(cos_id,))
        idg[add_id] = Instruction(add_id, "add_f32", deps=(mul_id,))
        idg[rsq_id] = Instruction(rsq_id, "sfu", deps=(add_id,))
        prev = rsq_id

    idg["st_out"] = Instruction("st_out", "global_mem", deps=(prev,))
    return idg


def idg_for(case):
    if case.name == "compute":
        return make_compute_fma_idg(case.iterations)
    if case.name == "vectoradd":
        return make_vector_add_idg(case.iterations)
    if case.name == "shared":
        return make_shared_barrier_idg(case.iterations)
    if case.name == "pointer":
        return make_pointer_chase_idg(case.iterations)
    if case.name == "sfu":
        return make_sfu_idg(case.iterations)
    raise ValueError(f"unknown kernel {case.name!r}")


def kernel_sets():
    return {
        # "synthetic_reduced": [
        #     KernelCase("vectoradd", 1_048_576, 1, 256),
        #     KernelCase("compute", 262_144, 100, 256),
        #     KernelCase("pointer", 262_144, 20, 256),
        #     KernelCase("shared", 262_144, 20, 256),
        #     KernelCase("sfu", 262_144, 20, 256),
        # ],
        "synthetic_large3": [
            KernelCase("compute", 1_048_576, 10, 256),
            KernelCase("vectoradd", 1_048_576, 10, 256),
            KernelCase("shared", 1_048_576, 10, 256),
            KernelCase("shared", 1_048_576, 10, 256),
            KernelCase("sfu", 1_048_576, 10, 256),
        ],
        # "synthetic_large3_n2": [
        #     KernelCase("compute", 2_097_152, 10, 256),
        #     KernelCase("vectoradd", 2_097_152, 10, 256),
        #     KernelCase("shared", 2_097_152, 10, 256),
        #     KernelCase("shared", 2_097_152, 10, 256),
        #     KernelCase("sfu", 2_097_152, 10, 256),
        # ],
        # "synthetic_large3_n4": [
        #     KernelCase("compute", 4_194_304, 10, 256),
        #     KernelCase("vectoradd", 4_194_304, 10, 256),
        #     KernelCase("shared", 4_194_304, 10, 256),
        # ],
    }


def simulate_single_sm(kernel_case, hw_case, cache):
    key = (kernel_case.name, kernel_case.iterations, kernel_case.block_size, hw_case.name)
    if key in cache:
        return cache[key]

    _, exe = cuda_launch_to_pipeline_config(
        n=kernel_case.block_size,
        block_size=kernel_case.block_size,
        resident_blocks_per_sm=hw_case.resident_blocks_per_sm,
    )
    idg = idg_for(kernel_case)
    result = PipelineSimulator(hw_case.hardware, scheduler=RRScheduler()).run_idg(kernel_case.name, idg, exe, tracing=False)
    cache[key] = (exe, result)
    return exe, result


def run_one(case_set, kernel_case, hw_case, cache):
    grid_size, _ = cuda_launch_to_pipeline_config(
        n=kernel_case.n,
        block_size=kernel_case.block_size,
        resident_blocks_per_sm=hw_case.resident_blocks_per_sm,
    )
    exe, result = simulate_single_sm(kernel_case, hw_case, cache)

    blocks_per_wave = hw_case.sm_count * hw_case.resident_blocks_per_sm
    waves = ceil_div(grid_size, blocks_per_wave)
    gpu_cycles = result.cycles * waves
    time_us = gpu_cycles / hw_case.clock_cycles_per_us

    return {
        "set": case_set,
        "kernel": kernel_case.name,
        "n": kernel_case.n,
        "iterations": kernel_case.iterations,
        "block_size": kernel_case.block_size,
        "hardware": hw_case.name,
        "family": hw_case.family,
        "fraction": hw_case.fraction,
        "sm_count": hw_case.sm_count,
        "resident_blocks_per_sm": hw_case.resident_blocks_per_sm,
        "grid_size": grid_size,
        "warps_per_block": exe.hardware_threads_per_group,
        "resident_warps_per_sm": exe.occupancy_warps,
        "blocks_per_wave": blocks_per_wave,
        "waves": waves,
        "instr_per_warp": result.instruction_count_per_hardware_thread,
        "single_sm_wave_cycles": result.cycles,
        "gpu_cycles": gpu_cycles,
        "clock_cycles_per_us": hw_case.clock_cycles_per_us,
        "time_us": time_us,
        "time_ms": time_us / 1000.0,
        "single_sm_instr_per_cycle": result.instructions_per_cycle,
        "single_sm_warps_per_cycle": result.hardware_threads_per_cycle,
    }


def add_slowdown_and_waste(rows):
    base_time = {}
    for row in rows:
        if row["hardware"] == "RTX3070":
            base_time[(row["set"], row["kernel"], row["n"], row["iterations"], row["block_size"])] = row["time_us"]

    for row in rows:
        key = (row["set"], row["kernel"], row["n"], row["iterations"], row["block_size"])
        b = base_time[key]
        slowdown = row["time_us"] / b if b else math.inf
        fraction = row["fraction"]
        ideal_slowdown = 1.0 / fraction if fraction else math.inf

        if row["hardware"] == "RTX3070" or ideal_slowdown == 1.0:
            sensitivity = 1.0
            waste_proxy = 0.0
        else:
            denom = ideal_slowdown - 1.0
            sensitivity = (slowdown - 1.0) / denom if denom > 0 else 1.0
            sensitivity = max(0.0, min(1.0, sensitivity))
            waste_proxy = 1.0 - sensitivity

        row["base_time_us"] = b
        row["slowdown_vs_base"] = slowdown
        row["ideal_slowdown_if_fully_bound"] = ideal_slowdown
        row["scaling_sensitivity"] = sensitivity
        row["waste_proxy"] = waste_proxy


def main():
    rows = []
    sim_cache = {}
    for set_name, kernels in kernel_sets().items():
        for kernel_case in kernels:
            for hw_case in make_hardware_cases(kernel_case.block_size):
                print(f"running {set_name} {kernel_case.name} {kernel_case.n} {kernel_case.iterations} {kernel_case.block_size} {hw_case.name}")
                rows.append(run_one(set_name, kernel_case, hw_case, sim_cache))

    add_slowdown_and_waste(rows)

    fieldnames = list(rows[0].keys())
    with open("pipeline_scaling_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to pipeline_scaling_results.csv")


if __name__ == "__main__":
    main()

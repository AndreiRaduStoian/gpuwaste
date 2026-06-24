from pipelinesim import PipelineSimulator
from schedulers import RRScheduler

from software import make_compute_fma_idg, make_pointer_chase_idg, make_sfu_idg, make_vector_add_idg, make_shared_barrier_idg
from software import cuda_launch_to_pipeline_config
from hardware import RTX_3070_CALIBRATION_HARDWARE

import math
import csv

OUT_CSV = "pipeline_base3070_predictions.csv"

N = [524288, 1572864, 2097152, 4194304]
ITERATIONS = [1, 10, 20]
NAMES = ["compute", "vectoradd", "shared", "pointer", "sfu"]
KERNELS = [make_compute_fma_idg, make_vector_add_idg, make_shared_barrier_idg, make_pointer_chase_idg, make_sfu_idg]
SM_COUNT = 46
BLOCK_SIZE = 256
CLOCK_CYCLES_PER_US = 1815.0
RESIDENT_BLOCKS_PER_SM = 6

# for n in N:
#     grid_size, exe = cuda_launch_to_pipeline_config(
#         n=n,
#         block_size=256,
#         resident_blocks_per_sm=RESIDENT_BLOCKS_PER_SM,
#     )

#     for idg, name in zip(KERNELS, NAMES):
#         kernel_name = name
#         result = PipelineSimulator(RTX_3070_CALIBRATION_HARDWARE, scheduler=RRScheduler()).run_idg(kernel_name, idg, exe)

#         waves = math.ceil(grid_size / (SM_COUNT * RESIDENT_BLOCKS_PER_SM))
#         gpu_cycles = result.cycles * waves
#         gpu_time_us = gpu_cycles / CLOCK_CYCLES_PER_US

#         print("Kernel:", kernel_name)
#         print("grid_size =", grid_size)
#         print("waves =", waves)
#         print("single_sm_wave_cycles =", result.cycles)
#         print("gpu_cycles =", gpu_cycles)
#         print("gpu_time_us =", gpu_time_us)
#         print("=================================================================")


def main():
    rows = []

    for n in N:
        for iterations in ITERATIONS:
            for name, kernel in zip(NAMES, KERNELS):
                grid_size, exe = cuda_launch_to_pipeline_config(
                    n=n,
                    block_size=BLOCK_SIZE,
                    resident_blocks_per_sm=RESIDENT_BLOCKS_PER_SM,
                )

                sim = PipelineSimulator(RTX_3070_CALIBRATION_HARDWARE, scheduler=RRScheduler())       
                result = sim.run_idg(name, kernel(iterations), exe, tracing=False)

                blocks_per_wave = SM_COUNT * RESIDENT_BLOCKS_PER_SM
                waves = math.ceil(grid_size / blocks_per_wave)
                gpu_cycles = result.cycles * waves
                time_us = gpu_cycles / CLOCK_CYCLES_PER_US

                row = {
                    "kernel": name,
                    "n": n,
                    "iterations": iterations,
                    "single_sm_wave_cycles": result.cycles,
                    "gpu_cycles": gpu_cycles,
                    "predicted_runtime_ms": time_us / 1000.0,
                }

                rows.append(row)

                print(
                    name,
                    "N=", n,
                    "I=", iterations,
                    "single_sm_cycles=", f"{row['single_sm_wave_cycles']:.2f}",
                    "gpu_cycles=", f"{row['gpu_cycles']:.2f}",
                    "ms=", f"{row['predicted_runtime_ms']:.6f}",
                )


    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
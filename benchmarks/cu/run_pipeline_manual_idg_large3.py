from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv

# Adjust these imports if your pipeline model files live somewhere else.
from idg import Instruction
from pipelinesim import PipelineSimulator, ExecutionConfig, build_warps_from_idg
from hardware import TOY_PTX_HARDWARE, make_ptx_static_mapper
from schedulers import RoundRobinWarpScheduler


N = 1_048_576
ITERATIONS = 10
BLOCK_SIZE = 256

# Pipeline model groups are not CUDA thread blocks directly.
# Here, one group represents one resident CTA-like group and group_size is
# the number of warps per block.
WARPS_PER_BLOCK = BLOCK_SIZE // 32

# Keep this moderate so the prototype runs quickly.
# Full CUDA grid would be N / 256 = 4096 blocks, which may be too large for
# this prototype. Start small; increase later if needed.
NUM_GROUPS = 46


def instr(i: str, op: str, deps=()):
    return Instruction(id=i, op=op, deps=tuple(deps))


def make_compute_idg(iterations: int) -> dict[str, Instruction]:
    idg: dict[str, Instruction] = {}

    idg["tid"] = instr("tid", "alu")
    idg["init_x"] = instr("init_x", "alu", ["tid"])
    idg["init_y"] = instr("init_y", "alu")
    idg["init_z"] = instr("init_z", "alu")

    x = "init_x"
    y = "init_y"
    z = "init_z"

    for it in range(iterations):
        f0 = f"it{it}_fma_x"
        f1 = f"it{it}_fma_y"
        f2 = f"it{it}_fma_z"

        # x = fmaf(x, y, z)
        idg[f0] = instr(f0, "fma.rn.f32", [x, y, z])

        # y = fmaf(y, z, x)
        idg[f1] = instr(f1, "fma.rn.f32", [y, z, f0])

        # z = fmaf(z, x, y)
        idg[f2] = instr(f2, "fma.rn.f32", [z, f0, f1])

        x, y, z = f0, f1, f2

    idg["sum_xy"] = instr("sum_xy", "add.f32", [x, y])
    idg["sum_xyz"] = instr("sum_xyz", "add.f32", ["sum_xy", z])
    idg["store_out"] = instr("store_out", "st.global.f32", ["sum_xyz", "tid"])

    return idg


def make_vectoradd_idg() -> dict[str, Instruction]:
    return {
        "tid": instr("tid", "alu"),
        "cmp": instr("cmp", "setp.lt.s32", ["tid"]),
        "ld_a": instr("ld_a", "ld.global.f32", ["cmp", "tid"]),
        "ld_b": instr("ld_b", "ld.global.f32", ["cmp", "tid"]),
        "add": instr("add", "add.f32", ["ld_a", "ld_b"]),
        "st_c": instr("st_c", "st.global.f32", ["add", "tid"]),
    }


def make_shared_idg(iterations: int) -> dict[str, Instruction]:
    idg: dict[str, Instruction] = {}

    idg["tid"] = instr("tid", "alu")
    idg["local"] = instr("local", "alu")
    idg["cmp"] = instr("cmp", "setp.lt.s32", ["tid"])
    idg["ld_in"] = instr("ld_in", "ld.global.f32", ["cmp", "tid"])
    idg["st_tile_init"] = instr("st_tile_init", "st.shared.f32", ["ld_in", "local"])
    idg["bar_init"] = instr("bar_init", "bar.sync", ["st_tile_init"])

    prev_bar = "bar_init"

    for it in range(iterations):
        left = f"it{it}_ld_left"
        mid = f"it{it}_ld_mid"
        right = f"it{it}_ld_right"
        compute1 = f"it{it}_mul_left"
        compute2 = f"it{it}_mul_mid"
        compute3 = f"it{it}_mul_right"
        add1 = f"it{it}_add_lm"
        add2 = f"it{it}_add_lmr"
        bar_before_store = f"it{it}_bar_before_store"
        st = f"it{it}_st_tile"
        bar_after_store = f"it{it}_bar_after_store"

        idg[left] = instr(left, "ld.shared.f32", [prev_bar, "local"])
        idg[mid] = instr(mid, "ld.shared.f32", [prev_bar, "local"])
        idg[right] = instr(right, "ld.shared.f32", [prev_bar, "local"])

        # The CUDA code has a barrier after reading left/mid/right and before
        # writing the updated tile value.
        idg[bar_before_store] = instr(bar_before_store, "bar.sync", [left, mid, right])

        idg[compute1] = instr(compute1, "mul.f32", [left, bar_before_store])
        idg[compute2] = instr(compute2, "mul.f32", [mid, bar_before_store])
        idg[compute3] = instr(compute3, "mul.f32", [right, bar_before_store])
        idg[add1] = instr(add1, "add.f32", [compute1, compute2])
        idg[add2] = instr(add2, "add.f32", [add1, compute3])

        idg[st] = instr(st, "st.shared.f32", [add2, "local"])
        idg[bar_after_store] = instr(bar_after_store, "bar.sync", [st])

        prev_bar = bar_after_store

    idg["ld_final"] = instr("ld_final", "ld.shared.f32", [prev_bar, "local"])
    idg["st_out"] = instr("st_out", "st.global.f32", ["ld_final", "tid"])

    return idg


def run_kernel(name: str, idg: dict[str, Instruction]) -> dict[str, object]:
    exe = ExecutionConfig(
        group_size=WARPS_PER_BLOCK,
        num_groups=NUM_GROUPS,
    )

    warps = build_warps_from_idg(exe, idg)

    sim = PipelineSimulator(
        hardware=deepcopy(TOY_PTX_HARDWARE),
        mapper=make_ptx_static_mapper(TOY_PTX_HARDWARE),
        scheduler=RoundRobinWarpScheduler(),
        execution_config=exe,
    )

    cycles = sim.run(warps)

    subsystem_counts: dict[str, int] = {}
    for event in sim.get_trace():
        subsystem_counts[event.subsystem] = subsystem_counts.get(event.subsystem, 0) + 1

    return {
        "kernel": name,
        "instructions_per_warp": len(idg),
        "num_groups": NUM_GROUPS,
        "warps_per_group": WARPS_PER_BLOCK,
        "total_warps": len(warps),
        "cycles": cycles,
        "alu_events": subsystem_counts.get("alu", 0),
        "mem_events": subsystem_counts.get("mem", 0),
        "barrier_events": subsystem_counts.get("barrier", 0),
        "sfu_events": subsystem_counts.get("sfu", 0),
        "control_events": subsystem_counts.get("control", 0),
    }


def main() -> None:
    kernels = {
        "compute": make_compute_idg(ITERATIONS),
        "vectoradd": make_vectoradd_idg(),
        "shared": make_shared_idg(ITERATIONS),
    }

    out_dir = Path("/workspace/gpuwaste/results/pipeline_manual_large3")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, idg in kernels.items():
        print(f"Running pipeline model for {name}...")
        row = run_kernel(name, idg)
        rows.append(row)
        print(row)

    out_csv = out_dir / "pipeline_manual_large3_baseline.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

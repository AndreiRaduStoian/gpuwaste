# Breaks


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from idg import Instruction, WarpState
from schedulers import RoundRobinWarpScheduler
from pipelinesim import PipelineSimulator
from hardware import TOY_HARDWARE, make_static_mapper


def make_example_warp(warp_id: int) -> WarpState:
    instructions = {
        "i0": Instruction("i0", "alu"),
        "i1": Instruction("i1", "alu"),
        "i2": Instruction("i2", "mem", deps=("i0",)),
        "i3": Instruction("i3", "alu", deps=("i1", "i2")),
    }

    return WarpState(
        warp_id=warp_id,
        group_id=0,
        instructions=instructions,
    )


warps = [make_example_warp(i) for i in range(4)]

sim = PipelineSimulator(
    hardware=TOY_HARDWARE,
    mapper=make_static_mapper(TOY_HARDWARE),
    scheduler=RoundRobinWarpScheduler(),
)

cycles = sim.run(warps)
print(f"Runtime: {cycles} cycles")
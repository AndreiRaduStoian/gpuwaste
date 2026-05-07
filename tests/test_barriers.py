import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from idg import Instruction, WarpState
from hardware import HardwareConfig, InstructionTiming, make_static_mapper
from pipelinesim import PipelineSimulator
from schedulers import RoundRobinWarpScheduler


def test_barrier_synchronizes_group():
    hardware = HardwareConfig(
        subsystems=("alu", "barrier"),
        issue_limit=1,
        timings={
            "slow": InstructionTiming("alu", cpi=1.0, latency=10.0),
            "fast": InstructionTiming("alu", cpi=1.0, latency=1.0),
            "bar.sync": InstructionTiming("barrier", cpi=1.0, latency=1.0),
            "after": InstructionTiming("alu", cpi=1.0, latency=1.0),
        },
    )

    warp0 = WarpState(
        warp_id=0,
        group_id=0,
        instructions={
            "i0": Instruction("i0", "slow"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    warp1 = WarpState(
        warp_id=1,
        group_id=0,
        instructions={
            "i0": Instruction("i0", "fast"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run([warp0, warp1])

    trace = sim.get_trace()

    post_barrier_warp1 = [
        e for e in trace
        if e.warp_id == 1 and e.instr_id == "i1"
    ][0]

    barrier_warp0 = [
        e for e in trace
        if e.warp_id == 0 and e.instr_id == "b0"
    ][0]

    assert post_barrier_warp1.issue_time >= barrier_warp0.complete_time


def test_barrier_synchronizes_only_within_group():
    hardware = HardwareConfig(
        subsystems=("alu", "barrier"),
        issue_limit=1,
        timings={
            "slow": InstructionTiming("alu", cpi=1.0, latency=20.0),
            "fast": InstructionTiming("alu", cpi=1.0, latency=1.0),
            "bar.sync": InstructionTiming("barrier", cpi=1.0, latency=1.0),
            "after": InstructionTiming("alu", cpi=1.0, latency=1.0),
        },
    )

    # Group 0:
    # warp 0 is slow before the barrier
    # warp 1 is fast before the barrier
    warp0 = WarpState(
        warp_id=0,
        group_id=0,
        instructions={
            "i0": Instruction("i0", "slow"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    warp1 = WarpState(
        warp_id=1,
        group_id=0,
        instructions={
            "i0": Instruction("i0", "fast"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    # Group 1:
    # both warps are fast and should not wait for group 0.
    warp2 = WarpState(
        warp_id=2,
        group_id=1,
        instructions={
            "i0": Instruction("i0", "fast"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    warp3 = WarpState(
        warp_id=3,
        group_id=1,
        instructions={
            "i0": Instruction("i0", "fast"),
            "b0": Instruction("b0", "bar.sync", deps=("i0",)),
            "i1": Instruction("i1", "after", deps=("b0",)),
        },
    )

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run([warp0, warp1, warp2, warp3])

    trace = sim.get_trace()

    group0_slow_barrier = [
        e for e in trace
        if e.warp_id == 0 and e.instr_id == "b0"
    ][0]

    group1_post_barrier = [
        e for e in trace
        if e.warp_id == 2 and e.instr_id == "i1"
    ][0]

    assert group1_post_barrier.issue_time < group0_slow_barrier.complete_time

test_barrier_synchronizes_group()
test_barrier_synchronizes_only_within_group()
print("done")
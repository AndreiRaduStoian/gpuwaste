import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

from idg import Instruction, WarpState
from hardware import HardwareConfig, InstructionTiming, make_static_mapper
from pipelinesim import PipelineSimulator
from schedulers import RoundRobinWarpScheduler


def test_issue_limit_one_instruction_per_cycle():
    hardware = HardwareConfig(
        subsystems=("alu",),
        issue_limit=1,
        timings={
            "alu": InstructionTiming("alu", cpi=0.0, latency=1.0),
        },
    )

    warps = [
        WarpState(
            warp_id=i,
            group_id=0,
            instructions={
                "i0": Instruction("i0", "alu"),
            },
        )
        for i in range(4)
    ]

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run(warps)

    issue_times = [e.issue_time for e in sim.get_trace()]
    print(issue_times)

    assert issue_times.count(0.0) == 1


def test_independent_subsystems_can_issue_same_cycle():
    hardware = HardwareConfig(
        subsystems=("alu", "mem"),
        issue_limit=2,
        timings={
            "alu": InstructionTiming("alu", cpi=1.0, latency=5.0),
            "mem": InstructionTiming("mem", cpi=1.0, latency=5.0),
        },
    )

    warp0 = WarpState(
        warp_id=0,
        group_id=0,
        instructions={
            "a": Instruction("a", "alu"),
        },
    )

    warp1 = WarpState(
        warp_id=1,
        group_id=0,
        instructions={
            "m": Instruction("m", "mem"),
        },
    )

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run([warp0, warp1])

    trace = sim.get_trace()

    assert len(trace) == 2
    assert trace[0].issue_time == 0.0
    assert trace[1].issue_time == 0.0


def test_subsystem_cpi_limits_issue_spacing():
    hardware = HardwareConfig(
        subsystems=("alu",),
        issue_limit=10,
        timings={
            "alu": InstructionTiming("alu", cpi=2.0, latency=1.0),
        },
    )

    warps = [
        WarpState(
            warp_id=i,
            group_id=0,
            instructions={
                "i0": Instruction("i0", "alu"),
            },
        )
        for i in range(3)
    ]

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run(warps)

    issue_times = [e.issue_time for e in sim.get_trace()]

    assert issue_times == [0.0, 2.0, 4.0]


def test_dependency_waits_for_completion_latency():
    hardware = HardwareConfig(
        subsystems=("alu",),
        issue_limit=10,
        timings={
            "alu": InstructionTiming("alu", cpi=1.0, latency=7.0),
        },
    )

    warp = WarpState(
        warp_id=0,
        group_id=0,
        instructions={
            "i0": Instruction("i0", "alu"),
            "i1": Instruction("i1", "alu", deps=("i0",)),
        },
    )

    sim = PipelineSimulator(
        hardware=hardware,
        mapper=make_static_mapper(hardware),
        scheduler=RoundRobinWarpScheduler(),
    )

    sim.run([warp])

    trace = sim.get_trace()
    by_id = {e.instr_id: e for e in trace}

    assert by_id["i0"].issue_time == 0.0
    assert by_id["i1"].issue_time >= 7.0


test_issue_limit_one_instruction_per_cycle()
test_independent_subsystems_can_issue_same_cycle()
test_subsystem_cpi_limits_issue_spacing()
test_dependency_waits_for_completion_latency()
print("done")
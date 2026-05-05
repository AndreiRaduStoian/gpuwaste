from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class InstructionTiming:
    """
    subsystem: abstract pipeline name
    cpi: λ, / inverse throughput
    latency: completion latency
    """
    subsystem: str
    cpi: float
    latency: float


@dataclass(frozen=True)
class HardwareConfig:
    """
    Hardware configuration for one simulated GPU core.
    """
    subsystems: Tuple[str, ...]
    issue_limit: int
    timings: Dict[str, InstructionTiming]

    def timing_for_class(self, instr_class: str) -> InstructionTiming:
        if instr_class not in self.timings:
            raise ValueError(f"No timing configured for instruction class: {instr_class}")

        timing = self.timings[instr_class]

        if timing.subsystem not in self.subsystems:
            raise ValueError(
                f"Timing for class {instr_class} maps to unknown subsystem {timing.subsystem}"
            )

        return timing


def classify_ptx_op(op: str) -> str:
    if op.startswith("ld.") or op.startswith("st."):
        return "mem"

    if op.startswith("bar."):
        return "barrier"

    if op == "bra" or op == "ret":
        return "control"

    if op.startswith(("sin.", "cos.", "ex2.", "lg2.", "sqrt.", "rsqrt.", "rcp.")):
        return "sfu"

    return "alu"


def make_ptx_static_mapper(hardware: HardwareConfig):
    def mapper(instr):
        instr_class = classify_ptx_op(instr.op)
        return hardware.timing_for_class(instr_class)

    return mapper


TOY_HARDWARE = HardwareConfig(
    subsystems=("alu", "mem"),
    issue_limit=1,
    timings={
        "alu": InstructionTiming(subsystem="alu", cpi=1.0, latency=4.0),
        "mem": InstructionTiming(subsystem="mem", cpi=2.0, latency=6.0),
    },
)


TOY_PTX_HARDWARE = HardwareConfig(
    subsystems=("alu", "mem", "barrier", "control", "sfu"),
    issue_limit=1,
    timings={
        "alu": InstructionTiming(subsystem="alu", cpi=1.0, latency=4.0),
        "mem": InstructionTiming(subsystem="mem", cpi=2.0, latency=20.0),
        "barrier": InstructionTiming(subsystem="barrier", cpi=1.0, latency=1.0),
        "control": InstructionTiming(subsystem="control", cpi=1.0, latency=1.0),
        "sfu": InstructionTiming(subsystem="sfu", cpi=4.0, latency=16.0),
    },
)
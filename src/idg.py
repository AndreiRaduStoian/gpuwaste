from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Instruction:
    id: str
    op: str
    deps: Tuple[str, ...] = ()

    reads: Tuple[str, ...] = ()
    writes: Tuple[str, ...] = ()
    raw: str = ""

    label: Optional[str] = None
    predicate: Optional[str] = None


@dataclass
class WarpState:
    """
    Runtime state of one warp's IDG.
    """
    warp_id: int
    group_id: int
    instructions: Dict[str, Instruction]

    issued: set[str] = field(default_factory=set)
    completed: Dict[str, float] = field(default_factory=dict)

    def ready_instructions(self, time: float) -> List[Instruction]:
        ready = []

        for instr in self.instructions.values():
            if instr.id in self.issued:
                continue

            if all(
                dep in self.completed and self.completed[dep] <= time
                for dep in instr.deps
            ):
                ready.append(instr)

        return ready

    def is_done(self) -> bool:
        return len(self.completed) == len(self.instructions)


def build_idg_from_def_use(
    parsed_instructions: List[Instruction],
    add_barrier_edges: bool = True,
) -> Dict[str, Instruction]:
    """
    Build IDG from a linear instruction stream using register def-use dependencies.
    """

    last_writer: Dict[str, str] = {}
    idg: Dict[str, Instruction] = {}

    current_region_instrs: set[str] = set()

    active_barrier: str | None = None

    for instr in parsed_instructions:
        deps = set(instr.deps)

        # Normal register def-use edges
        for reg in instr.reads:
            if reg in last_writer:
                deps.add(last_writer[reg])

        # Barrier ordering
        if add_barrier_edges:
            if instr.op.startswith("bar."):
                deps.update(current_region_instrs)

            else:
                if active_barrier is not None:
                    deps.add(active_barrier)

        new_instr = Instruction(
            id=instr.id,
            op=instr.op,
            deps=tuple(sorted(deps)),
            reads=instr.reads,
            writes=instr.writes,
            raw=instr.raw,
            label=instr.label,
            predicate=instr.predicate,
        )

        idg[new_instr.id] = new_instr

        for reg in instr.writes:
            last_writer[reg] = instr.id

        if add_barrier_edges:
            if instr.op.startswith("bar."):
                active_barrier = instr.id
                current_region_instrs = set()
            else:
                current_region_instrs.add(instr.id)

    return idg


def make_warp_from_idg(warp_id: int, group_id: int, idg: Dict[str, Instruction]) -> WarpState:
    return WarpState(
        warp_id=warp_id,
        group_id=group_id,
        instructions=dict(idg),
    )
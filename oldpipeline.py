from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import heapq
import itertools


@dataclass(frozen=True)
class Instruction:
    # Node of instruction dependency graph

    id: str
    op: str
    deps: Tuple[str, ...] = ()


@dataclass(frozen=True)
class InstructionTiming:
    # Timing info for a pipeline (compute, mem, SFU etc)

    subsystem: str
    cpi: float
    latency: float


@dataclass
class WarpState:
    # Runtime state of a warps IDG
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

            if all(dep in self.completed and self.completed[dep] <= time for dep in instr.deps):
                ready.append(instr)

        return ready

    def is_done(self) -> bool:
        return len(self.completed) == len(self.instructions)


@dataclass
class SubsystemState:
    name: str
    next_issue_time: float = 0.0


@dataclass
class HardwareConfig:
    subsystems: Tuple[str, ...]
    issue_limit: int


class Scheduler:
    # Receive issuable (warp,instruction) and return them in some order.

    def order(self, candidates: List[Tuple[WarpState, Instruction]]) -> List[Tuple[WarpState, Instruction]]:
        raise NotImplementedError


class LowestWarpFirstScheduler(Scheduler):
    def order(self, candidates: List[Tuple[WarpState, Instruction]]) -> List[Tuple[WarpState, Instruction]]:
        return sorted(candidates, key=lambda item: (item[0].warp_id, item[1].id))


class RoundRobinWarpScheduler(Scheduler):
    def __init__(self):
        self.next_warp = 0

    def order(self, candidates: List[Tuple[WarpState, Instruction]]) -> List[Tuple[WarpState, Instruction]]:
        if not candidates:
            return []

        max_warp = max(w.warp_id for w, _ in candidates)

        def key(item: Tuple[WarpState, Instruction]):
            warp, instr = item
            distance = (warp.warp_id - self.next_warp) % (max_warp + 1)
            return distance, instr.id

        ordered = sorted(candidates, key=key)

        if ordered:
            self.next_warp = (ordered[0][0].warp_id + 1) % (max_warp + 1)

        return ordered


# -----------------------------
# Sim
# -----------------------------

class PipelineSimulator:

    def __init__(
        self,
        hardware: HardwareConfig,
        mapper: Callable[[Instruction], InstructionTiming],
        scheduler: Optional[Scheduler] = None,
    ):
        self.hardware = hardware
        self.mapper = mapper
        self.scheduler = scheduler or LowestWarpFirstScheduler()

        self.subsystems = {
            name: SubsystemState(name)
            for name in hardware.subsystems
        }

    def run(self, warps: List[WarpState]) -> float:
        time = 0.0
        event_counter = itertools.count()

        # (completion_time, counter, warp_id, instruction_id)
        completions = []

        while not all(warp.is_done() for warp in warps):
            # Mark instructions completed up to current time
            while completions and completions[0][0] <= time:
                complete_time, _, warp_id, instr_id = heapq.heappop(completions)
                warps[warp_id].completed[instr_id] = complete_time

            if all(warp.is_done() for warp in warps):
                self._dump_state(time, warps, completions)
                break

            candidates = self._collect_candidates(warps, time)

            issued_count = 0
            issued_any = False

            for warp, instr in self.scheduler.order(candidates):
                if issued_count >= self.hardware.issue_limit:
                    break

                if instr.id in warp.issued:
                    continue

                timing = self.mapper(instr)
                subsystem = self.subsystems[timing.subsystem]

                if subsystem.next_issue_time > time:
                    continue

                # Issue instruction.
                warp.issued.add(instr.id)
                subsystem.next_issue_time = time + timing.cpi

                complete_time = time + timing.latency
                heapq.heappush(
                    completions,
                    (complete_time, next(event_counter), warp.warp_id, instr.id),
                )

                issued_count += 1
                issued_any = True

            if issued_any:
                continue

            next_time = self._next_event_time(time, completions)

            if next_time is None:
                self._dump_state(time, warps, completions)
                raise RuntimeError(
                    "Simulation deadlock: no ready instruction and no future event."
                )

            time = next_time

        return max(
            max(warp.completed.values(), default=0.0)
            for warp in warps
        )

    def _collect_candidates(
        self,
        warps: List[WarpState],
        time: float,
    ) -> List[Tuple[WarpState, Instruction]]:
        candidates = []

        for warp in warps:
            for instr in warp.ready_instructions(time):
                timing = self.mapper(instr)

                if timing.subsystem not in self.subsystems:
                    raise ValueError(f"Unknown subsystem: {timing.subsystem}")

                subsystem = self.subsystems[timing.subsystem]

                if subsystem.next_issue_time <= time:
                    candidates.append((warp, instr))

        return candidates

    def _next_event_time(self, current_time: float, completions) -> Optional[float]:
        times = []

        if completions:
            times.append(completions[0][0])

        for subsystem in self.subsystems.values():
            if subsystem.next_issue_time > current_time:
                times.append(subsystem.next_issue_time)

        return min(times) if times else None
        
    def _dump_state(self, time, warps, completions):
        print("\n=== DEADLOCK DEBUG DUMP ===")
        print(f"Current time: {time}\n")

        print("Subsystems:")
        for name, sub in self.subsystems.items():
            print(f"  {name}: next_issue_time={sub.next_issue_time}")

        print("\nWarps:")
        for warp in warps:
            print(f"  Warp {warp.warp_id}:")
            print(f"    issued: {sorted(warp.issued)}")
            print(f"    completed: {warp.completed}")

            ready = warp.ready_instructions(time)
            print(f"    ready now: {[instr.id for instr in ready]}")

            not_issued = [
                instr.id for instr in warp.instructions.values()
                if instr.id not in warp.issued
            ]
            print(f"    not issued: {not_issued}")

            blocked = []
            for instr in warp.instructions.values():
                if instr.id in warp.issued:
                    continue

                missing = [
                    dep for dep in instr.deps
                    if dep not in warp.completed
                ]
                if missing:
                    blocked.append((instr.id, missing))

            print(f"    blocked deps: {blocked}")
            print()

        print("Pending completions:")
        for entry in completions:
            print(f"  {entry}")

        print("=== END DEBUG DUMP ===\n")


# -------------------------
# Test
# -------------------------


def example_mapper(instr: Instruction) -> InstructionTiming:
    if instr.op == "alu":
        return InstructionTiming(subsystem="alu", cpi=1.0, latency=4.0)

    if instr.op == "mem":
        return InstructionTiming(subsystem="mem", cpi=2.0, latency=6.0)

    raise ValueError(f"Unknown op: {instr.op}")


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


hardware = HardwareConfig(
    subsystems=("alu", "mem"),
    issue_limit=1,
)

warps = [make_example_warp(i) for i in range(4)]

sim = PipelineSimulator(
    hardware=hardware,
    mapper=example_mapper,
    scheduler=RoundRobinWarpScheduler(),
)

cycles = sim.run(warps)
print(f"Runtime: {cycles} cycles")

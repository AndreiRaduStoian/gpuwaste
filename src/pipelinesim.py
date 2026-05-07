from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import heapq
import itertools

from idg import Instruction, WarpState
from schedulers import Scheduler, LowestWarpFirstScheduler, RoundRobinWarpScheduler
from hardware import InstructionTiming, HardwareConfig


@dataclass
class SubsystemState:
    name: str
    next_issue_time: float = 0.0


@dataclass(frozen=True)
class TraceEvent:
    group_id: int
    warp_id: int
    instr_id: str
    op: str
    subsystem: str
    issue_time: float
    complete_time: float
    deps: Tuple[str, ...]
    raw: str = ""


@dataclass(frozen=True)
class ExecutionConfig:
    group_size: int
    num_groups: int

    @property
    def occupancy(self) -> int:
        # In paper: ω = |γ| * γ.
        return self.group_size * self.num_groups


def build_warps_from_idg(exe: ExecutionConfig, idg: Dict[str, Instruction]) -> List[WarpState]:
    warps = []
    warp_id = 0

    for group_id in range(exe.num_groups):
        for _ in range(exe.group_size):
            warps.append(
                WarpState(
                    warp_id=warp_id,
                    group_id=group_id,
                    instructions=dict(idg),
                )
            )
            warp_id += 1

    return warps
# -----------------------------
# Sim
# -----------------------------

class PipelineSimulator:

    def __init__(
        self,
        hardware: HardwareConfig,
        mapper: Callable[[Instruction], InstructionTiming],
        scheduler: Optional[Scheduler] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ):
        self.hardware = hardware
        self.mapper = mapper
        self.scheduler = scheduler or LowestWarpFirstScheduler()

        self.subsystems = {
            name: SubsystemState(name)
            for name in hardware.subsystems
        }

        self.trace: List[TraceEvent] = []

    # def run(self, warps: List[WarpState]) -> float:
    #     time = 0.0
    #     event_counter = itertools.count()

    #     # (completion_time, counter, warp_id, instruction_id)
    #     completions = []

    #     while not all(warp.is_done() for warp in warps):
    #         # Mark instructions completed up to current time
    #         while completions and completions[0][0] <= time:
    #             complete_time, _, warp_id, instr_id = heapq.heappop(completions)
    #             warps[warp_id].completed[instr_id] = complete_time

    #         if all(warp.is_done() for warp in warps):
    #             self._dump_state(time, warps, completions)
    #             break

    #         candidates = self._collect_candidates(warps, time)

    #         issued_count = 0
    #         issued_any = False

    #         for warp, instr in self.scheduler.order(candidates):
    #             if issued_count >= self.hardware.issue_limit:
    #                 break

    #             if instr.id in warp.issued:
    #                 continue

    #             timing = self.mapper(instr)
    #             subsystem = self.subsystems[timing.subsystem]

    #             if subsystem.next_issue_time > time:
    #                 continue

    #             # Issue instruction.
    #             warp.issued.add(instr.id)
    #             subsystem.next_issue_time = time + timing.cpi

    #             complete_time = time + timing.latency
    #             heapq.heappush(
    #                 completions,
    #                 (complete_time, next(event_counter), warp.warp_id, instr.id),
    #             )

    #             issued_count += 1
    #             issued_any = True

    #         if issued_any:
    #             continue

    #         next_time = self._next_event_time(time, completions)

    #         if next_time is None:
    #             self._dump_state(time, warps, completions)
    #             raise RuntimeError(
    #                 "Simulation deadlock: no ready instruction and no future event."
    #             )

    #         time = next_time

    #     return max(
    #         max(warp.completed.values(), default=0.0)
    #         for warp in warps
    #     )

    def run(self, warps: List[WarpState]) -> float:
        self.trace.clear()
        time = 0.0
        event_counter = itertools.count()

        warp_map = {warp.warp_id: warp for warp in warps}

        group_members = {}
        for warp in warps:
            group_members.setdefault(warp.group_id, set()).add(warp.warp_id)

        # A barrier is globally completed for a group only when all warps
        # in that group have reached it.
        barrier_waiting = {}

        # (completion_time, counter, warp_id, instruction_id)
        completions = []

        while True:
            while completions and completions[0][0] <= time:
                complete_time, _, warp_id, instr_id = heapq.heappop(completions)

                warp = warp_map[warp_id]
                instr = warp.instructions[instr_id]

                if instr.op.startswith("bar."):
                    key = (warp.group_id, instr.id)

                    if key not in barrier_waiting:
                        barrier_waiting[key] = {}

                    barrier_waiting[key][warp_id] = complete_time

                    arrived = set(barrier_waiting[key].keys())
                    required = group_members[warp.group_id]

                    if arrived == required:
                        release_time = max(barrier_waiting[key].values())

                        for waiting_warp_id in required:
                            waiting_warp = warp_map[waiting_warp_id]
                            waiting_warp.completed[instr.id] = release_time

                        del barrier_waiting[key]

                else:
                    warp.completed[instr_id] = complete_time

            if all(warp.is_done() for warp in warps):
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

                warp.issued.add(instr.id)
                subsystem.next_issue_time = time + timing.cpi

                complete_time = time + timing.latency
                self.trace.append(
                    TraceEvent(
                        group_id=warp.group_id,
                        warp_id=warp.warp_id,
                        instr_id=instr.id,
                        op=instr.op,
                        subsystem=timing.subsystem,
                        issue_time=time,
                        complete_time=complete_time,
                        deps=instr.deps,
                        raw=instr.raw,
                    )
                )

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

                print("Barrier waiting state:")
                for key, waiting in barrier_waiting.items():
                    print(f"  {key}: {waiting}")

                raise RuntimeError(
                    "Simulation deadlock: no ready instruction and no future event."
                )

            time = next_time

        return max(
            max(warp.completed.values(), default=0.0)
            for warp in warps
        )

    def _collect_candidates(self, warps: List[WarpState], time: float) -> List[Tuple[WarpState, Instruction]]:
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

    def get_trace(self) -> List[TraceEvent]:
        return list(self.trace)


    def print_trace(self, limit: Optional[int] = None) -> None:
        events = self.trace if limit is None else self.trace[:limit]

        for event in events:
            print(
                f"t={event.issue_time:8.2f} -> {event.complete_time:8.2f} | "
                f"g={event.group_id} w={event.warp_id} | "
                f"{event.instr_id:>5} {event.op:<20} | "
                f"{event.subsystem:<10} | "
                f"deps={event.deps}"
            )

    def write_trace_csv(self, path: str) -> None:
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow([
                "group_id",
                "warp_id",
                "instr_id",
                "op",
                "subsystem",
                "issue_time",
                "complete_time",
                "deps",
                "raw",
            ])

            for event in self.trace:
                writer.writerow([
                    event.group_id,
                    event.warp_id,
                    event.instr_id,
                    event.op,
                    event.subsystem,
                    event.issue_time,
                    event.complete_time,
                    " ".join(event.deps),
                    event.raw,
                ])

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



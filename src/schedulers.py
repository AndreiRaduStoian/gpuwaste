from __future__ import annotations

from typing import List, Tuple

from idg import Instruction, WarpState


# Notes from reading CUDA programming guide latest.
"""
At runtime, the GPU scheduler utilizes stream priorities to determine task execution order, but these priorities serve as hints
rather than guarantees. When selecting work to launch, pending tasks in higher-priority streams take
precedence over those in lower-priority streams. Higher-priority tasks do not preempt already running
lower-priority tasks. The GPU does not reassess work queues during task execution, and increasing a
stream’s priority will not interrupt ongoing work. Stream priorities influence task execution without
enforcing strict ordering, so users can leverage stream priorities to influence task execution without
relying on strict ordering guarantees.
"""

class Scheduler:
    """
    Receives issuable (warp, instruction) pairs and returns them
    in the order the simulator should try issuing them.
    """

    def order(
        self,
        candidates: List[Tuple[WarpState, Instruction]],
    ) -> List[Tuple[WarpState, Instruction]]:
        raise NotImplementedError


class LowestWarpFirstScheduler(Scheduler):
    def order(
        self,
        candidates: List[Tuple[WarpState, Instruction]],
    ) -> List[Tuple[WarpState, Instruction]]:
        return sorted(candidates, key=lambda item: (item[0].warp_id, item[1].id))


class RoundRobinWarpScheduler(Scheduler):
    def __init__(self):
        self.next_warp = 0

    def order(
        self,
        candidates: List[Tuple[WarpState, Instruction]],
    ) -> List[Tuple[WarpState, Instruction]]:
        if not candidates:
            return []

        max_warp = max(warp.warp_id for warp, _ in candidates)

        def key(item: Tuple[WarpState, Instruction]):
            warp, instr = item
            distance = (warp.warp_id - self.next_warp) % (max_warp + 1)
            return distance, instr.id

        ordered = sorted(candidates, key=key)

        if ordered:
            self.next_warp = (ordered[0][0].warp_id + 1) % (max_warp + 1)

        return ordered
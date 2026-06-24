import heapq
import itertools
import math

from schedulers import FIFOScheduler
from hardware import SubsystemState

def build_resident_hardware_threads(exe, idg):
    states = []
    hardware_thread_id = 0
    for group_id in range(exe.concurrent_groups_per_core):
        for _ in range(exe.hardware_threads_per_group):
            states.append(HardwareThreadState(hardware_thread_id, group_id, idg))
            hardware_thread_id += 1
    return states


def static_mapper(instr, hardware, exe):
    return hardware.timing_for_op(instr.op)


class HardwareThreadState:
    def __init__(self, hardware_thread_id, group_id, instructions):
        self.hardware_thread_id = hardware_thread_id
        self.group_id = group_id
        self.instructions = dict(instructions)
        self.issued = set()
        self.completed = {}

    def ready_instructions(self, time):
        out = []
        for instr in self.instructions.values():
            if instr.id in self.issued:
                continue
            ready = True
            for dep in instr.deps:
                if dep not in self.completed or self.completed[dep] > time:
                    ready = False
                    break
            if ready:
                out.append(instr)
        return out

    def is_done(self):
        return len(self.completed) == len(self.instructions)


class PipelineSimulator:
    def __init__(self, hardware, mapper=static_mapper, scheduler=None):
        self.hardware = hardware
        self.mapper = mapper
        self.scheduler = scheduler or FIFOScheduler()

    def run_idg(self, kernel_name, idg, exe, tracing=True):
        if tracing:
            trace = []
        states = build_resident_hardware_threads(exe, idg)
        state_by_id = {s.hardware_thread_id: s for s in states}

        group_members = {}
        for state in states:
            group_members.setdefault(state.group_id, set()).add(state.hardware_thread_id)

        subsystems = {name: SubsystemState(name) for name in self.hardware.subsystems}

        issue_limit = int(self.hardware.issue_limit_ipc)
        issue_cycle = None
        issued_this_cycle = 0

        completions = []
        event_counter = itertools.count()
        barrier_waiting = {}
        time = 0.0

        while True:
            self._complete_ready(completions, state_by_id, group_members, barrier_waiting, time)

            if all(state.is_done() for state in states):
                break

            current_cycle = int(math.floor(time))

            if issue_cycle != current_cycle:
                issue_cycle = current_cycle
                issued_this_cycle = 0

            while issued_this_cycle < issue_limit:
                candidates = self._collect_candidates(states, exe, subsystems, time)
                ordered = self.scheduler.order(candidates)

                if not ordered:
                    break

                state, instr = ordered[0]
                timing = self.mapper(instr, self.hardware, exe)
                subsystem = subsystems[timing.subsystem]

                state.issued.add(instr.id)

                subsystem.next_issue_time = time + timing.lambda_cpi
                complete_time = time + timing.completion_latency

                heapq.heappush(
                    completions,
                    (
                        complete_time,
                        next(event_counter),
                        state.hardware_thread_id,
                        instr.id,
                    ),
                )
                
                if tracing:
                    trace.append(TraceEvent(state.group_id, state.hardware_thread_id, instr.id, instr.op, timing.subsystem, time, complete_time, instr.deps, instr.raw))
                issued_this_cycle += 1

            next_time = self._next_event_time(time, completions, subsystems)

            if issued_this_cycle >= issue_limit:
                next_cycle_time = current_cycle + 1
                if next_cycle_time > time:
                    next_time = next_cycle_time if next_time is None else min(next_time, next_cycle_time)

            if next_time is None:
                self._dump_deadlock(
                    time,
                    states,
                    subsystems,
                    completions,
                    barrier_waiting,
                    None,
                )
                break

            time = next_time

        cycles = max((max(s.completed.values(), default=0.0) for s in states), default=0.0)
        if not tracing:
            trace = None
        return SimulationResult(kernel_name, exe, cycles, len(idg), trace)

    def _complete_ready(self, completions, state_by_id, group_members, barrier_waiting, time):
        while completions and completions[0][0] <= time:
            complete_time, _, hardware_thread_id, instr_id = heapq.heappop(completions)
            state = state_by_id[hardware_thread_id]
            instr = state.instructions[instr_id]

            if instr.op != "barrier":
                state.completed[instr_id] = complete_time
                continue

            key = (state.group_id, instr.id)
            waiting = barrier_waiting.setdefault(key, {})
            waiting[hardware_thread_id] = complete_time

            required = group_members[state.group_id]

            if set(waiting) == required:
                release_time = max(waiting.values())

                for waiting_id in required:
                    state_by_id[waiting_id].completed[instr.id] = release_time

                del barrier_waiting[key]

    def _collect_candidates(self, states, exe, subsystems, time):
        out = []

        for state in states:
            for instr in state.ready_instructions(time):
                timing = self.mapper(instr, self.hardware, exe)
                subsystem = subsystems[timing.subsystem]

                if subsystem.next_issue_time <= time:
                    out.append((state, instr))

        return out

    def _next_event_time(self, time, completions, subsystems):
        times = []

        if completions and completions[0][0] > time:
            times.append(completions[0][0])

        for subsystem in subsystems.values():
            if subsystem.next_issue_time > time:
                times.append(subsystem.next_issue_time)

        if not times:
            return None

        return min(times)

    def _dump_deadlock(self, time, states, subsystems, completions, barrier_waiting, next_global_issue_time):
        print("time", time)
        print("next_global_issue_time", next_global_issue_time)
        print("subsystems", subsystems)
        print("completions", completions[:10])
        print("barrier_waiting", barrier_waiting)
        for state in states[:16]:
            print(
                "thread",
                state.hardware_thread_id,
                "group",
                state.group_id,
                "issued",
                sorted(state.issued),
                "completed",
                sorted(state.completed),
                "ready",
                [i.id for i in state.ready_instructions(time)],
            )


class TraceEvent:
    def __init__(self, group_id, hardware_thread_id, instr_id, op, subsystem, issue_time, complete_time, deps=(), raw=""):
        self.group_id = group_id
        self.hardware_thread_id = hardware_thread_id
        self.instr_id = instr_id
        self.op = op
        self.subsystem = subsystem
        self.issue_time = float(issue_time)
        self.complete_time = float(complete_time)
        self.deps = tuple(deps)
        self.raw = raw

    def __repr__(self):
        return (
            f"TraceEvent("
            f"grp={self.group_id}, "
            f"ht={self.hardware_thread_id}, "
            f"instr='{self.instr_id}', "
            f"op='{self.op}', "
            f"subsys='{self.subsystem}', "
            f"issue={self.issue_time:.2f}, "
            f"done={self.complete_time:.2f}, "
            f"deps={self.deps}"
            f")"
        )


class SimulationResult:
    def __init__(self, kernel_name, execution_config, cycles, instruction_count_per_hardware_thread, trace=None):
        self.kernel_name = kernel_name
        self.execution_config = execution_config
        self.cycles = float(cycles)
        self.instruction_count_per_hardware_thread = int(instruction_count_per_hardware_thread)
        self.trace = tuple(trace) if trace is not None else None

    @property
    def total_hardware_threads(self):
        return self.execution_config.occupancy_hardware_threads

    @property
    def total_instructions(self):
        return self.total_hardware_threads * self.instruction_count_per_hardware_thread

    @property
    def hardware_threads_per_cycle(self):
        if self.cycles == 0:
            return math.inf
        return self.total_hardware_threads / self.cycles

    @property
    def instructions_per_cycle(self):
        if self.cycles == 0:
            return math.inf
        return self.total_instructions / self.cycles


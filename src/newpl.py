import argparse
import csv
import heapq
import itertools
import math
import os


# ------------------------------------------------------------
# core model objects
# ------------------------------------------------------------

class Instruction:
    def __init__(self, id, op, deps=(), raw=""):
        self.id = id
        self.op = op
        self.deps = tuple(deps)
        self.raw = raw


class InstructionTiming:
    def __init__(self, subsystem, lambda_cpi, completion_latency):
        self.subsystem = subsystem
        self.lambda_cpi = float(lambda_cpi)
        self.completion_latency = float(completion_latency)


class CoreHardwareConfig:
    def __init__(self, subsystems, issue_limit_ipc, timings):
        self.subsystems = tuple(subsystems)
        self.issue_limit_ipc = float(issue_limit_ipc)
        self.timings = dict(timings)

    @property
    def issue_lambda_cpi(self):
        return 1.0 / self.issue_limit_ipc

    def timing_for_op(self, op):
        return self.timings[op]


class ExecutionConfig:
    def __init__(self, hardware_threads_per_group, concurrent_groups_per_core):
        self.hardware_threads_per_group = int(hardware_threads_per_group)
        self.concurrent_groups_per_core = int(concurrent_groups_per_core)

    @property
    def occupancy_hardware_threads(self):
        return self.hardware_threads_per_group * self.concurrent_groups_per_core

    @property
    def occupancy_warps(self):
        return self.occupancy_hardware_threads


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


class SubsystemState:
    def __init__(self, name):
        self.name = name
        self.next_issue_time = 0.0

    def __repr__(self):
        return f"SubsystemState(name={self.name!r}, next_issue_time={self.next_issue_time})"


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


class SimulationResult:
    def __init__(self, kernel_name, execution_config, cycles, instruction_count_per_hardware_thread, trace):
        self.kernel_name = kernel_name
        self.execution_config = execution_config
        self.cycles = float(cycles)
        self.instruction_count_per_hardware_thread = int(instruction_count_per_hardware_thread)
        self.trace = tuple(trace)

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


# ------------------------------------------------------------
# schedulers
# ------------------------------------------------------------

class Scheduler:
    def order(self, candidates):
        return list(candidates)


class LowestHardwareThreadFirstScheduler(Scheduler):
    def order(self, candidates):
        return sorted(candidates, key=lambda x: (x[0].hardware_thread_id, x[1].id))


class RoundRobinHardwareThreadScheduler(Scheduler):
    def __init__(self):
        self.next_hardware_thread = 0

    def order(self, candidates):
        if not candidates:
            return []
        ids = [state.hardware_thread_id for state, _ in candidates]
        modulus = max(ids) + 1

        def key(item):
            state, instr = item
            return ((state.hardware_thread_id - self.next_hardware_thread) % modulus, instr.id)

        ordered = sorted(candidates, key=key)
        self.next_hardware_thread = (ordered[0][0].hardware_thread_id + 1) % modulus
        return ordered


# ------------------------------------------------------------
# simulator
# ------------------------------------------------------------

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


# class MyPipeline:
#     def __init__(self, hardware, mapper=static_mapper, scheduler=None):
#         self.hardware = hardware
#         self.mapper = mapper
#         self.scheduler = scheduler or LowestHardwareThreadFirstScheduler()

#     def run_idg(self, kernel_name, idg, exe):
#         trace = []
#         states = build_resident_hardware_threads(exe, idg)
#         state_by_id = {s.hardware_thread_id: s for s in states}
#         group_members = {}
#         for state in states:
#             group_members.setdefault(state.group_id, set()).add(state.hardware_thread_id)

#         global_time = 0.0
#         subsystems = {name: SubsystemState(name) for name in self.hardware.subsystems}
#         completions = []
#         event_counter = itertools.count()

#         while True:
#             # 1) Is there a ready pipeline?
#             # 2) For the ready pipelines, are there matching instruction types that are also ready?

class PipelineSimulator:
    def __init__(self, hardware, mapper=static_mapper, scheduler=None):
        self.hardware = hardware
        self.mapper = mapper
        self.scheduler = scheduler or LowestHardwareThreadFirstScheduler()

    def run_idg(self, kernel_name, idg, exe):
        trace = []
        states = build_resident_hardware_threads(exe, idg)
        state_by_id = {s.hardware_thread_id: s for s in states}

        subsystems = {name: SubsystemState(name) for name in self.hardware.subsystems}
        next_global_issue_time = 0.0
        completions = []
        event_counter = itertools.count()
        time = 0.0

        while True:
            self._complete_ready(completions, state_by_id, time)

            if all(state.is_done() for state in states):
                break

            while next_global_issue_time <= time:
                candidates = self._collect_candidates(states, exe, subsystems, time)
                ordered = self.scheduler.order(candidates)

                if not ordered:
                    break

                state, instr = ordered[0]
                timing = self.mapper(instr, self.hardware, exe)
                subsystem = subsystems[timing.subsystem]

                state.issued.add(instr.id)

                subsystem.next_issue_time = time + timing.lambda_cpi
                next_global_issue_time = time + self.hardware.issue_lambda_cpi

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

                trace.append(
                    TraceEvent(
                        state.group_id,
                        state.hardware_thread_id,
                        instr.id,
                        instr.op,
                        timing.subsystem,
                        time,
                        complete_time,
                        instr.deps,
                        instr.raw,
                    )
                )

            next_time = self._next_event_time(
                time,
                completions,
                subsystems,
                next_global_issue_time,
            )

            if next_time is None:
                self._dump_deadlock(
                    time,
                    states,
                    subsystems,
                    completions,
                    {},
                    next_global_issue_time,
                )
                break

            time = next_time

        cycles = max((max(s.completed.values(), default=0.0) for s in states), default=0.0)
        return SimulationResult(kernel_name, exe, cycles, len(idg), trace)

    def _complete_ready(self, completions, state_by_id, time):
        while completions and completions[0][0] <= time:
            complete_time, _, hardware_thread_id, instr_id = heapq.heappop(completions)
            state = state_by_id[hardware_thread_id]
            state.completed[instr_id] = complete_time

    def _collect_candidates(self, states, exe, subsystems, time):
        out = []

        for state in states:
            for instr in state.ready_instructions(time):
                timing = self.mapper(instr, self.hardware, exe)
                subsystem = subsystems[timing.subsystem]

                if subsystem.next_issue_time <= time:
                    out.append((state, instr))

        return out

    def _next_event_time(self, time, completions, subsystems, next_global_issue_time):
        times = []

        if completions and completions[0][0] > time:
            times.append(completions[0][0])

        for subsystem in subsystems.values():
            if subsystem.next_issue_time > time:
                times.append(subsystem.next_issue_time)

        if next_global_issue_time > time:
            times.append(next_global_issue_time)

        if not times:
            return None

        return min(times)

# class PipelineSimulator:
#     def __init__(self, hardware, mapper=static_mapper, scheduler=None):
#         self.hardware = hardware
#         self.mapper = mapper
#         self.scheduler = scheduler or LowestHardwareThreadFirstScheduler()

#     def run_idg(self, kernel_name, idg, exe):
#         trace = []
#         states = build_resident_hardware_threads(exe, idg)
#         state_by_id = {s.hardware_thread_id: s for s in states}
#         group_members = {}
#         for state in states:
#             group_members.setdefault(state.group_id, set()).add(state.hardware_thread_id)

#         subsystems = {name: SubsystemState(name) for name in self.hardware.subsystems}
#         next_global_issue_time = 0.0
#         completions = []
#         event_counter = itertools.count()
#         barrier_waiting = {}
#         time = 0.0


#         while True:
#             while completions and completions[0][0] <= time:
#                 complete_time, _, hardware_thread_id, instr_id = heapq.heappop(completions)
#                 state = state_by_id[hardware_thread_id]
#                 instr = state.instructions[instr_id]

#                 # if instr.op == "barrier
#                 #     key = (state.group_id, instr.id)
#                 #     waiting = barrier_waiting.setdefault(key, {})
#                 #     waiting[hardware_thread_id] = complete_time
#                 #     required = group_members[state.group_id]

#                 #     if set(waiting) == required:
#                 #         release_time = max(waiting.values())
#                 #         for waiting_id in required:
#                 #             state_by_id[waiting_id].completed[instr.id] = release_time
#                 #         del barrier_waiting[key]
#                 # else:
#                 state.completed[instr_id] = complete_time

#             if all(state.is_done() for state in states):
#                 break

#             candidates = self._collect_candidates(states, exe, subsystems, time, next_global_issue_time)
#             ordered = self.scheduler.order(candidates)

#             if ordered:
#                 state, instr = ordered[0]
#                 timing = self.mapper(instr, self.hardware, exe)
#                 subsystem = subsystems[timing.subsystem]

#                 state.issued.add(instr.id)
#                 subsystem.next_issue_time = time + timing.lambda_cpi
#                 next_global_issue_time = time + self.hardware.issue_lambda_cpi

#                 complete_time = time + timing.completion_latency
#                 heapq.heappush(completions, (complete_time, next(event_counter), state.hardware_thread_id, instr.id))
#                 trace.append(TraceEvent(state.group_id, state.hardware_thread_id, instr.id, instr.op, timing.subsystem, time, complete_time, instr.deps, instr.raw))

#             next_time = self._next_event_time(time, completions, subsystems, next_global_issue_time)
#             if next_time is None:
#                 self._dump_deadlock(time, states, subsystems, completions, barrier_waiting, next_global_issue_time)
#                 break
#             time = next_time

#         cycles = max((max(s.completed.values(), default=0.0) for s in states), default=0.0)
#         return SimulationResult(kernel_name, exe, cycles, len(idg), trace)

#     def _collect_candidates(self, states, exe, subsystems, time, next_global_issue_time):
#         if next_global_issue_time > time:
#             return []
#         out = []
#         for state in states:
#             for instr in state.ready_instructions(time):
#                 timing = self.mapper(instr, self.hardware, exe)
#                 subsystem = subsystems[timing.subsystem]
#                 if subsystem.next_issue_time <= time:
#                     out.append((state, instr))
#         return out

#     def _next_event_time(self, time, completions, subsystems, next_global_issue_time):
#         times = []
#         if completions and completions[0][0] > time:
#             times.append(completions[0][0])
#         if next_global_issue_time > time:
#             times.append(next_global_issue_time)
#         for subsystem in subsystems.values():
#             if subsystem.next_issue_time > time:
#                 times.append(subsystem.next_issue_time)
#         if not times:
#             return None
#         return min(times)

    def _dump_deadlock(self, time, states, subsystems, completions, barrier_waiting, next_global_issue_time):
        print("time", time)
        print("next_global_issue_time", next_global_issue_time)
        print("subsystems", subsystems)
        print("completions", completions[:10])
        print("barrier_waiting", barrier_waiting)
        for state in states[:16]:
            print("thread", state.hardware_thread_id, "group", state.group_id, "issued", sorted(state.issued), "completed", sorted(state.completed), "ready", [i.id for i in state.ready_instructions(time)])


# ------------------------------------------------------------
# hardware preset
# ------------------------------------------------------------

PAPER_TOY_HARDWARE = CoreHardwareConfig(
    subsystems=("compute", "memory", "barrier"),
    issue_limit_ipc=1.0,
    timings={
        "compute": InstructionTiming("compute", 1.0, 4.0),
        "memory": InstructionTiming("memory", 2.0, 6.0),
        "barrier": InstructionTiming("barrier", 1.0, 1.0),
    },
)


# ------------------------------------------------------------
# test kernel
# ------------------------------------------------------------

def make_fig1_toy_idg():
    return {
        "c0": Instruction("c0", "compute"),
        "m0": Instruction("m0", "memory", deps=("c0",)),
        "c1": Instruction("c1", "compute", deps=("m0",)),
        "c2": Instruction("c2", "compute", deps=("m0",)),
        "c3": Instruction("c3", "compute", deps=("c1", "c2")),
        "m1": Instruction("m1", "memory", deps=("c3",)),
    }


# ------------------------------------------------------------
# experiments and IO
# ------------------------------------------------------------

def sweep_occupancy(kernel_name, idg, hardware, hardware_threads_per_group_values, concurrent_groups_per_core_values, scheduler_factory=RoundRobinHardwareThreadScheduler):
    results = []
    for htg in hardware_threads_per_group_values:
        for groups in concurrent_groups_per_core_values:
            exe = ExecutionConfig(htg, groups)
            sim = PipelineSimulator(hardware, scheduler=scheduler_factory())
            results.append(sim.run_idg(kernel_name, idg, exe))
    return results


def write_results_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "hardware_threads_per_group", "concurrent_groups_per_core", "occupancy_hardware_threads", "instructions_per_hardware_thread", "cycles", "hardware_threads_per_cycle", "instructions_per_cycle"])
        for r in results:
            e = r.execution_config
            writer.writerow([r.kernel_name, e.hardware_threads_per_group, e.concurrent_groups_per_core, e.occupancy_hardware_threads, r.instruction_count_per_hardware_thread, r.cycles, r.hardware_threads_per_cycle, r.instructions_per_cycle])


def write_trace_csv(result, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "hardware_thread_id", "instr_id", "op", "subsystem", "issue_time", "complete_time", "deps", "raw"])
        for event in result.trace:
            writer.writerow([event.group_id, event.hardware_thread_id, event.instr_id, event.op, event.subsystem, event.issue_time, event.complete_time, " ".join(event.deps), event.raw])


def print_results(results):
    header = f"{'kernel':<24} {'ht/g':>5} {'groups':>6} {'occ':>5} {'instr/ht':>8} {'cycles':>10} {'HT/cyc':>10} {'instr/cyc':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        e = r.execution_config
        print(f"{r.kernel_name:<24} {e.hardware_threads_per_group:>5} {e.concurrent_groups_per_core:>6} {e.occupancy_hardware_threads:>5} {r.instruction_count_per_hardware_thread:>8} {r.cycles:>10.2f} {r.hardware_threads_per_cycle:>10.4f} {r.instructions_per_cycle:>10.4f}")


def run_default_experiments():
    return sweep_occupancy(
        "fig1_toy",
        make_fig1_toy_idg(),
        PAPER_TOY_HARDWARE,
        [1],
        [1, 2, 4, 10],
    )


if __name__ == "__main__":
    print_results(run_default_experiments())
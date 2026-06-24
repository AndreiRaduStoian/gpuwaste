# Defines hardware config and includes all configs used in thesis.

class InstructionTiming:
    def __init__(self, subsystem, lambda_cpi, completion_latency):
        self.subsystem = subsystem
        self.lambda_cpi = float(lambda_cpi)
        self.completion_latency = float(completion_latency)


class SubsystemState:
    def __init__(self, name):
        self.name = name
        self.next_issue_time = 0.0

    def __repr__(self):
        return f"SubsystemState(name={self.name!r}, next_issue_time={self.next_issue_time})"


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


# For unit tests
PAPER_TOY_HARDWARE = CoreHardwareConfig(
    subsystems=("compute", "memory"),
    issue_limit_ipc=4.0,
    timings={
        "compute": InstructionTiming("compute", 1.0, 4.0),
        "memory": InstructionTiming("memory", 2.0, 6.0),
    },
)

BARRIER_SMOKE_HARDWARE = CoreHardwareConfig(
    subsystems=("compute", "barrier"),
    issue_limit_ipc=4.0,
    timings={
        "compute": InstructionTiming("compute", 1.0, 4.0),
        "barrier": InstructionTiming("barrier", 1.0, 1.0),
    },
)


# For the plot like fig16, we chose mul.s32 as the compute op for these two.
FERMI_BARRIER_HARDWARE = CoreHardwareConfig(
    subsystems=("compute", "barrier"),
    issue_limit_ipc=1.0,
    timings={
        "compute": InstructionTiming("compute", 1.0, 18.0),
        "barrier": InstructionTiming("barrier", 2.0, 40.0),
    },
)
PASCAL_BARRIER_HARDWARE = CoreHardwareConfig(
    subsystems=("compute", "barrier"),
    issue_limit_ipc=4.0,
    timings={
        "compute": InstructionTiming("compute", 0.25, 6.0),
        "barrier": InstructionTiming("barrier", 2.25, 70.0),
    },
)


# For comparison and scaling
RTX_3070_CALIBRATION_HARDWARE = CoreHardwareConfig(
    # ampere GA104 has four warp schedulers per SM.
    issue_limit_ipc = 4.0,
    subsystems=("alu", "int_alu", "global_mem", "local_mem", "barrier", "sfu"),
    # rounded to 1/IL as per paper.
    timings={
        "fma_f32": InstructionTiming("alu", 0.25, 1.00),         # MADD/1
        "add_f32": InstructionTiming("alu", 0.50, 1.75),         # SP/1
        "int_alu": InstructionTiming("int_alu", 0.25, 1.00),     # IMADD/1
        "sfu":     InstructionTiming("sfu", 4.00, 6.50),         # SF/1

        # issue interval from RTX 3070 Global/Float/MainMemory.
        # completion latency from Global/Float/CacheLevel2
        "global_mem": InstructionTiming("global_mem", 14.50, 231.00),

        # barrier(sync) not measured by microbench.
        # no results for local mem in microbench. (except a single completion latency measurement of 27c roughly in line with the rest in Jan Lemeire paper)
        # following are from turing 
        "local_mem": InstructionTiming("local_mem", 2.00, 32.00),
        "barrier": InstructionTiming("barrier", 1.50, 17.00),
    },
)
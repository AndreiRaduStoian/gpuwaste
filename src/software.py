from hardware import ExecutionConfig

class Instruction:
    def __init__(self, id, op, deps=(), raw=""):
        self.id = id
        self.op = op
        self.deps = tuple(deps)
        self.raw = raw


def make_fig1_toy_idg():
    return {
        "c0": Instruction("c0", "compute"),
        "m0": Instruction("m0", "memory", deps=("c0",)),
        "c1": Instruction("c1", "compute", deps=("m0",)),
        "c2": Instruction("c2", "compute", deps=("m0",)),
        "c3": Instruction("c3", "compute", deps=("c1", "c2")),
        "m1": Instruction("m1", "memory", deps=("c3",)),
    }

def make_barrier_smoke_idg():
    return {
        "c0": Instruction("c0", "compute"),
        "b0": Instruction("b0", "barrier", deps=("c0",)),
        "c1": Instruction("c1", "compute", deps=("b0",)),
    }

def make_iterative_barrier_idg(iterations=256, include_final_barrier=False):
    idg = {}
    previous = None

    for i in range(iterations):
        instr_id = f"i{i:04d}"
        deps = (previous,) if previous is not None else ()
        idg[instr_id] = Instruction(instr_id, "compute", deps=deps)
        previous = instr_id

        if include_final_barrier or i != iterations - 1:
            barrier_id = f"b{i:04d}"
            idg[barrier_id] = Instruction(barrier_id, "barrier", deps=(previous,))
            previous = barrier_id

    return idg


def make_compute_fma_idg(iterations):
    idg = {}

    x_prev = None
    y_prev = None
    z_prev = None

    for i in range(iterations):
        x = f"it{i:04d}_x_fma"
        y = f"it{i:04d}_y_fma"
        z = f"it{i:04d}_z_fma"

        x_deps = tuple(d for d in (x_prev, y_prev, z_prev) if d is not None)
        idg[x] = Instruction(x, "fma_f32", deps=x_deps)

        y_deps = tuple(d for d in (y_prev, z_prev, x) if d is not None)
        idg[y] = Instruction(y, "fma_f32", deps=y_deps)

        z_deps = tuple(d for d in (z_prev, x, y) if d is not None)
        idg[z] = Instruction(z, "fma_f32", deps=z_deps)

        x_prev, y_prev, z_prev = x, y, z

    # out[tid] = x + y + z
    add0 = "final_add_xy"
    add1 = "final_add_xyz"
    st = "final_store_out"

    idg[add0] = Instruction(add0, "add_f32", deps=(x_prev, y_prev))
    idg[add1] = Instruction(add1, "add_f32", deps=(add0, z_prev))
    idg[st] = Instruction(st, "global_mem", deps=(add1,))

    return idg


def make_vector_add_idg(iterations):
    return {
        "ld_a": Instruction("ld_a", "global_mem"),
        "ld_b": Instruction("ld_b", "global_mem"),
        "add":  Instruction("add", "add_f32", deps=("ld_a", "ld_b")),
        "st_c": Instruction("st_c", "global_mem", deps=("add",)),
    }


def make_shared_barrier_idg(iterations):
    idg = {}

    idg["ld_in"] = Instruction("ld_in", "global_mem")
    idg["st_tile_init"] = Instruction("st_tile_init", "local_mem", deps=("ld_in",))
    idg["b_init"] = Instruction("b_init", "barrier", deps=("st_tile_init",))

    prev_barrier = "b_init"

    for i in range(iterations):
        ld_l = f"it{i:04d}_ld_left"
        ld_m = f"it{i:04d}_ld_mid"
        ld_r = f"it{i:04d}_ld_right"
        b_read = f"it{i:04d}_b_after_reads"

        # Three shared-memory reads.
        idg[ld_l] = Instruction(ld_l, "local_mem", deps=(prev_barrier,))
        idg[ld_m] = Instruction(ld_m, "local_mem", deps=(prev_barrier,))
        idg[ld_r] = Instruction(ld_r, "local_mem", deps=(prev_barrier,))

        # Barrier after all threads have read old tile values.
        idg[b_read] = Instruction(b_read, "barrier", deps=(ld_l, ld_m, ld_r))

        # Approximate weighted expression as three FP operations.
        # Compiler may emit mul/fma combinations; fma_f32 is close enough here.
        f0 = f"it{i:04d}_mul_left"
        f1 = f"it{i:04d}_fma_mid"
        f2 = f"it{i:04d}_fma_right"
        st = f"it{i:04d}_st_tile"
        b_write = f"it{i:04d}_b_after_write"

        idg[f0] = Instruction(f0, "fma_f32", deps=(b_read,))
        idg[f1] = Instruction(f1, "fma_f32", deps=(f0,))
        idg[f2] = Instruction(f2, "fma_f32", deps=(f1,))
        idg[st] = Instruction(st, "local_mem", deps=(f2,))

        # Barrier after all threads have written new tile values.
        idg[b_write] = Instruction(b_write, "barrier", deps=(st,))

        prev_barrier = b_write

    idg["ld_tile_final"] = Instruction("ld_tile_final", "local_mem", deps=(prev_barrier,))
    idg["st_out"] = Instruction("st_out", "global_mem", deps=("ld_tile_final",))

    return idg


def make_pointer_chase_idg(iterations):
    idg = {}
    prev = None
    for i in range(iterations):
        instr_id = f"it{i:04d}_ld_next"
        deps = (prev,) if prev else ()
        idg[instr_id] = Instruction(instr_id, "global_mem", deps=deps)
        prev = instr_id
    idg["st_out"] = Instruction("st_out", "global_mem", deps=(prev,))
    return idg


def make_sfu_idg(iterations):
    idg = {}
    prev = None
    for i in range(iterations):
        sin_id = f"it{i:04d}_sinf"
        cos_id = f"it{i:04d}_cosf"
        mul_id = f"it{i:04d}_x2"
        add_id = f"it{i:04d}_add1"
        rsq_id = f"it{i:04d}_rsqrtf"

        idg[sin_id] = Instruction(sin_id, "sfu", deps=(prev,) if prev else ())
        idg[cos_id] = Instruction(cos_id, "sfu", deps=(sin_id,))
        idg[mul_id] = Instruction(mul_id, "fma_f32", deps=(cos_id,))
        idg[add_id] = Instruction(add_id, "add_f32", deps=(mul_id,))
        idg[rsq_id] = Instruction(rsq_id, "sfu", deps=(add_id,))
        prev = rsq_id

    idg["st_out"] = Instruction("st_out", "global_mem", deps=(prev,))
    return idg




def cuda_launch_to_pipeline_config(n, block_size, resident_blocks_per_sm, warp_size=32):

    def ceil_div(a,b):
        return (a + b - 1) // b

    grid_size = ceil_div(n, block_size)
    warps_per_block = ceil_div(block_size, warp_size)

    exe = ExecutionConfig(
        hardware_threads_per_group=warps_per_block,
        concurrent_groups_per_core=resident_blocks_per_sm,
    )

    return grid_size, exe
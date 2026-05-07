import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))
print(SRC_ROOT)

from ptx_parser import build_idg_from_ptx
from schedulers import RoundRobinWarpScheduler
from pipelinesim import PipelineSimulator, ExecutionConfig, build_warps_from_idg
from hardware import TOY_PTX_HARDWARE, make_ptx_static_mapper


PTX_FILE = PROJECT_ROOT / "benchmarks" / "ptx" / "hotspot.ptx"
TRACE_FILE = PROJECT_ROOT / "outputs" / "traces" / "hotspot_trace.csv"

TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)


with open(PTX_FILE, "r", encoding="utf-8") as f:
    ptx_text = f.read()


idg = build_idg_from_ptx(ptx_text)

print(f"Loaded PTX file: {PTX_FILE}")
print(f"IDG nodes: {len(idg)}")


exe = ExecutionConfig(
    group_size=4,
    num_groups=1,
)

warps = build_warps_from_idg(exe, idg)

print(f"group_size |γ|: {exe.group_size}")
print(f"num_groups γ: {exe.num_groups}")
print(f"occupancy ω: {exe.occupancy}")


sim = PipelineSimulator(
    hardware=TOY_PTX_HARDWARE,
    mapper=make_ptx_static_mapper(TOY_PTX_HARDWARE),
    scheduler=RoundRobinWarpScheduler(),
    execution_config=exe,
)


cycles = sim.run(warps)

print(f"Runtime: {cycles} cycles")
sim.print_trace(limit=50)
sim.write_trace_csv(str(TRACE_FILE))
print(f"Wrote trace to {TRACE_FILE}")
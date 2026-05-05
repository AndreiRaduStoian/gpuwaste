import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from idg import make_warp_from_idg
from ptx_parser import build_idg_from_ptx
from schedulers import RoundRobinWarpScheduler
from pipelinesim import PipelineSimulator
from hardware import TOY_PTX_HARDWARE, make_ptx_static_mapper


PTX_FILE = "hotspot.ptx"


with open(PTX_FILE, "r", encoding="utf-8") as f:
    ptx_text = f.read()


idg = build_idg_from_ptx(ptx_text)

print(f"Loaded PTX file: {PTX_FILE}")
print(f"IDG nodes: {len(idg)}")


warps = [
    make_warp_from_idg(warp_id=i, group_id=0, idg=idg)
    for i in range(4)
]


sim = PipelineSimulator(
    hardware=TOY_PTX_HARDWARE,
    mapper=make_ptx_static_mapper(TOY_PTX_HARDWARE),
    scheduler=RoundRobinWarpScheduler(),
)


cycles = sim.run(warps)

print(f"Runtime: {cycles} cycles")

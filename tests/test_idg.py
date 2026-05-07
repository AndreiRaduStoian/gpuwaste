import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))


from idg import Instruction, build_idg_from_def_use


def test_register_def_use_dependency():
    instrs = [
        Instruction("i0", "alu", writes=("%r1",)),
        Instruction("i1", "alu", reads=("%r1",), writes=("%r2",)),
    ]

    idg = build_idg_from_def_use(instrs)

    assert idg["i0"].deps == ()
    assert idg["i1"].deps == ("i0",)


def test_barrier_orders_regions():
    instrs = [
        Instruction("i0", "alu"),
        Instruction("i1", "alu"),
        Instruction("b0", "bar.sync"),
        Instruction("i2", "alu"),
    ]

    idg = build_idg_from_def_use(instrs)

    assert set(idg["b0"].deps) == {"i0", "i1"}
    assert "b0" in idg["i2"].deps


def test_barrier_orders_regions_with_deps():
    instrs = [
        Instruction("i0", "alu", writes=("%r1",)),
        Instruction("i1", "alu", reads=("%r1",), writes=("%r2",)),
        Instruction("b0", "bar.sync"),
        Instruction("i2", "alu", reads=("%r2",)),
    ]

    idg = build_idg_from_def_use(instrs)

    assert set(idg["b0"].deps) == {"i0", "i1"}
    assert set(idg["i2"].deps) == {"i1", "b0"}


test_register_def_use_dependency()
test_barrier_orders_regions()
test_barrier_orders_regions_with_deps()
print("done")
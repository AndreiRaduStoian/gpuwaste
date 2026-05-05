import re
from typing import List, Tuple, Optional

from idg import Instruction, build_idg_from_def_use


REGISTER_RE = re.compile(r"%[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+)?")
PREDICATE_RE = re.compile(r"^(@!?%[a-zA-Z_][a-zA-Z0-9_]*)(?:\s+)(.*)$")


def split_predicate(line: str) -> Tuple[Optional[str], str]:
    """
    Handles:
        @%p10 bra $L__BB0_2;
        @!%p1 bra label;
    """
    match = PREDICATE_RE.match(line)

    if not match:
        return None, line

    predicate = match.group(1)
    rest = match.group(2).strip()

    return predicate, rest


def strip_comment(line: str) -> str:
    return line.split("//", 1)[0].strip()


def is_ignored_line(line: str) -> bool:
    if not line:
        return True

    # PTX directives / declarations / labels / braces
    if line.startswith("."):
        return True

    if line.endswith(":"):
        return True

    if line in {"{", "}"}:
        return True

    return False


def split_opcode_and_operands(line: str) -> Tuple[str, str]:
    line = line.rstrip(";").strip()

    parts = line.split(None, 1)
    if len(parts) == 1:
        return parts[0], ""

    return parts[0], parts[1]


def split_operands(operand_text: str) -> List[str]:
    operands = []
    current = []
    bracket_depth = 0

    for ch in operand_text:
        if ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1

        if ch == "," and bracket_depth == 0:
            operands.append("".join(current).strip())
            current = []
        else:
            current.append(ch)

    if current:
        operands.append("".join(current).strip())

    return operands


def registers_in(text: str) -> Tuple[str, ...]:
    return tuple(REGISTER_RE.findall(text))


def classify_reads_writes(op, operands):
    if not operands:
        return (), ()

    if op.startswith("ld."):
        writes = registers_in(operands[0])
        reads = tuple(
            reg
            for operand in operands[1:]
            for reg in registers_in(operand)
        )
        return reads, writes

    if op.startswith("st."):
        reads = tuple(
            reg
            for operand in operands
            for reg in registers_in(operand)
        )
        return reads, ()

    if op.startswith("bar."):
        return (), ()

    if op == "bra":
        reads = tuple(
            reg
            for operand in operands
            for reg in registers_in(operand)
        )
        return reads, ()

    if op == "ret":
        return (), ()

    # destination-first default
    writes = registers_in(operands[0])
    reads = tuple(
        reg
        for operand in operands[1:]
        for reg in registers_in(operand)
    )

    return reads, writes


def parse_ptx_instruction(line: str, instr_id: str) -> Instruction:
    predicate, line_without_predicate = split_predicate(line)

    op, operand_text = split_opcode_and_operands(line_without_predicate)
    operands = split_operands(operand_text)

    reads, writes = classify_reads_writes(op, operands)

    if predicate is not None:
        pred_reg = predicate.replace("@!", "").replace("@", "")
        reads = (pred_reg,) + reads

    return Instruction(
        id=instr_id,
        op=op,
        reads=reads,
        writes=writes,
        raw=line,
        predicate=predicate,
    )


def parse_ptx_to_instruction_list(ptx_text: str) -> List[Instruction]:
    instructions = []
    counter = 0

    for raw_line in ptx_text.splitlines():
        line = strip_comment(raw_line)

        if is_ignored_line(line):
            continue

        if line.startswith(".entry") or line.startswith(".func"):
            continue

        if line.startswith("(") or line.startswith(")"):
            continue

        instr = parse_ptx_instruction(line, f"i{counter}")
        instructions.append(instr)
        counter += 1

    return instructions


def build_idg_from_ptx(ptx_text: str):
    instructions = parse_ptx_to_instruction_list(ptx_text)
    return build_idg_from_def_use(instructions)


if __name__ == "__main__":
    with open("hotspot.ptx", "r", encoding="utf-8") as f:
        ptx = f.read()

    instrs = parse_ptx_to_instruction_list(ptx)

    print("instruction count:", len(instrs))
    print()

    print("first 30 instructions:")
    for instr in instrs[:30]:
        print(
            instr.id,
            instr.op,
            "pred=", instr.predicate,
            "reads=", instr.reads,
            "writes=", instr.writes,
        )

    print()
    print("barriers:")
    for instr in instrs:
        if instr.op.startswith("bar."):
            print(instr.id, instr.raw)

    print()
    idg = build_idg_from_ptx(ptx)
    print("IDG node count:", len(idg))

    print()
    print("first 30 IDG nodes:")
    for instr_id, instr in list(idg.items())[:30]:
        print(instr_id, instr.op, "deps=", instr.deps)
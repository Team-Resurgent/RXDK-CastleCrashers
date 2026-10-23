#!/usr/bin/env python3
"""List the functions that must keep their registers in the context.

  context_functions.py [--write]

XenonRecomp keeps r0, r2, r11, r12, r14-r31, f0 and f14-f31 in locals of each
function (non_argument_as_local, non_volatile_as_local in
config/castlecrashers.toml): the compiler can then hold them in the
processor's registers. A function that reads one of them before writing it
expects its caller's value there (the compiler's __finally funclets read
their parent's frame pointer, some helpers take an argument in r12): it keeps
its registers in the context, and so do its callers, which then leave their
values there. Reads through a call count: a function calling one that reads
r31 first, without writing r31 itself, reads it first too.

The scan reads the instructions (the comments in build/ppc, the same whether
the registers are locals or not), in order, ignoring control flow; saves of
a register to the stack (a store based on r1) are not reads of it. With
--write the list replaces context_register_functions in the configuration.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPC = ROOT / "build/ppc"
CONFIG = ROOT / "config/castlecrashers.toml"
TRACKED = {f"r{n}" for n in [0, 2, 11, 12] + list(range(14, 32))} | {f"f{n}" for n in [0] + list(range(14, 32))}
# Functions tools/postprocess.py patches by the context's register names.
PATCHED = {"sub_82092E28", "sub_820D4030"}

FUNC = re.compile(r"PPC_FUNC_IMPL\(__imp__(sub_[0-9A-F]+)\) \{(.*?)\n\}\n", re.S)
COMMENT = re.compile(r"^\t// (\S+) ?(.*)$", re.M)
REG = re.compile(r"\b([rf]\d+)\b")
MEM = re.compile(r"(-?\w+)\((r\d+)\)")


def helpers(only: str = "") -> set[int]:
    """The register save and restore helpers' entry points (their calls are
    left out with the registers in locals), or only those named with only."""
    text = CONFIG.read_text(encoding="utf-8-sig")
    out = set()
    for name, count, step in [("restgprlr_14", 18, 4), ("savegprlr_14", 18, 4), ("restfpr_14", 18, 4),
                              ("savefpr_14", 18, 4), ("restvmx_14", 18, 8), ("savevmx_14", 18, 8),
                              ("restvmx_64", 64, 8), ("savevmx_64", 64, 8)]:
        m = re.search(rf"^{name}_address = (0x[0-9A-Fa-f]+)", text, re.M)
        if m and only in name:
            out |= {int(m.group(1), 16) + i * step for i in range(count)}
    return out


def events(body: str, start: int, end: int, skip: set[int]):
    """(kind, value) in order: ('read', reg), ('write', reg), ('call', name)."""
    for op, args in COMMENT.findall(body):
        base = op.rstrip(".")
        if base in ("bl", "b") or (base.startswith("b") and "0x" in args and base not in ("bctr", "bctrl")):
            m = re.search(r"0x([0-9a-f]+)", args)
            if m:
                target = int(m.group(1), 16)
                outside = not (start <= target < end)
                if (base == "bl" or outside) and target not in skip:
                    yield ("call", f"sub_{target:08X}")
            continue
        operands = [a.strip() for a in args.split(",")] if args else []
        # r0 as a memory operation's base is 0, not the register.
        memory = base.startswith(("l", "st", "dcb", "icb", "tw", "td")) or base.endswith("x")
        regs = lambda text: [r for r in REG.findall(text) if r in TRACKED and not (memory and r == "r0")]
        if base.startswith("st") or base in ("dcbz", "dcbt", "dcbtst", "dcbf", "dcbst", "icbi"):
            # A store reads its source and address; to the stack, the source
            # is a save. Update forms write their base.
            mem = MEM.search(args)
            to_stack = bool(mem and mem.group(2) == "r1")
            for i, a in enumerate(operands):
                for r in regs(a):
                    if i == 0 and to_stack and base.startswith("st"):
                        continue
                    yield ("read", r)
            if mem and base.endswith("u") or base.endswith("ux"):
                for r in regs(mem.group(2) if mem else operands[1] if len(operands) > 1 else ""):
                    yield ("write", r)
            continue
        if base.startswith(("cmp", "fcmp", "tw", "td", "mt", "cr", "sync", "isync", "eieio", "db")):
            for a in operands:
                for r in regs(a):
                    yield ("read", r)
            continue
        if base == "lmw":
            n = int(operands[0][1:])
            for r in regs(operands[1]):
                yield ("read", r)
            for k in range(n, 32):
                yield ("write", f"r{k}")
            continue
        if not operands:
            continue
        # The first operand is the destination: the rest are read first
        # (rlwimi and the update loads read it too).
        for a in operands[1:]:
            for r in regs(a):
                yield ("read", r)
        dest = regs(operands[0])
        if base.startswith("rlwimi") or base.startswith("rldimi"):
            for r in dest:
                yield ("read", r)
        for r in dest:
            yield ("write", r)
        if base.startswith("l") and (base.endswith("u") or base.endswith("ux")):
            mem = MEM.search(args)
            for r in regs(mem.group(2) if mem else operands[1] if len(operands) > 1 else ""):
                yield ("write", r)


def main() -> int:
    skip = helpers()
    funcs: dict[str, list] = {}
    for path in sorted(PPC.glob("ppc_recomp.*.cpp")):
        text = path.read_text(errors="replace")
        for m in FUNC.finditer(text):
            name, body = m.group(1), m.group(2)
            start = int(name[4:], 16)
            # The function's end: past its last instruction's comment address
            # is unknown here, so branches are local when they land on one of
            # its labels.
            labels = {int(x, 16) for x in re.findall(r"^loc_([0-9A-F]+):", body, re.M)}
            evs = []
            for kind, value in events(body, start, start + 1, skip):
                if kind == "call" and int(value[4:], 16) in labels:
                    continue
                evs.append((kind, value))
            funcs[name] = evs
    # Registers read before written, through calls, to a fixed point.
    first: dict[str, set[str]] = {f: set() for f in funcs}
    changed = True
    while changed:
        changed = False
        for f, evs in funcs.items():
            written, reads = set(), set()
            for kind, value in evs:
                if kind == "read":
                    if value not in written:
                        reads.add(value)
                elif kind == "write":
                    written.add(value)
                else:
                    reads |= first.get(value, set()) - written
            if reads - first[f]:
                first[f] |= reads
                changed = True
    readers = {f for f, regs in first.items() if regs}
    callers = {f for f, evs in funcs.items() for kind, value in evs if kind == "call" and value in readers}
    # The save and restore helpers too: the functions that keep the context
    # call them to save and restore its registers.
    # The vector registers stay in the context (vector_as_local = false), so
    # __savevmx_* and __restvmx_* are still called, with their address in
    # r11 and r12: their callers leave it in the context.
    vmx = helpers("vmx")
    vmx_callers = set()
    for path in sorted(PPC.glob("ppc_recomp.*.cpp")):
        for m in FUNC.finditer(path.read_text(errors="replace")):
            targets = {int(x, 16) for x in re.findall(r"^\t// bl? 0x([0-9a-f]+)$", m.group(2), re.M)}
            if targets & vmx:
                vmx_callers.add(m.group(1))
    keep = sorted(readers | callers | vmx_callers | PATCHED | {f"sub_{a:08X}" for a in skip})
    print(f"context_functions: {len(readers)} read a register first, {len(keep)} keep the context "
          f"of {len(funcs)}")
    if "--write" in sys.argv:
        text = CONFIG.read_text(encoding="utf-8-sig")
        block = "context_register_functions = [\n" + "".join(f"    0x{f[4:]},\n" for f in keep) + "]\n"
        if "context_register_functions = [" in text:
            text = re.sub(r"context_register_functions = \[\n(?:    0x[0-9A-F]+,\n)*\]\n", block, text)
        else:
            text = text.replace("\n\n# Functions that end in a jump table", "\n" + block + "\n# Functions that end in a jump table", 1)
        CONFIG.write_text(text, encoding="utf-8")
    else:
        for f in sorted(readers)[:40]:
            print(f, " ".join(sorted(first[f])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

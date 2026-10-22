#!/usr/bin/env python3
"""Function boundaries for functions that end in a jump table.

  find_switch_funcs.py <image.bin> <switch tables .toml> <recomp log>

XenonAnalyse ends a function at its bctr, so the cases of a jump table land
outside it ("Switch case ... is trying to jump outside function"). For each
such switch this walks forward from the function start, following every
branch and case label, to the return or tail branch past the furthest one,
and prints the `functions` entries for the recompiler's TOML.
"""
from __future__ import annotations

import re
import struct
import sys
import tomllib

BASE = 0x82000000


def main() -> int:
    image = open(sys.argv[1], "rb").read()
    tables = tomllib.load(open(sys.argv[2], "rb"))["switch"]
    labels = {t["base"]: t["labels"] for t in tables}
    log = open(sys.argv[3]).read()
    bad = sorted({int(a, 16) for a in re.findall(r"Switch case at ([0-9A-F]+) is trying", log)})
    mapping = sorted(int(a, 16) for a in re.findall(r"\{ 0x([0-9A-F]+),", open("build/ppc/ppc_func_mapping.cpp").read()))

    def word(addr: int) -> int:
        return struct.unpack_from(">I", image, addr - BASE)[0]

    entries = []
    for switch in bad:
        start = max(a for a in mapping if a <= switch)
        reach = switch
        addr = start
        while True:
            w = word(addr)
            op = w >> 26
            if addr in labels:
                reach = max(reach, *labels[addr])
            target = None
            if op == 18 and not (w & 3):  # b
                off = w & 0x03FFFFFC
                target = addr + (off - 0x04000000 if off & 0x02000000 else off)
            elif op == 16 and not (w & 3):  # bc
                off = w & 0xFFFC
                target = addr + (off - 0x10000 if off & 0x8000 else off)
            if target is not None and start <= target < start + 0x10000:
                reach = max(reach, target)
            if addr >= reach and (w == 0x4E800020 or (op == 18 and not (w & 1))):
                break
            addr += 4
        entries.append((start, addr + 4 - start))
    # Merge overlapping ranges (one function may hold several tables).
    entries.sort()
    merged = []
    for start, size in entries:
        if merged and start < merged[-1][0] + merged[-1][1]:
            s0, z0 = merged[-1]
            merged[-1] = (s0, max(z0, start + size - s0))
        else:
            merged.append((start, size))
    for start, size in merged:
        print(f"    {{ address = 0x{start:08X}, size = 0x{size:X} }},")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

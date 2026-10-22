#!/usr/bin/env python3
"""Rewrite XenonRecomp's output for the original Xbox.

  postprocess.py <generated dir>

Every lvx/stvx (and their 128 forms) reverses the 16 bytes of the vector with
simde_mm_shuffle_epi8(x, VectorMaskL). On the Pentium III that shuffle is
emulated byte by byte; ppc_bswap128 does the same with four bswaps.

A jump table (switch) takes its index's low 32 bits, as the guest's cmplwi
bounds it, and an index past its cases reports the switch (ppc_bad_switch)
instead of reaching __builtin_unreachable(): a table found with too few
cases would jump anywhere. Files are rewritten in place, and only when they
change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SWITCH = re.compile(r"\tswitch \(ctx\.(r\d+)\.u64\) \{\n\tcase 0:\n\t\tgoto loc_([0-9A-F]+);")
UNREACHABLE = "\tdefault:\n\t\t__builtin_unreachable();"


def rewrite_switches(text: str) -> tuple[str, int]:
    out, pos, count = [], 0, 0
    while m := SWITCH.search(text, pos):
        end = text.index(UNREACHABLE, m.end())
        register, first = m.group(1), m.group(2)
        out.append(text[pos:m.start()])
        out.append(text[m.start():end].replace(f"ctx.{register}.u64", f"ctx.{register}.u32", 1))
        out.append(f"\tdefault:\n\t\tppc_bad_switch(0x{first}, ctx.{register}.u32);")
        pos = end + len(UNREACHABLE)
        count += 1
    out.append(text[pos:])
    return "".join(out), count

SHUFFLE = "simde_mm_shuffle_epi8("
MASK = "simde_mm_load_si128((simde__m128i*)VectorMaskL)"


def split_args(text: str, start: int) -> tuple[list[str], int]:
    """The comma-separated arguments of the call whose '(' is at start-1."""
    depth, args, begin, i = 0, [], start, start
    while True:
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                args.append(text[begin:i].strip())
                return args, i + 1
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[begin:i].strip())
            begin = i + 1
        i += 1


def rewrite(text: str) -> tuple[str, int]:
    out, pos, count = [], 0, 0
    while (i := text.find(SHUFFLE, pos)) != -1:
        args, end = split_args(text, i + len(SHUFFLE))
        out.append(text[pos:i])
        if len(args) == 2 and args[1] == MASK:
            inner, n = rewrite(args[0])
            out.append(f"ppc_bswap128({inner})")
            count += n + 1
        else:
            out.append(text[i:end])
        pos = end
    out.append(text[pos:])
    return "".join(out), count


def main() -> int:
    total = switches = 0
    for path in sorted(Path(sys.argv[1]).glob("ppc_recomp.*.cpp")):
        text = path.read_text()
        new, count = rewrite(text)
        new, n = rewrite_switches(new)
        if new != text:
            path.write_text(new)
        total += count
        switches += n
    print(f"postprocess: {total} byte-reversing shuffles -> ppc_bswap128, {switches} switches checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

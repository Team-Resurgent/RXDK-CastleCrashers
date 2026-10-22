#!/usr/bin/env python3
"""Rewrite XenonRecomp's output for the original Xbox.

  postprocess.py <generated dir>

Every lvx/stvx (and their 128 forms) reverses the 16 bytes of the vector with
simde_mm_shuffle_epi8(x, VectorMaskL). On the Pentium III that shuffle is
emulated byte by byte; ppc_bswap128 does the same with four bswaps.

A jump table (switch) takes its index's low 32 bits, as the guest's cmplwi
bounds it, and an index past its cases reports the switch (ppc_bad_switch)
instead of reaching __builtin_unreachable(): a table found with too few
cases would jump anywhere.

PATCHES change the title itself for the original Xbox: each replaces one
exact line in one function, and the build fails if the line is not there.
Files are rewritten in place, and only when they change.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# (function, generated line, replacement, why)
PATCHES: list[tuple[str, str, str, str]] = [
    # The job workers rasterize the vector art, each into its own 6 MB arena
    # (memory is committed as it is touched: an unused arena costs nothing).
    # On one processor more workers only hold more raster buffers at once.
    ("sub_82092E28", "cr6.compare<uint32_t>(ctx.r31.u32, 5, xer);", "cr6.compare<uint32_t>(ctx.r31.u32, 1, xer);",
     "one job worker, not five"),
    # The worker's loop packing its float raster into the texture: done in
    # runtime/title.cpp instead (cc_raster_done), the same to the bit.
    ("sub_820D4030", "ctx.r9.s64 = int64_t(ctx.r9.s32) * int64_t(ctx.r8.s32);", "ctx.r9.s64 = 0;",
     "the pack loop to the runtime"),
    # A worker has drawn a job's shape into its texture (job +564, +552 by
    # +556 texels): the renderer takes it (runtime/title.cpp).
    ("sub_820D4030", "\tsub_820D4A90(ctx, base);",
     "\t{ extern void cc_raster_done(PPCContext&, uint32_t); cc_raster_done(ctx, ctx.r31.u32); }\n"
     "\tsub_820D4A90(ctx, base);", "the rasterized texture to the renderer"),
]

SWITCH =re.compile(r"\tswitch \(ctx\.(r\d+)\.u64\) \{\n\tcase 0:\n\t\tgoto loc_([0-9A-F]+);")
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


def apply_patches(text: str, applied: set[int]) -> str:
    for i, (function, old, new, _) in enumerate(PATCHES):
        start = text.find(f"PPC_FUNC_IMPL(__imp__{function}) {{")
        if start == -1:
            continue
        end = text.index("\n}\n", start)
        body = text[start:end]
        if body.count(new) == 1:  # already patched: an earlier run
            pass
        elif body.count(old) == 1:
            text = text[:start] + body.replace(old, new) + text[end:]
        else:
            raise SystemExit(f"postprocess: {function} has no single line {old!r}")
        applied.add(i)
    return text


def main() -> int:
    total = switches = 0
    applied: set[int] = set()
    for path in sorted(Path(sys.argv[1]).glob("ppc_recomp.*.cpp")):
        text = path.read_text()
        new, count = rewrite(text)
        new, n = rewrite_switches(new)
        new = apply_patches(new, applied)
        if new != text:
            path.write_text(new)
        total += count
        switches += n
    missing = [PATCHES[i][0] for i in range(len(PATCHES)) if i not in applied]
    if missing:
        raise SystemExit(f"postprocess: no function {', '.join(missing)} to patch")
    print(f"postprocess: {total} byte-reversing shuffles -> ppc_bswap128, {switches} switches checked, "
          f"{len(PATCHES)} patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

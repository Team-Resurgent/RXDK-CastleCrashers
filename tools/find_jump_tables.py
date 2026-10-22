#!/usr/bin/env python3
"""Jump tables XenonAnalyse does not find.

  find_jump_tables.py <image.bin> <switch tables .toml> [<function mapping .cpp>]

This title's compiler also builds switches as

    lis rB,table@ha ; addi rB,rB,table@l ; rlwinm rC,rI,2,0,29
    lwzx rX,rB,rC ; mtctr rX ; bctr

with no bounds check, the table of absolute addresses right after the bctr.
Without a [[switch]] entry the recompiler takes the bctr for a call through a
pointer, to an address inside the function where there is no function. This
prints a [[switch]] entry for each such bctr the tables file does not cover:
base is the bctr, r the index register, the labels the table's entries (those
after the table, inside the function: the table has no length).
"""
from __future__ import annotations

import bisect
import re
import struct
import sys
import tomllib

BASE = 0x82000000


def main() -> int:
    image = open(sys.argv[1], "rb").read()
    known = tomllib.load(open(sys.argv[2], "rb")).get("switch", [])
    covered = {t["base"] for t in known}
    mapping = sys.argv[3] if len(sys.argv) > 3 else "build/ppc/ppc_func_mapping.cpp"
    funcs = sorted(int(a, 16) for a in re.findall(r"\{ 0x([0-9A-F]+),", open(mapping).read()))
    code_start, code_end = funcs[0], funcs[-1] + 0x1000

    def word(a: int) -> int:
        return struct.unpack_from(">I", image, a - BASE)[0]

    def signed16(v: int) -> int:
        return v - 0x10000 if v & 0x8000 else v

    def emit(a: int, index: int, labels: list[int]) -> None:
        print("[[switch]]")
        print(f"base = 0x{a:08X}")
        print(f"r = {index}")
        print(f"default = 0x{labels[0]:08X}")
        print("labels = [")
        for v in labels:
            print(f"    0x{v:08X},")
        print("]")
        print()

    def bound(a: int, index: int) -> int | None:
        """N + 1 from a cmplwi rIndex,N before the jump, if there is one."""
        for back in range(4, 96, 4):
            w = word(a - back)
            if w >> 26 == 10 and (w >> 16) & 31 == index:  # cmpli
                return (w & 0xFFFF) + 1
        return None

    def offset_table(a: int) -> tuple[int, int, int, int, int] | None:
        """(index register, table, entry size, entry shift, targets' base) of
        an offset table jump at the bctr at a, or None."""
        seq, k = [], a - 4
        while len(seq) < 12 and k > a - 0x60:
            if word(k) != 0x60000000:
                seq.append(word(k))
            k -= 4
        if not seq or (seq[0] & 0xFC1FFFFF) != 0x7C0903A6:  # mtctr rB
            return None
        rb = (seq[0] >> 21) & 31
        add = seq[1] if len(seq) > 1 else 0
        if (add & 0xFC0007FE) != 0x7C000214 or (add >> 21) & 31 != rb:  # add rB,..
            return None
        a1, a2 = (add >> 16) & 31, (add >> 11) & 31
        rs = a2 if a1 == rb else a1
        base_hi = base_lo = table_hi = table_lo = load = None
        shift, index = 0, None
        for w in seq[2:]:
            op, rd, ra = w >> 26, (w >> 21) & 31, (w >> 16) & 31
            if load is None:
                if op == 14 and rd == rb and ra == rb and base_lo is None:
                    base_lo = w & 0xFFFF
                elif op == 15 and rd == rb and base_hi is None:
                    base_hi = w & 0xFFFF
                elif op == 21 and rd == rs and ra == rs:  # rlwinm rS,rS,sh
                    shift = (w >> 11) & 31
                elif (w & 0xFC0007FE) in (0x7C00022E, 0x7C0000AE) and rd == rs:  # lhzx, lbzx
                    load = w
                    rt, rx = ra, (w >> 11) & 31
                    index = rx
                else:
                    return None
            else:
                if op == 21 and ra == rx and index == rx:  # rlwinm rX,rI,sh
                    index = rd
                elif op == 14 and rd == rt and ra == rt and table_lo is None:
                    table_lo = w & 0xFFFF
                elif op == 15 and rd == rt and table_hi is None:
                    table_hi = w & 0xFFFF
                if table_hi is not None and table_lo is not None:
                    break
        if None in (base_hi, base_lo, table_hi, table_lo, load):
            return None
        size = 2 if (load & 0x7FE) == 0x22E else 1
        table = ((table_hi << 16) + signed16(table_lo)) & 0xFFFFFFFF
        target_base = ((base_hi << 16) + signed16(base_lo)) & 0xFFFFFFFF
        return index, table, size, shift, target_base

    found = 0
    for a in range(code_start + 16, code_end, 4):
        if word(a) != 0x4E800420:  # bctr
            continue
        if any(a - 0x80 <= c <= a for c in covered):
            continue
        # Offset tables, with nops anywhere between: lis/addi rT (table),
        # [rlwinm rX,rI,1 for halfwords], lhzx/lbzx rS,rT,rX, [rlwinm rS,rS,2
        # when the entries count instructions], lis/addi rB (the targets'
        # base), add rB,rB,rS, mtctr rB, bctr.
        offset = offset_table(a)
        if offset:
            index, table, size, shift, target_base = offset
            i = bisect.bisect_right(funcs, a)
            f = funcs[i - 1]
            end = funcs[i] if i < len(funcs) else code_end
            # The analysis ends the function at the bctr: with a bounds check
            # every entry counts. With none the table ends where its entries
            # leave the function (zlib's inflate: 30 modes, then other data)
            # or stop being instructions; the analysis may have ended it at
            # the bctr, so up to a generous bound past it.
            count = bound(a, index)
            labels = []
            for i in range(count or 256):
                o = table + i * size
                v = struct.unpack_from(">H" if size == 2 else ">B", image, o - BASE)[0]
                target = target_base + (v << shift)
                if target < f or target & 3 or target >= (f + 0x10000 if count else max(end, a + 0x400)):
                    break
                labels.append(target)
            if labels:
                found += 1
                emit(a, index, labels)
            continue
        mtctr, lwzx = word(a - 4), word(a - 8)
        if (mtctr & 0xFC1FFFFF) != 0x7C0903A6:  # mtctr rX
            continue
        rx = (mtctr >> 21) & 31
        if (lwzx & 0xFC0007FE) != 0x7C00002E or (lwzx >> 21) & 31 != rx:  # lwzx rX,rB,rC
            continue
        rb, rc = (lwzx >> 16) & 31, (lwzx >> 11) & 31
        # Back to the table address (lis/addi into rB) and the index register
        # (rlwinm rC,rI,2,0,29).
        hi = lo = index = None
        for back in range(12, 64, 4):
            w = word(a - back)
            op = w >> 26
            if index is None and op == 21 and (w >> 16) & 31 == rc and w & 0xFFFE == (2 << 11) | (0 << 6) | (29 << 1):
                index = (w >> 21) & 31
            elif lo is None and op == 14 and (w >> 21) & 31 == rb and (w >> 16) & 31 == rb:
                lo = w & 0xFFFF
            elif hi is None and op == 15 and (w >> 21) & 31 == rb:
                hi = w & 0xFFFF
            if hi is not None and lo is not None and index is not None:
                break
        if hi is None or lo is None or index is None:
            continue
        table = ((hi << 16) + (lo - 0x10000 if lo & 0x8000 else lo)) & 0xFFFFFFFF
        f = funcs[bisect.bisect_right(funcs, a) - 1]
        end = funcs[bisect.bisect_right(funcs, a)] if bisect.bisect_right(funcs, a) < len(funcs) else code_end
        labels = []
        count = bound(a, index)
        if count:
            # A bounds check: every entry is a case, before the table as well
            # as after it.
            labels = [word(table + 4 * i) for i in range(count)]
            if any(v & 3 or not f <= v < f + 0x10000 for v in labels):
                print(f"# find_jump_tables: {a:08X}: entries outside the function", file=sys.stderr)
                continue
        t = table
        while not count:
            v = word(t)
            # Entries point past the table, into this function's code (the
            # analysis may have split it: up to a generous bound).
            if v & 3 or not (table < v < max(end, table + 0x4000)) or v < f:
                break
            labels.append(v)
            t += 4
            if labels and t >= min(labels):
                break
        if not labels:
            continue
        found += 1
        emit(a, index, labels)
    print(f"# find_jump_tables: {found} tables", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

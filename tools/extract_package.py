#!/usr/bin/env python3
"""Extract an Xbox Live Arcade package (STFS, LIVE/PIRS) into a folder.

  extract_package.py <package file> <out dir>

The package is the file under <title id>/000D0000/ in a XBLA download. Every
file in it (default.xex, media/...) is written under <out dir>.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def main() -> int:
    data = Path(sys.argv[1]).read_bytes()
    out = Path(sys.argv[2])
    if data[:4] not in (b"LIVE", b"PIRS", b"CON "):
        raise SystemExit(f"{sys.argv[1]} is not an STFS package")
    header_size = struct.unpack(">I", data[0x340:0x344])[0]
    descriptor = data[0x379:0x379 + 0x24]
    table_blocks = struct.unpack("<H", descriptor[3:5])[0]
    table_block = int.from_bytes(descriptor[5:8], "little")
    base = (header_size + 0xFFF) & ~0xFFF
    # Packages with a header that ends in the 0xB000 page keep one hash table
    # per level, others two (Free60's STFS notes).
    shift = 0 if ((header_size + 0xFFF) & 0xF000) == 0xB000 else 1

    def block(n: int) -> int:
        """File offset of data block n: hash tables sit between the blocks."""
        r = (((n + 0xAA) // 0xAA) << shift) + n
        if n >= 0xAA:
            r += ((n + 0x70E4) // 0x70E4) << shift
            if n >= 0x70E4:
                r += 1 << shift
        return base + (r << 12)

    table = b"".join(data[block(table_block + i):block(table_block + i) + 0x1000] for i in range(table_blocks))
    entries = []
    for i in range(len(table) // 0x40):
        e = table[i * 0x40:(i + 1) * 0x40]
        flags = e[0x28]
        if flags & 0x3F == 0:
            break
        entries.append((e[:flags & 0x3F].decode("latin-1"), flags, int.from_bytes(e[0x2F:0x32], "little"),
                        struct.unpack(">h", e[0x32:0x34])[0], struct.unpack(">I", e[0x34:0x38])[0]))
    paths = {-1: out}
    count = 0
    for index, (name, flags, start, parent, size) in enumerate(entries):
        path = paths[parent] / name
        if flags & 0x80:
            paths[index] = path
            path.mkdir(parents=True, exist_ok=True)
            continue
        if not flags & 0x40:
            raise SystemExit(f"{name}: non-contiguous files are not supported")
        blocks = (size + 0xFFF) // 0x1000
        contents = b"".join(data[block(start + k):block(start + k) + 0x1000] for k in range(blocks))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents[:size])
        count += 1
    print(f"extract_package: {count} files into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Castle Crashers for the original Xbox

**RXDK-CastleCrashers** ports The Behemoth's *Castle Crashers* (Xbox Live
Arcade, 2008) to the **original Xbox** by static recompilation. The title's
PowerPC code is translated to C++ with
[XenonRecomp](https://github.com/hedge-dev/XenonRecomp), compiled for the
Xbox's Pentium III with [RXDK](https://github.com/Team-Resurgent), and run on
a small runtime that provides the Xbox 360 kernel and XAM calls the game
makes, using the original Xbox's own. The runtime started as
[RXDK-Jetpac](../RXDK-Jetpac)'s.

No game code or data is in this repository. You need your own copy of the
game: the XBLA package (`584108B7/000D0000/<package file>`) and its title
update 2 (`TU_<...>`), which the build applies.

## Status

Early work in progress. **Nothing is playable yet.**

- Title update 2 is applied to `default.xex` and the whole title
  recompiles (7,170 functions).
- In xemu it runs its logos, legal screen, attract sequence and title menu,
  drawn on the NV2A, at about 30 frames a second, and with a scripted
  controller (`build/inputs.txt`) goes from the menu through the gameplay
  loading (every pack and sound) into the character select. Sound is
  silent.
- The title draws its vector art into textures on the processor at load
  time, and on the Pentium III that is slow: the gameplay loading takes
  minutes.
- Memory, aiming at a retail 64 MB console:
  - Guest memory is committed when it is touched: the title commits a 32 MB
    heap, a 6 MB arena per 360 hardware thread and a 16 MB GPU block, and
    touches a fraction of them (`src/runtime/memory.cpp`). The guest window
    is 512 MB of address space for what it commits.
  - One job worker instead of five: the workers rasterize the title's
    vector art into textures, each with its own arena.
  - Each rasterized texture is compressed to DXT5 as the worker finishes it
    and its guest memory given back (`gpu::renderer::prepare_texture`);
    until it is first drawn it is kept LZ-compressed (about 6:1 over DXT5,
    for the thousands of animation frames a pack draws at once).
  - The sound effects' XMA data is given back as it is read (sound is
    silent), host stacks are 64 KB, and the game code is built with -Os.
  - At 64 MB (xemu, iND-BiOS) the title runs through the character select
    loading with about 10 MB free.

## Building

On Windows, with Visual Studio (with its C++ Clang tools), Python 3, Git and
RXDK:

    python tools/build.py --package "<path to>/584108B7/000D0000/<package file>" --update "<path to>/TU_<...>"

The result is `out/castlecrashers.xbe` with the game's files staged next to
it in `out/Build/castlecrashers/`, and the disc image
`out/XISO/castlecrashers.iso`. Later builds need neither `--package` nor
`--update`; `--clean` regenerates everything from them.

## How it works

| Piece | Where |
| --- | --- |
| Package extraction (STFS) | `tools/extract_package.py` |
| Title update applied, memory image and import list from `default.xex` | `tools/xexprep` (uses XenonRecomp's XEX loader and patcher) |
| Instructions the pinned XenonRecomp does not handle | `tools/xenonrecomp` (applied when it is cloned) |
| Recompiler configuration: helper addresses, jump tables, function bounds | `config/`, `tools/find_jump_tables.py`, `tools/find_switch_funcs.py` |
| Pentium III changes to the generated code | `src/ppc/ppc_context.h`, `src/ppc/ppc_xbox_simd.h`, `tools/postprocess.py` |
| Guest memory: a 256 MB window at guest `0x80000000` | `src/runtime/memory.cpp` |
| Kernel: threads, objects and waits, memory, the loader's queries | `src/runtime/kernel.cpp` |
| Files (`game:\` is the title's folder, `D:\`; `update:\` is `D:\update`) | `src/runtime/io.cpp` |
| XAM: the controller, no Live profile, no system UI | `src/runtime/xam.cpp` |
| The GPU: the title's command buffers drawn with Direct3D 8 | `src/runtime/gpu.cpp`, `src/gpu` |
| Imports not implemented yet: logging stubs | `tools/gen_stubs.py` |
| Changes to the title itself: one job worker, polling that sleeps, rasterized textures handed to the renderer | `tools/postprocess.py` (`PATCHES`), `src/runtime/title.cpp` |

Notes on the Pentium III:

- The generated code uses SSE4.1 through SIMDe. The Pentium III has SSE1
  only, so SIMDe emulates integer vectors one element at a time. Every
  lvx/stvx byte reversal is four `bswap`s, and the other emulated
  operations are called out of line.
- The Pentium III has no DAZ bit in MXCSR: only flush-to-zero is used.
- Guest addresses are 32 bits like host pointers, so `base + address` wraps
  to the host address without masking.

## Credits

[XenonRecomp](https://github.com/hedge-dev/XenonRecomp) by hedge-dev
(MIT), and [Xenia](https://github.com/xenia-project/xenia) for its
documentation of the Xbox 360 kernel's structures and its shader
interpreter.

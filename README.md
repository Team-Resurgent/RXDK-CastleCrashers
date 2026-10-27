# Castle Crashers for the original Xbox

<p align="center"><b>The Xbox Live Arcade's Castle Crashers, statically recompiled for the original Xbox with the open RXDK SDK — runs in 64 MB, from your own copy of the game</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-work%20in%20progress-orange.svg" alt="Status: work in progress">
  <a href="https://discord.gg/VcdSfajQGK"><img src="https://img.shields.io/badge/chat-on%20discord-7289da.svg?logo=discord" alt="Discord"></a>
</p>

<p align="center">
  <a href="https://ko-fi.com/J3J7L5UMN"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="ko-fi"></a>
  <a href="https://www.patreon.com/teamresurgent"><img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon"></a>
</p>

> **Work in progress.** The game boots and is playable, but it is not finished:
> expect bugs, missing effects, slowdowns and changes from one commit to the
> next. See [Status](#status) for where it stands.

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

Work in progress: playable, not finished.

- Title update 2 is applied to `default.xex` and the whole title
  recompiles (7,170 functions).
- It runs on a real console (tested on a development kit) and in xemu at
  64 MB: logos, legal screen, attract sequence, menus, character select and
  levels, drawn on the NV2A with the controller.
- Speed: at the start of the first level about 36 fps on the console (51 in
  xemu, whose processor is faster than the Pentium III). The recompiled code
  keeps the PowerPC registers in locals, so the compiler holds them in the
  processor's registers; the few hundred functions that need them in the
  context are found by `tools/context_functions.py`. The title's vertex
  shaders run as native code, and runs of sprites with the same state are
  drawn together.
- Sound and music play on the APU, each sound on a voice of its own, from
  sounds converted when the game is built: XAudio's source voices play
  their converted sound with the title's volume, pan, pitch and loops, and
  XAudio itself mixes nothing.
- The title draws its vector art into textures on the processor; its spans
  and the packing into textures run as native code (`src/runtime/title.cpp`),
  the same to the bit as the recompiled code. The art of some packs is drawn
  when it is first needed rather than at load time.
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
  - Textures not drawn for a few seconds are compressed back to that form.
  - The sounds' XMA data is given back as it is read (the converted sounds
    play instead), host stacks are 64 KB, and the game code is built with
    -Os, but for the five generated files with its hottest code (-O2).
- Known issues: loading is slow on a console; some effects (pixel shaders
  such as glows and blurs) are not drawn; short audio gaps have been heard
  on a console, cause not found yet. Only the first levels have been played
  through.

## Building

On Windows, with Visual Studio (with its C++ Clang tools), Python 3, Git,
RXDK, and [ffmpeg](https://ffmpeg.org) on the PATH (its XMA decoder converts
the sounds):

    python tools/build.py --package "<path to>/584108B7/000D0000/<package file>" --update "<path to>/TU_<...>"

The result is `out/castlecrashers.xbe` with the game's files staged next to
it in `out/Build/castlecrashers/`, and the disc image
`out/XISO/castlecrashers.iso`. Later builds need neither `--package` nor
`--update`; `--clean` regenerates everything from them.

The first build also converts every sound (`tools/convert_audio.py`, a
minute or so): the Pentium III cannot decode XMA, so each sound is decoded
with ffmpeg and stored as IMA ADPCM in `build/audio/Sounds.jpa` and
`Music.jpa` (about 175 MB), which go on the disc as `D:\audio`.

## How it works

| Piece | Where |
| --- | --- |
| Package extraction (STFS) | `tools/extract_package.py` |
| Title update applied, memory image and import list from `default.xex` | `tools/xexprep` (uses XenonRecomp's XEX loader and patcher) |
| Instructions the pinned XenonRecomp does not handle, and the registers in locals per function | `tools/xenonrecomp` (applied when it is cloned or changes), `tools/context_functions.py` |
| Recompiler configuration: helper addresses, jump tables, function bounds | `config/`, `tools/find_jump_tables.py`, `tools/find_switch_funcs.py` |
| Pentium III changes to the generated code | `src/ppc/ppc_context.h`, `src/ppc/ppc_xbox_simd.h`, `tools/postprocess.py` |
| Guest memory: a 512 MB window at guest `0x7F000000`, committed as it is touched | `src/runtime/memory.cpp` |
| Kernel: threads, objects and waits, memory, the loader's queries | `src/runtime/kernel.cpp` |
| Files (`game:\` is the title's folder, `D:\`; `update:\` is `D:\update`) | `src/runtime/io.cpp` |
| XAM: the controller, no Live profile, no system UI | `src/runtime/xam.cpp` |
| The GPU: the title's command buffers drawn with Direct3D 8, its vertex shaders as native code | `src/runtime/gpu.cpp`, `src/gpu` |
| Sound: XAudio's source voices played on APU voices from the converted sounds | `tools/convert_audio.py`, `src/runtime/xma.cpp`, `src/runtime/audio.cpp`, `src/runtime/xaudio.cpp` |
| Imports not implemented yet: logging stubs | `tools/gen_stubs.py` |
| Changes to the title itself: one job worker, polling that sleeps, native rasterizer spans and packing, rasterized textures handed to the renderer, spin loops that yield | `tools/postprocess.py` (`PATCHES`), `src/runtime/title.cpp` |

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

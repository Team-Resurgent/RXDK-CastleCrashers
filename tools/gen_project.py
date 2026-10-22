#!/usr/bin/env python3
"""Write the RXDK projects for the original Xbox build.

  rxdk.project.json           the title: src/main.cpp, the runtime and the
                              import stubs; the user's files deployed to D:\
  gamecode/rxdk.project.json  the recompiled title (build/ppc), from the
                              user's own default.xex: its own library, so
                              runtime changes never recompile it
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PPC = ROOT / "build/ppc"
SIMDE = ROOT / "extern/XenonRecomp/thirdparty/simde"
LIBRARIES = ["libd3d8.lib", "libdsound.lib", "libxgraphics.lib", "libxapi.lib", "libcpp.lib", "libc.lib",
             "libkernel.lib"]
# Guest addresses wrap modulo 2^32 in host pointer arithmetic (guest.h).
FLAGS = ["-fno-strict-overflow", "-Wno-unused-label", "-Wno-unused-variable", "-Wno-unused-but-set-variable"]
# PPC_CALL_TRACE: each recompiled function records itself in its thread's
# ring of recent calls (ppc_context.h), for a crash's call chain. Slower:
# set CC_CALL_TRACE=1 to build with it.
DEFINES = ["PPC_CALL_TRACE"] if os.environ.get("CC_CALL_TRACE") == "1" else []
# The package's data folders, deployed to D:\ as they are.
DATA = ["art", "bsps", "fonts", "game", "levels", "music", "shaders", "sounds"]


def rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def write(directory: Path, project: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rxdk.project.json").write_text(json.dumps(project, indent=2) + "\n")


def main() -> int:
    code = sorted(PPC.glob("ppc_recomp.*.cpp")) + [PPC / "ppc_func_mapping.cpp"]
    # The generated code includes its own copy of ppc_context.h: keep it ours
    # (copied only when it changed, so the generated code is not rebuilt).
    ours, copy = ROOT / "src/ppc/ppc_context.h", PPC / "ppc_context.h"
    if copy.exists() and copy.read_bytes() != ours.read_bytes():
        shutil.copyfile(ours, copy)
    if len(code) < 2:
        raise SystemExit(f"recompile the title first: nothing in {PPC}")
    includes = [PPC, ROOT / "src/ppc", SIMDE]
    gamecode = ROOT / "gamecode"
    write(gamecode, {
        "name": "gamecode", "type": "library", "configuration": "release",
        "sources": [rel(p, gamecode) for p in code],
        "includePaths": [rel(p, gamecode) for p in includes],
        # -O2 after RXDK's -O3: as small and as fast.
        "defines": DEFINES,
        "compileFlags": FLAGS + ["-O2"],
        "cppStandard": "c++20", "outputDir": "out"})

    runtime = sorted((ROOT / "src/runtime").glob("*.cpp")) + sorted((ROOT / "src/gpu").rglob("*.cpp"))
    sources = [ROOT / "src/main.cpp"] + runtime + [PPC / "ppc_import_stubs.cpp"]
    write(ROOT, {
        "name": "castlecrashers", "configuration": "release",
        "sources": [rel(p, ROOT) for p in sources],
        "includePaths": [rel(p, ROOT) for p in [ROOT / "src"] + includes],
        "projectReferences": ["gamecode"],
        "libraries": LIBRARIES,
        "defines": DEFINES,
        "compileFlags": FLAGS,
        "cppStandard": "c++20", "outputDir": "out",
        "deployPaths": ["image.bin=build/image.bin", "default.xex=build/default.xex"]
        + [f"{d}=game/package/{d}" for d in DATA]
        # The title update's data files (runtime/io.cpp: update:\).
        + ["update=game/update"]
        # The wave banks as ADPCM (tools/convert_audio.py, runtime/xma.cpp).
        + (["audio=build/audio"] if (ROOT / "build/audio").exists() else [])
        # A test input script (runtime/xam.cpp), only when there is one.
        + (["inputs.txt=build/inputs.txt"] if (ROOT / "build/inputs.txt").exists() else []),
        "imageBuild": {"testName": "Castle Crashers"}})
    print(f"gen_project: {len(code)} generated sources, {len(runtime)} runtime sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

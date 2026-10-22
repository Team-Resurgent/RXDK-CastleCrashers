#!/usr/bin/env python3
"""Build Castle Crashers for the original Xbox from your own XBLA package.

  python tools/build.py --package "<...>/584108B7/000D0000/<package file>"
                        --update "<...>/TU_<...>"   (title update 2)

  1. extract_package.py: the package's files into game/package, the title
     update's into game/update
  2. the host tools (XenonRecomp, XenonAnalyse, xexprep) with CMake, Ninja
     and clang-cl from Visual Studio
  3. xexprep: the title update applied (build/default.xex), the memory image
     (build/image.bin) and import list
  4. XenonRecomp (config/castlecrashers.toml): the title as C++ in
     build/ppc, then postprocess.py and gen_stubs.py
  5. gen_project.py and the RXDK build: out/castlecrashers.xbe and
     out/XISO/castlecrashers.iso

Steps whose outputs exist are skipped; --clean starts over. XenonRecomp comes
from extern/XenonRecomp (cloned at the pinned commit when missing). The
package and everything generated from it stay on your computer: none of it
is in the repository.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XENONRECOMP = ROOT / "extern/XenonRecomp"
XENONRECOMP_URL = "https://github.com/hedge-dev/XenonRecomp.git"
XENONRECOMP_COMMIT = "ddd128b"
PACKAGE = ROOT / "game/package"
UPDATE = ROOT / "game/update"
BUILD = ROOT / "build"
HOST = BUILD / "host"


def run(cmd: list, **kw) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if subprocess.run([str(c) for c in cmd], **kw).returncode:
        sys.exit(f"build: failed: {cmd[0]}")


def visual_studio() -> Path:
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    if vswhere.exists():
        found = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                               capture_output=True, text=True).stdout.strip()
        if found:
            return Path(found)
    for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
        path = Path(r"C:\Program Files\Microsoft Visual Studio\2022") / edition
        if path.exists():
            return path
    sys.exit("build: Visual Studio 2022 (with its C++ Clang tools) not found")


def host_tools() -> None:
    if (HOST / "xexprep.exe").exists() and (HOST / "XenonRecomp/XenonRecomp/XenonRecomp.exe").exists():
        return
    if not (XENONRECOMP / "CMakeLists.txt").exists():
        run(["git", "clone", "--recursive", XENONRECOMP_URL, XENONRECOMP])
        run(["git", "-C", XENONRECOMP, "checkout", XENONRECOMP_COMMIT])
        run(["git", "-C", XENONRECOMP, "submodule", "update", "--init", "--recursive"])
        # Instructions this commit does not handle (tools/xenonrecomp).
        run(["git", "-C", XENONRECOMP, "apply", ROOT / "tools/xenonrecomp/recompiler.patch"])
    vs = visual_studio()
    ninja = vs / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe"
    vcvars = vs / "VC/Auxiliary/Build/vcvars64.bat"
    script = (f'call "{vcvars}" >nul && cmake -S "{ROOT / "tools/xexprep"}" -B "{HOST}" -G Ninja '
              f'-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl '
              f'-DCMAKE_MAKE_PROGRAM="{ninja}" && cmake --build "{HOST}" --target xexprep XenonAnalyse XenonRecomp')
    # Through a batch file: subprocess would escape the quotes of a cmd /c line.
    HOST.mkdir(parents=True, exist_ok=True)
    (HOST / "build.bat").write_text("@" + script + "\r\n")
    run(["cmd", "/c", HOST / "build.bat"])


def rxdk() -> Path:
    root = Path(os.environ.get("RXDK", r"C:\ProgramData\RXDK"))
    cli = os.environ.get("RXDK_CLI")
    for candidate in ([Path(cli)] if cli else [root / "tools/Rxdk.Cli.exe", root / "engine/Rxdk.Cli.exe"]):
        if candidate.exists():
            return candidate
    sys.exit("build: RXDK not found (install it, or set RXDK or RXDK_CLI)")


def recompile(ppc: Path) -> None:
    """The title as C++ into a staging folder, then only the files that
    changed into ppc: the RXDK build recompiles what is newer."""
    staging = BUILD / "ppc.staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    run([HOST / "xexprep.exe", BUILD / "default.xex", BUILD / "image.bin", staging / "ppc_imports.inc"])
    # The configuration's paths are relative to it: a copy next to it that
    # writes to the staging folder.
    config = ROOT / "config/castlecrashers.toml"
    staged_config = ROOT / "config/.staging.toml"
    staged_config.write_text(config.read_text(encoding="utf-8-sig").replace(
        'out_directory_path = "../build/ppc"', 'out_directory_path = "../build/ppc.staging"'))
    try:
        run([HOST / "XenonRecomp/XenonRecomp/XenonRecomp.exe", staged_config, ROOT / "src/ppc/ppc_context.h"],
            stdout=open(BUILD / "recomp.log", "w"), stderr=subprocess.STDOUT)
    finally:
        staged_config.unlink(missing_ok=True)
    run([sys.executable, ROOT / "tools/postprocess.py", staging])
    ppc.mkdir(parents=True, exist_ok=True)
    changed = 0
    for path in staging.iterdir():
        target = ppc / path.name
        if not target.exists() or target.read_bytes() != path.read_bytes():
            shutil.copyfile(path, target)
            changed += 1
    for path in ppc.glob("ppc_recomp.*.cpp"):
        if not (staging / path.name).exists():
            path.unlink()
    (ppc / "ppc_func_mapping.cpp").touch()
    shutil.rmtree(staging, ignore_errors=True)
    print(f"build: recompiled, {changed} generated files changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, help="the XBLA package file (needed once)")
    parser.add_argument("--update", type=Path, help="the title update 2 package file (needed once)")
    parser.add_argument("--clean", action="store_true", help="regenerate everything from the package")
    args = parser.parse_args()
    cli = rxdk()
    if args.clean:
        shutil.rmtree(BUILD / "ppc", ignore_errors=True)
        (BUILD / "image.bin").unlink(missing_ok=True)

    if not (PACKAGE / "default.xex").exists():
        if not args.package:
            sys.exit("build: give your package with --package (the file under 584108B7/000D0000/)")
        run([sys.executable, ROOT / "tools/extract_package.py", args.package, PACKAGE])
    if not (UPDATE / "default.xexp").exists():
        if not args.update:
            sys.exit("build: give title update 2 with --update (its TU_... file)")
        run([sys.executable, ROOT / "tools/extract_package.py", args.update, UPDATE])

    host_tools()
    ppc = BUILD / "ppc"
    if not (BUILD / "image.bin").exists():
        run([HOST / "xexprep.exe", "--patch", PACKAGE / "default.xex", UPDATE / "default.xexp", BUILD / "default.xex"])
    # Recompiled when the configuration or the recompiler's tools change.
    inputs = [ROOT / "config/castlecrashers.toml", ROOT / "config/castlecrashers_switch_tables.toml",
              ROOT / "tools/postprocess.py", HOST / "XenonRecomp/XenonRecomp/XenonRecomp.exe"]
    mapping = ppc / "ppc_func_mapping.cpp"
    if not mapping.exists() or any(p.stat().st_mtime > mapping.stat().st_mtime for p in inputs):
        recompile(ppc)
    run([sys.executable, ROOT / "tools/gen_stubs.py", ppc, ROOT / "src/runtime"])
    # The sounds decoded and stored as ADPCM, once (needs ffmpeg).
    audio = BUILD / "audio"
    if not all((audio / bank).exists() for bank in ("Sounds.jpa", "Music.jpa")):
        if not shutil.which("ffmpeg"):
            sys.exit("build: the sounds are converted with ffmpeg (its xma2 decoder): install it and put it on PATH")
        run([sys.executable, ROOT / "tools/convert_audio.py", PACKAGE, UPDATE, audio])
    run([sys.executable, ROOT / "tools/gen_project.py"])
    # The RXDK build leaves staged files that left the manifest: a test input
    # script must not stay on the disc once build/inputs.txt is gone.
    if not (BUILD / "inputs.txt").exists():
        (ROOT / "out/Build/castlecrashers/inputs.txt").unlink(missing_ok=True)
    run([cli, "build", "--project-root", ROOT])
    print(f"build: {ROOT / 'out/castlecrashers.xbe'} and {ROOT / 'out/XISO/castlecrashers.iso'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

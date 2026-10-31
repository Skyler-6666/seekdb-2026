#!/usr/bin/env python3
"""Run the existing exhaustive production-source differential, then local timings.

Default baseline is the round-one submission so timings show incremental gains.
Use --baseline 5bce9e06d1281347ad61344e4b01fab9158c58a2 to compare with upstream.
No result here is a native seekdb build, SQL verification, or competition score.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

BASELINE = "32cf87c11ae460301394f539ae25896748bee73d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--compiler", choices=["auto", "msvc", "g++", "clang++"], default="auto")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    build = args.build_dir.resolve()
    compiler = args.compiler
    if compiler == "auto":
        compiler = "msvc" if os.name == "nt" else ("g++" if shutil.which("g++") else "clang++")
    # This generates actual baseline/current class bodies, actual UTF8 decoder,
    # helper templates and a separate TU with actual Unicode predicates.
    subprocess.run([sys.executable, str(here / "tokenizer_differential.py"),
                    "--baseline", args.baseline, "--build-dir", str(build), "--compiler", compiler], check=True)
    support = (here / "tokenizer_differential.cpp").read_text(encoding="utf-8")
    if support.count("\nint main(") != 1:
        raise RuntimeError("Review source-extracted timing support: main signature changed")
    (build / "tokenizer_round2_support.h").write_text(support.split("\nint main(", 1)[0], encoding="utf-8")
    source = here / "tokenizer_round2_benchmark.cpp"
    executable = build / ("tokenizer_round2_benchmark.exe" if os.name == "nt" else "tokenizer_round2_benchmark")
    if compiler == "msvc":
        command = ["cl", "/nologo", "/std:c++17", "/EHsc", "/O2", "/utf-8", "/I" + str(build), str(source),
                   str(build / "tokenizer_unicode.cpp"), "/Fe:" + str(executable), "/Fo:" + str(build) + os.sep]
        if shutil.which("cl"):
            subprocess.run(command, cwd=build, check=True)
        else:
            vcvars = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
            if not vcvars.is_file():
                raise RuntimeError("cl not found; activate a Visual Studio x64 developer shell")
            batch = build / "tokenizer_round2_compile.cmd"
            batch.write_text('@echo off\ncall "' + str(vcvars) + '" >nul\n' + subprocess.list2cmdline(command)
                             + '\nexit /b %errorlevel%\n', encoding="utf-8")
            subprocess.run(["cmd", "/d", "/c", str(batch)], cwd=build, check=True)
    else:
        subprocess.run([compiler, "-std=c++17", "-O3", "-I", str(build), str(source), str(build / "tokenizer_unicode.cpp"),
                        "-o", str(executable)], cwd=build, check=True)
    run = subprocess.run([str(executable)], cwd=build, check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    result.update({"baseline": args.baseline, "compiler": compiler, "platform": platform.platform(),
                   "regression": json.loads((build / "tokenizer_result.json").read_text(encoding="utf-8"))})
    (build / "tokenizer_round2_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

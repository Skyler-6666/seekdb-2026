#!/usr/bin/env python3
"""Per-file C++ compile check for the file-sql sources.

The repository only supports Unity builds, so a single source file cannot be
compiled through its CMake object target directly.  Instead we reuse the exact
compiler flags CMake generated for the target that owns the file
(``CMakeFiles/<target>.dir/flags.make``) and run the compiler with
``-fsyntax-only`` on just that file.  This is fast enough for a 4-core runner
and still uses the project's real include paths, defines and warnings
(``-Wall -Wextra -Werror``).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time

DEFAULT_FILES = [
    "src/sql/table_function/file_sql_utils.cpp",
    "src/sql/table_function/file_sql_scan.cpp",
    "src/sql/table_function/file_sql_bind.cpp",
    "src/sql/table_function/table_function.cpp",
    "src/sql/session/ob_basic_session_info.cpp",
    "src/sql/session/ob_system_variable_factory.cpp",
    "src/share/system_variable/ob_system_variable_init.cpp",
    "src/share/system_variable/ob_sys_var_meta.cpp",
]

SOURCE_RE = re.compile(r'"([^"]+\.(?:cpp|cc|cxx|C))"\s+"([^"]+)"')


def log(msg):
    print(msg, flush=True)


def cxx_compiler(build_dir):
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = re.match(r"CMAKE_CXX_COMPILER:[^=]*=(.+)$", line.strip())
                if match:
                    return match.group(1).strip()
    return "clang++"


def collect_build_targets(build_dir):
    """map: absolute source path -> CMakeFiles/<target>.dir directory."""
    sources = {}
    for root, _dirs, files in os.walk(build_dir):
        if "DependInfo.cmake" not in files:
            continue
        path = os.path.join(root, "DependInfo.cmake")
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        for match in SOURCE_RE.finditer(text):
            src = os.path.normpath(match.group(1))
            sources.setdefault(src, root)
    return sources


def read_flags(target_dir):
    flags_path = os.path.join(target_dir, "flags.make")
    defines = includes = flags = ""
    try:
        with open(flags_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("CXX_DEFINES"):
                    defines = line.split("=", 1)[1].strip()
                elif line.startswith("CXX_INCLUDES"):
                    includes = line.split("=", 1)[1].strip()
                elif line.startswith("CXX_FLAGS"):
                    flags = line.split("=", 1)[1].strip()
    except OSError:
        return None
    return defines, includes, flags


def check_one(compiler, build_dir, target_dir, source):
    parsed = read_flags(target_dir)
    if parsed is None:
        return False, "flags.make not found in %s" % target_dir
    defines, includes, flags = parsed
    if not defines and not includes and not flags:
        return False, "empty flags.make in %s" % target_dir
    command = [compiler]
    command += shlex.split(defines)
    command += shlex.split(includes)
    command += shlex.split(flags)
    command += [
        "-fsyntax-only",
        "-fno-color-diagnostics",
        "-ferror-limit=40",
        "-Wno-error=unused-command-line-argument",
        source,
    ]
    proc = subprocess.run(
        command, cwd=build_dir, capture_output=True, text=True, errors="replace"
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def unity_sources(build_dir, needles):
    """Unity translation units that pull in one of ``needles`` (file basenames)."""
    found = []
    for root, _dirs, files in os.walk(build_dir):
        for name in files:
            if not (name.startswith("unity_") and name.endswith(".cxx")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError:
                continue
            if any(needle in text for needle in needles):
                found.append(path)
    return found


def check_unity(build_dir):
    """Compile the Unity translation units containing file_sql sources.

    Unity-only problems (anonymous-namespace symbol collisions between
    concatenated files, macro leakage) cannot be reproduced by per-file checks,
    so the real Unity object is compiled as well.
    """
    compile_db = os.path.join(build_dir, "compile_commands.json")
    if not os.path.exists(compile_db):
        log("=== CHECK-ERROR: %s is missing" % compile_db)
        return 1
    import json

    with open(compile_db, encoding="utf-8", errors="replace") as handle:
        entries = json.load(handle)
    wanted = unity_sources(build_dir, ["file_sql_utils.cpp", "file_sql_scan.cpp", "file_sql_bind.cpp"])
    if not wanted:
        log("=== CHECK-ERROR: no unity source contains the file_sql sources")
        return 1
    wanted = {os.path.normpath(path) for path in wanted}
    commands = {}
    for entry in entries:
        path = os.path.normpath(os.path.join(entry.get("directory", ""), entry["file"]))
        if path in wanted:
            commands[path] = entry
    failures = []
    for path in sorted(wanted):
        name = os.path.basename(path)
        entry = commands.get(path)
        log("=== UNITY-START %s" % path)
        if entry is None:
            log("=== UNITY-END %s: FAIL (not in compile_commands.json)" % name)
            failures.append(name)
            continue
        if "arguments" in entry:
            argv = list(entry["arguments"])
        else:
            argv = shlex.split(entry["command"])
        cleaned = []
        skip_next = False
        for arg in argv:
            if skip_next:
                skip_next = False
                continue
            if arg in ("-o", "-MF", "-MT"):
                skip_next = True
                continue
            if arg in ("-c", "-MD", "-MMD"):
                continue
            cleaned.append(arg)
        cleaned += ["-fsyntax-only", "-fno-color-diagnostics", "-Wno-error=unused-command-line-argument"]
        start = time.time()
        proc = subprocess.run(cleaned, cwd=entry.get("directory"), capture_output=True,
                              text=True, errors="replace")
        elapsed = time.time() - start
        output = (proc.stdout or "") + (proc.stderr or "")
        if output.strip():
            lines = output.strip().splitlines()
            if len(lines) > 1200:
                lines = ["... %d lines omitted ..." % (len(lines) - 1200)] + lines[-1200:]
            for line in lines:
                log(line)
        ok = proc.returncode == 0
        log("=== UNITY-END %s: %s (%.1fs)" % (name, "PASS" if ok else "FAIL", elapsed))
        if not ok:
            failures.append(name)
    log("")
    log("=== UNITY-SUMMARY: units=%d failed=%d" % (len(wanted), len(failures)))
    for item in failures:
        log("=== UNITY-FAILED: %s" % item)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default="build_release")
    parser.add_argument("--repo", default=os.getcwd())
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--unity", action="store_true",
                        help="compile the unity translation units instead of single files")
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    repo = os.path.abspath(args.repo)
    wanted = args.only or DEFAULT_FILES

    if not os.path.isdir(build_dir):
        log("=== CHECK-ERROR: build directory %s is missing" % build_dir)
        return 2
    if args.unity:
        return check_unity(build_dir)
    compiler = cxx_compiler(build_dir)
    if not os.path.isabs(compiler) and not os.path.exists(compiler):
        import shutil

        found = shutil.which(compiler)
        if not found:
            log("=== CHECK-ERROR: C++ compiler %r not found" % compiler)
            return 2
        compiler = found
    log("=== CHECK-INFO: compiler=%s build_dir=%s" % (compiler, build_dir))

    targets = collect_build_targets(build_dir)
    log("=== CHECK-INFO: discovered %d compiled sources" % len(targets))

    failures = []
    missing = []
    total_start = time.time()
    for rel in wanted:
        source = os.path.normpath(os.path.join(repo, rel))
        if not os.path.exists(source):
            log("=== CHECK-SKIP %s (file not present)" % rel)
            continue
        target_dir = targets.get(source)
        if target_dir is None:
            log("=== CHECK-SKIP %s (not found in any CMake target)" % rel)
            missing.append(rel)
            continue
        log("=== CHECK-START %s (target %s)" % (rel, os.path.basename(target_dir)[:-4]))
        start = time.time()
        ok, output = check_one(compiler, build_dir, target_dir, source)
        elapsed = time.time() - start
        if output.strip():
            lines = output.strip().splitlines()
            if len(lines) > 1200:
                lines = ["... %d lines omitted ..." % (len(lines) - 1200)] + lines[-1200:]
            for line in lines:
                log(line)
        log("=== CHECK-END %s: %s (%.1fs)" % (rel, "PASS" if ok else "FAIL", elapsed))
        if not ok:
            failures.append(rel)

    log("")
    log("=== CHECK-SUMMARY: checked=%d passed=%d failed=%d skipped=%d total=%.1fs"
        % (len(wanted), len(wanted) - len(failures) - len(missing), len(failures),
           len(missing), time.time() - total_start))
    for item in failures:
        log("=== CHECK-FAILED: %s" % item)
    for item in missing:
        log("=== CHECK-NO-TARGET: %s" % item)
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

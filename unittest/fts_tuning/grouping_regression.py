#!/usr/bin/env python3
"""Compile source-extracted grouping, atomic and upsert code with portable adapters.

This is a focused control-flow/type/operation-count regression, not a native seekdb
build, charset implementation, concurrent hash-table test, or performance score.
Run in a C++17 compiler environment: python unittest/fts_tuning/grouping_regression.py
MSVC users should run from an x64 Native Tools command prompt.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile


BASELINE = "5bce9e06d1281347ad61344e4b01fab9158c58a2"
PREVIOUS = "32cf87c11ae460301394f539ae25896748bee73d"
GROUPING_SOURCE = "src/storage/fts/ob_fts_stop_word.cpp"


def extract_block(source, signature):
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise RuntimeError(f"Unbalanced source block: {signature}")
    return source[start:end]


def build_source(repo):
    old_source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{GROUPING_SOURCE}"], cwd=repo
    ).decode("utf-8")
    previous_source = subprocess.check_output(
        ["git", "show", f"{PREVIOUS}:{GROUPING_SOURCE}"], cwd=repo
    ).decode("utf-8")
    current_source = (repo / GROUPING_SOURCE).read_text(encoding="utf-8")
    hashmap = (repo / "src/oblib/lib/hash/ob_hashmap.h").read_text(encoding="utf-8")
    hashtable = (repo / "src/oblib/lib/hash/ob_hashtable.h").read_text(encoding="utf-8")
    hashutils = (repo / "src/oblib/lib/hash/ob_hashutils.h").read_text(encoding="utf-8")
    fts_types = (repo / "src/data_plane/api/data_plane/fts/ob_fts_struct.h").read_text(
        encoding="utf-8"
    )
    # The control-flow adapter is sequential; production now explicitly omits
    # bucket latches for these invocation-local maps. Hash/equality/allocator
    # template parameters must still be the defaults modeled by the adapter.
    if not re.search(r"ObHashMap<ObFTWord,\s*int64_t"
                     r"(?:,\s*common::hash::NoPthreadDefendMode)?>\s+ObFTWordMap", fts_types):
        raise RuntimeError("Review test adapter: production ObFTWordMap type changed")
    if not re.search(r"typedef\s+pre_proc<_value_type>\s+preproc;", hashtable):
        raise RuntimeError("Review test adapter: production callback preprocessor changed")
    replacements = {
        "@PAIR_TYPE@": "template<class _key_type, class _value_type>\n"
        + extract_block(hashutils, "struct HashMapPair") + ";",
        "@PREPROC@": "template<class T>\n"
        + extract_block(hashutils, "struct pre_proc") + ";",
        "@MAP_ATOMIC@": "template<class _callback>\n"
        + extract_block(hashmap, "int atomic_refactored(const _key_type &key"),
        "@TABLE_ATOMIC_FORWARD@": "template<class _callback>\n"
        + extract_block(hashtable, "int atomic_refactored(const _key_type &key"),
        "@TABLE_ATOMIC@": "template<class _callback, class _preproc>\n"
        + extract_block(hashtable, "int atomic(const _key_type &key"),
        "@MAP_UPSERT@": "template<class _callback>\n"
        + extract_block(hashmap, "int set_or_update(const _key_type &key"),
        "@TABLE_UPSERT_FORWARD@": "template<class _callback>\n"
        + extract_block(hashtable, "int set_or_update(const _key_type &key"),
        "@TABLE_UPSERT@": "template<class _callback, class _preproc>\n"
        + extract_block(hashtable, "int set_or_update(const _key_type &key, const _value_type &value,\n"
                                  "                    _callback &callback, _preproc &preproc)"),
        "@INTERNAL_SET@": extract_block(hashtable, "inline int internal_set(hashbucket &bucket"),
        "@BASELINE_GROUPING@": extract_block(old_source, "int ObAddWord::groupby_word(")
        .replace("ObAddWord::", "BaselineAddWord::", 1),
        "@PREVIOUS_GROUPING@": extract_block(previous_source, "int ObAddWord::groupby_word(")
        .replace("ObAddWord::", "PreviousAddWord::", 1),
        "@CURRENT_GROUPING@": extract_block(current_source, "int ObAddWord::groupby_word("),
    }
    generated = Path(__file__).with_name("grouping_harness.cpp.in").read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if generated.count(marker) != 1:
            raise RuntimeError(f"Harness marker count invalid: {marker}")
        generated = generated.replace(marker, value)
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", help="C++17 compiler, default CXX or available compiler")
    parser.add_argument("--build-dir", type=Path, help="Keep generated source/executable here")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source = build_source(repo)
    import os
    compiler = args.cxx or os.environ.get("CXX") or next(
        (path for name in ("c++", "g++", "clang++", "cl") if (path := shutil.which(name))),
        None,
    )
    if not compiler:
        parser.error("C++17 compiler unavailable; run from compiler developer environment")
    temporary = None
    if args.build_dir:
        build_dir = args.build_dir.resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="seekdb-grouping-")
        build_dir = Path(temporary.name)
    try:
        cpp = build_dir / "grouping_harness.cpp"
        exe = build_dir / ("grouping_harness.exe" if os.name == "nt" else "grouping_harness")
        cpp.write_text(source, encoding="utf-8")
        if Path(compiler).name.lower() in ("cl", "cl.exe"):
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/O2",
                       "/utf-8", str(cpp), "/Fe:" + str(exe), "/Fo:" + str(build_dir) + os.sep]
        else:
            command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                       str(cpp), "-o", str(exe)]
        subprocess.run(command, cwd=build_dir, check=True)
        result = subprocess.run([str(exe)], cwd=build_dir, check=True, capture_output=True, text=True)
        print(result.stdout, end="")
        report = {"status": "passed", "baseline": BASELINE, "previous": PREVIOUS,
                  "compiler": compiler, "platform": platform.platform(), "output": result.stdout,
                  "candidate_source_sha256": hashlib.sha256((repo / GROUPING_SOURCE).read_bytes()).hexdigest(),
                  "generated_harness_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                  "scope": "source-extracted grouping/atomic/upsert/internal_set with portable allocator, lock, key/hash/equality/get/set adapters; not native seekdb or an official score"}
        (build_dir / "grouping_result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the real old/new IK classifier with portable dependency glue.

This is an isolated C++ source-level regression/microbenchmark, not a seekdb
server build or an official competition benchmark. No server or Linux is needed.
The UTF-8 decoder, charset helper templates, Unicode predicates, error values,
and both classifier bodies are extracted unchanged from this checkout/baseline.
Only logging, branch hints, and collation-to-charset dispatch are shimmed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

BASELINE = "5bce9e06d1281347ad61344e4b01fab9158c58a2"
HEADER = "src/storage/fts/ik/ob_ik_char_util.h"


def no_includes(source):
    return re.sub(r"^\s*#include[^\n]*", "", source, flags=re.MULTILINE)


def between(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=BASELINE)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--compiler", choices=["auto", "msvc", "g++", "clang++"], default="auto")
    parser.add_argument("--benchmark", action="store_true", help="also run a local, unofficial classifier microbenchmark")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    build = args.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    old = subprocess.check_output(["git", "show", f"{args.baseline}:{HEADER}"], cwd=root).decode("utf-8")
    current = (root / HEADER).read_text(encoding="utf-8")
    charset = (root / "src/oblib/lib/charset/ob_charset.h").read_text(encoding="utf-8")
    enum = re.search(r"enum ObCharsetType\s*\{.*?\};", charset, re.DOTALL).group()
    errno = (root / "src/oblib/lib/ob_errno.h").read_text(encoding="utf-8")
    errors = []
    for name in ["OB_SUCCESS", "OB_INVALID_ARGUMENT", "OB_NOT_SUPPORTED", "OB_UNEXPECT_INTERNAL_ERROR"]:
        errors.append(re.search(r"constexpr int " + name + r"\s*=\s*[^;]+;", errno).group())
    ctype = (root / "src/oblib/lib/charset/ob_ctype.h").read_text(encoding="utf-8")
    # ob_wc_t is upstream's unsigned long alias. Keep the platform width (all
    # tested decoder outputs fit within 21 bits); UTF16 uses the current generic
    # upstream decoder, not an invented replacement UTF16 implementation.
    cs_errors = "\n".join(re.findall(r"^#define OB_CS_(?:ILSEQ|ILUNI|TOOSMALL\w*)[^\n]*", ctype, re.MULTILINE))
    decoder = between((root / "src/oblib/lib/charset/mb_wc.h").read_text(encoding="utf-8"),
                      "template <bool RANGE_CHECK, bool SUPPORT_MB4>\nstatic ALWAYS_INLINE int ob_mb_wc_utf8_prototype",
                      'extern "C" int ob_mb_wc_utf8mb4_thunk')
    helpers = between((root / "src/oblib/lib/charset/ob_charset_string_helper.h").read_text(encoding="utf-8"),
                      "template<ObCharsetType cs_type>", "class ObFastStringScanner")
    shim = """#pragma once
#include <cstdint>
#include <cstring>
#include <type_traits>
using ob_wc_t = unsigned long;
using uint16 = uint16_t;
using uint32 = uint32_t;
#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#else
#define ALWAYS_INLINE __attribute__((always_inline)) inline
#endif
#define OB_LIKELY(x) (x)
#define OB_UNLIKELY(x) (x)
#define UNUSED(x) (void)(x)
#define OB_ISNULL(x) ((x) == nullptr)
#define OB_FAIL(x) ((ret = (x)) != oceanbase::common::OB_SUCCESS)
#define STORAGE_FTS_LOG(...) ((void)0)
#define ARRAYSIZEOF(x) (sizeof(x) / sizeof((x)[0]))
""" + cs_errors + "\n" + decoder + "\nnamespace oceanbase { namespace common {\n" + enum + "\n" + "\n".join(errors) + """
using ObCollationType = ObCharsetType;
struct ObCharset {
  static ObCharsetType charset_type_by_coll(ObCollationType coll) { return coll; }
};
""" + helpers + "\n} using namespace common; }\n"
    (build / "tokenizer_shim.h").write_text(shim, encoding="utf-8")
    unicode_h = no_includes((root / "src/storage/fts/utils/unicode_utils.h").read_text(encoding="utf-8"))
    (build / "tokenizer_unicode.h").write_text('#pragma once\n#include "tokenizer_shim.h"\n' + unicode_h, encoding="utf-8")
    unicode_cpp = no_includes((root / "src/storage/fts/utils/unicode_utils.cpp").read_text(encoding="utf-8"))
    (build / "tokenizer_unicode.cpp").write_text('#include "tokenizer_unicode.h"\n' + unicode_cpp, encoding="utf-8")
    old = no_includes(old).replace("_OCEANBASE_STORAGE_FTS_IK_OB_IK_CHAR_UTIL_H_", "TOKENIZER_LEGACY_H_").replace("ObFTCharUtil", "LegacyFTCharUtil")
    (build / "tokenizer_classes.h").write_text('#include "tokenizer_unicode.h"\n' + old + "\n" + no_includes(current), encoding="utf-8")
    source = Path(__file__).with_name("tokenizer_differential.cpp")
    executable = build / ("tokenizer_differential.exe" if os.name == "nt" else "tokenizer_differential")
    compiler = args.compiler
    if compiler == "auto":
        compiler = "msvc" if os.name == "nt" else ("g++" if shutil.which("g++") else "clang++")
    if compiler == "msvc":
        compile_args = ["cl", "/nologo", "/std:c++17", "/EHsc", "/O2", "/utf-8", "/I" + str(build), str(source), str(build / "tokenizer_unicode.cpp"), "/Fe:" + str(executable), "/Fo:" + str(build) + os.sep]
        if shutil.which("cl"):
            subprocess.run(compile_args, cwd=build, check=True)
        else:
            vcvars = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat"
            if not vcvars.is_file():
                raise RuntimeError("cl not found; activate a Visual Studio x64 developer shell")
            command = '@echo off\ncall "' + str(vcvars) + '" >nul\n' + subprocess.list2cmdline(compile_args) + '\nexit /b %errorlevel%\n'
            batch = build / "tokenizer_compile.cmd"
            batch.write_text(command, encoding="utf-8")
            subprocess.run(["cmd", "/d", "/c", str(batch)], cwd=build, check=True)
    else:
        subprocess.run([compiler, "-std=c++17", "-O3", "-I", str(build), str(source), str(build / "tokenizer_unicode.cpp"), "-o", str(executable)], cwd=build, check=True)
    run_args = [str(executable)] + (["--benchmark"] if args.benchmark else [])
    result = subprocess.run(run_args, cwd=build, check=True, text=True, capture_output=True)
    report = json.loads(result.stdout)
    report.update({"baseline": args.baseline, "compiler": compiler,
                   "baseline_header_sha256": hashlib.sha256(subprocess.check_output(["git", "show", f"{args.baseline}:{HEADER}"], cwd=root)).hexdigest(),
                   "candidate_header_sha256": hashlib.sha256(current.encode("utf-8")).hexdigest(),
                   "scope": "isolated production character classifier with actual UTF8 decoder and Unicode predicates; dependency glue only; not a native seekdb or official score"})
    (build / "tokenizer_result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

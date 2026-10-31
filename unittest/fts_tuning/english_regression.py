#!/usr/bin/env python3
"""Compare source-extracted English normalizers using real UTF8 case conversion.

Extracts old/new complete get_next, real ObCharset::tolower, UTF8 decoder/encoder,
Unicode case tables, ctype table, and real BENG/grouping consumer functions.
Allocator, datum/string, upstream token stream and other charset dispatch are
portable adapters, not a native seekdb build, SQL test or performance score.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from grouping_regression import extract_block

BASELINE = "32cf87c11ae460301394f539ae25896748bee73d"
SOURCE = "src/share/text_analysis/ob_token_stream.cpp"


def source_for(repo):
    current = (repo / SOURCE).read_text(encoding="utf-8")
    old = subprocess.check_output(["git", "show", f"{BASELINE}:{SOURCE}"], cwd=repo).decode("utf-8")
    charset = (repo / "src/oblib/lib/charset/ob_charset.cpp").read_text(encoding="utf-8")
    ctype = (repo / "src/oblib/lib/charset/ob_ctype.h").read_text(encoding="utf-8")
    utf8 = (repo / "src/oblib/lib/charset/ob_ctype_utf8.cc").read_text(encoding="utf-8")
    tables = (repo / "src/oblib/lib/charset/ob_ctype_utf8_tab.h").read_text(encoding="utf-8")
    definitions = "\n".join(re.findall(r"^#define OB_CS_(?:ILSEQ|ILUNI|TOOSMALL\w*)[^\n]*", ctype, re.MULTILINE))
    definitions += "\n" + "\n".join(re.findall(r"^#define\s+_MY_(?:U|L|NMR)\s+[^\n]*", ctype, re.MULTILINE))
    definitions += "\n" + re.search(r"^#define\s+ob_isalnum[^\n]*", ctype, re.MULTILINE).group()
    replacements = {
        "@DEFINITIONS@": definitions,
        "@CASE_TABLES@": tables,
        "@UTF8_CONVERSION@": "\n".join(extract_block(utf8, name) for name in (
            "static int ob_mb_wc_utf8mb4(", "static int ob_wc_mb_utf8mb4(",
            "static inline void ob_tolower_utf8mb4(", "static size_t ob_casedn_utf8mb4(",
        )),
        "@LOWERCASE@": extract_block(charset, "int ObCharset::tolower(const ObCharsetInfo *cs_info"),
        "@NORMALIZER_BASE@": "\n".join(extract_block(current, signature) for signature in (
            "void ObTokenNormalizer::reset()", "void ObTokenNormalizer::reuse()")),
        "@NORMALIZERS@": "\n".join(
            extract_block(old, signature).replace("ObBasicEnglishNormalizer::", "BaselineNormalizer::", 1)
            + "\n" + extract_block(current, signature)
            for signature in ("int ObBasicEnglishNormalizer::get_next(",
                              "void ObBasicEnglishNormalizer::reset()",
                              "void ObBasicEnglishNormalizer::reuse()")),
        "@BENG_CONSUMER@": extract_block(
            (repo / "src/storage/fts/ob_beng_ft_parser.cpp").read_text(encoding="utf-8"),
            "int ObBEngFTParser::get_next_token("),
        "@GROUP_CONSUMER@": extract_block(current, "int ObTextTokenGroupNormalizer::build_grouping_map()"),
    }
    template = Path(__file__).with_name("english_harness.cpp.in").read_text(encoding="utf-8")
    for marker, source in replacements.items():
        if template.count(marker) != 1:
            raise RuntimeError("Invalid harness marker: " + marker)
        template = template.replace(marker, source)
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", help="C++17 compiler binary")
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    compiler = args.cxx or os.environ.get("CXX") or next(
        (p for c in ("c++", "g++", "clang++", "cl") if (p := shutil.which(c))), None)
    if not compiler:
        parser.error("Activate a C++17 compiler environment (MSVC x64 developer shell on Windows)")
    temporary = None
    if args.build_dir:
        build = args.build_dir.resolve()
        build.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="seekdb-english-")
        build = Path(temporary.name)
    try:
        cpp = build / "english_harness.cpp"
        exe = build / ("english_harness.exe" if os.name == "nt" else "english_harness")
        cpp.write_text(source_for(repo), encoding="utf-8")
        if Path(compiler).name.lower() in ("cl", "cl.exe"):
            # Legacy extracted trim loops mix int32_t/uint32_t; charset source
            # annotates unused arguments through a GCC attribute shim.
            command = [compiler, "/nologo", "/std:c++17", "/EHsc", "/W4", "/WX", "/wd4100", "/wd4018", "/O2", "/utf-8",
                       str(cpp), "/Fe:" + str(exe), "/Fo:" + str(build) + os.sep]
        else:
            command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter",
                       "-Wno-implicit-fallthrough", "-Wno-unused-variable", "-Wno-sign-compare", str(cpp), "-o", str(exe)]
        subprocess.run(command, cwd=build, check=True)
        subprocess.run([str(exe)], cwd=build, check=True)
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()

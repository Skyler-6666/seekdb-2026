"""Configure-time behaviour of the optional hipVS backend.

Runs cmake against fake hipVS/cuVS install trees and checks whether
cmake/SeekdbHipvs.cmake enables SEEKDB_WITH_HIPVS. The fake tree carries
real headers and a real libcuvs_c.so (built from mock_cuvs.cc), so the
configure-time API probe is exercised end to end. No GPU, ROCm, or seekdb
build is involved.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent
MOCK_SOURCE = SOURCE_DIR / "mock_cuvs.cc"
MOCK_INCLUDE = SOURCE_DIR / "mock_include"

PACKAGE_CONFIG = """
add_library(cuvs::c_api SHARED IMPORTED)
set_target_properties(cuvs::c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../libcuvs_c.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include")
"""


def find_cxx():
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def make_install(root, *, package, hnsw_header, broken=False, cxx=None):
    include = root / "include"
    shutil.copytree(MOCK_INCLUDE, include)
    if broken:
        shutil.copyfile(include / "cuvs/neighbors/hnsw_broken.h",
                        include / "cuvs/neighbors/hnsw.h")
    if not hnsw_header:
        (include / "cuvs/neighbors/hnsw.h").unlink()
    lib_dir = root / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    if hnsw_header and cxx is not None:
        library = lib_dir / "libcuvs_c.so"
        result = subprocess.run(
            [cxx, "-shared", "-fPIC", "-std=c++11", f"-I{include}",
             str(MOCK_SOURCE), "-o", str(library)],
            text=True, encoding="utf-8", capture_output=True)
        if result.returncode != 0:
            raise AssertionError(f"mock libcuvs_c.so build failed: {result.stderr}")
    else:
        (lib_dir / "libcuvs_c.so").touch()
    if package:
        config_dir = root / "lib/cmake/cuvs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "cuvs-config.cmake").write_text(PACKAGE_CONFIG, encoding="utf-8")
    return root


class HipvsOptionTests(unittest.TestCase):
    def configure(self, workspace, *, prefix=None, env_root=None, mode=None):
        build = workspace / "build"
        command = [
            "cmake", "-S", str(SOURCE_DIR), "-B", str(build),
            f"-DSEEKDB_REPO_ROOT={REPO_ROOT}",
            "-DCMAKE_SYSTEM_PROCESSOR=x86_64",
        ]
        if prefix is not None:
            command.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        if mode is not None:
            command.append(f"-DSEEKDB_ENABLE_HIPVS={mode}")
        env = dict(os.environ)
        env.pop("HIPVS_ROOT", None)
        if env_root is not None:
            env["HIPVS_ROOT"] = str(env_root)
        result = subprocess.run(command, text=True, encoding="utf-8",
                                capture_output=True, env=env)
        return result

    def assert_enabled(self, result):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("hipVS HNSW backend: enabled", result.stdout)
        self.assertIn("CHECK_DEFS=SEEKDB_WITH_HIPVS=1", result.stdout)
        self.assertIn("cuvs::c_api", result.stdout)
        # The library directory must reach the final link as an rpath: the
        # server has to start without the caller exporting LD_LIBRARY_PATH.
        self.assertIn("LINKER:-rpath,", result.stdout)

    def assert_disabled(self, result, reason):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("SEEKDB_WITH_HIPVS=1", result.stdout)
        self.assertIn(reason, result.stdout + result.stderr)

    def require_cxx(self):
        cxx = find_cxx()
        if not cxx:
            self.skipTest("a C++ compiler is required to exercise the API probe")
        return cxx

    def test_auto_enables_cuvs_package_with_hnsw_extensions(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, cxx=cxx)
            self.assert_enabled(self.configure(Path(tmp), prefix=root))

    def test_auto_ignores_plain_cuvs_without_hnsw_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "cuvs", package=True, hnsw_header=False)
            self.assert_disabled(self.configure(Path(tmp), prefix=root),
                                 "no hipVS install with cuvs/neighbors/hnsw.h")

    def test_auto_finds_headers_only_install_through_hipvs_root(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=False,
                                hnsw_header=True, cxx=cxx)
            self.assert_enabled(self.configure(Path(tmp), env_root=root))

    def test_auto_rejects_install_whose_headers_do_not_match_the_source(self):
        cxx = self.require_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, broken=True, cxx=cxx)
            self.assert_disabled(
                self.configure(Path(tmp), prefix=root),
                "installed hipVS does not provide the API this source uses")

    def test_auto_is_quiet_when_no_install_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_disabled(self.configure(Path(tmp)),
                                 "hipVS HNSW backend: disabled")

    def test_explicit_on_without_install_warns_but_still_configures(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.configure(Path(tmp), mode="ON")
            self.assert_disabled(result, "SEEKDB_ENABLE_HIPVS=ON but no hipVS install")
            self.assertIn("CMake Warning", result.stdout + result.stderr)

    def test_off_never_looks_for_hipvs(self):
        cxx = find_cxx()
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True,
                                hnsw_header=True, cxx=cxx)
            self.assert_disabled(self.configure(Path(tmp), prefix=root, mode="OFF"),
                                 "hipVS HNSW backend: disabled (SEEKDB_ENABLE_HIPVS=OFF)")


if __name__ == "__main__":
    unittest.main()

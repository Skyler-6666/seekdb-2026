"""Configure-time behaviour of the optional hipVS backend.

Runs cmake against fake hipVS/cuVS install trees and checks whether
cmake/SeekdbHipvs.cmake enables SEEKDB_WITH_HIPVS. No GPU, ROCm, or seekdb
build is involved; only the option wiring is covered.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = Path(__file__).resolve().parent

PACKAGE_CONFIG = """
add_library(cuvs::c_api SHARED IMPORTED)
set_target_properties(cuvs::c_api PROPERTIES
  IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../libcuvs_c.so"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include")
"""


def make_install(root: Path, *, package: bool, hnsw_header: bool) -> Path:
    include = root / "include"
    (include / "cuvs/core").mkdir(parents=True, exist_ok=True)
    (include / "cuvs/core/c_api.h").touch()
    if hnsw_header:
        (include / "cuvs/neighbors").mkdir(parents=True, exist_ok=True)
        (include / "cuvs/neighbors/hnsw.h").touch()
    (root / "lib").mkdir(parents=True, exist_ok=True)
    (root / "lib/libcuvs_c.so").touch()
    if package:
        config_dir = root / "lib/cmake/cuvs"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "cuvs-config.cmake").write_text(PACKAGE_CONFIG, encoding="utf-8")
    return root


class HipvsOptionTests(unittest.TestCase):
    def configure(self, workspace: Path, *, prefix=None, env_root=None, mode=None):
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

    def assert_disabled(self, result, reason):
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertNotIn("SEEKDB_WITH_HIPVS=1", result.stdout)
        self.assertIn(reason, result.stdout + result.stderr)

    def test_auto_enables_cuvs_package_with_hnsw_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True, hnsw_header=True)
            self.assert_enabled(self.configure(Path(tmp), prefix=root))

    def test_auto_ignores_plain_cuvs_without_hnsw_extensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "cuvs", package=True, hnsw_header=False)
            self.assert_disabled(self.configure(Path(tmp), prefix=root),
                                 "no hipVS install with cuvs/neighbors/hnsw.h")

    def test_auto_finds_headers_only_install_through_hipvs_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=False, hnsw_header=True)
            self.assert_enabled(self.configure(Path(tmp), env_root=root))

    def test_auto_is_quiet_when_no_install_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assert_disabled(self.configure(Path(tmp)),
                                 "hipVS HNSW backend: disabled")

    def test_explicit_on_without_install_warns_but_still_configures(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.configure(Path(tmp), mode="ON")
            self.assert_disabled(result, "SEEKDB_ENABLE_HIPVS=ON but no hipVS install")
            self.assertIn("WARNING", result.stdout + result.stderr)

    def test_off_never_looks_for_hipvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_install(Path(tmp) / "hipvs", package=True, hnsw_header=True)
            self.assert_disabled(self.configure(Path(tmp), prefix=root, mode="OFF"),
                                 "hipVS HNSW backend: disabled (SEEKDB_ENABLE_HIPVS=OFF)")


if __name__ == "__main__":
    unittest.main()

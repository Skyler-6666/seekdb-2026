"""Run the unchanged, pinned official benchmark on a dedicated native instance.

This wrapper does not generate performance results. The official shell program
must run successfully, with every fixed parameter and all nine scored metrics.
The official program deletes/recreates its dedicated fts_large_bench database.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "5bce9e06d1281347ad61344e4b01fab9158c58a2"
CONFIG = {"rows": 20000, "batch": 500, "rounds": 3000, "query_rounds": 200, "samples": 3, "warmup": 30, "skip_load": 0}
OFFICIAL_HASHES = {
    "fts_large_bench.sh": "596ea5d1c0038e666b78aa94bef22d2dfb87ffc9e84dbc7e2a1da1f746424462",
    "fts_large_bench_gen.py": "b0c816dd8211ac0471f60b72a37a3fb5c62580357a5b6d7ac95f5a8c2e3f143d",
    "fts_large_bench_setup.sql": "311826751df7f51ac45156fb4284aeca9bd9197370aceb88f4fb92832b458108",
    "fts_large_bench_score.py": "bea52eb1c68d05693bb3ac5b910ab0e2fa0e70e01582f40466898ee1e2e3ef55",
    "fts_large_bench_baseline.json": "f4e97eddb2e49c982b967fc8047f7cf37177eebe53d648f6746d85aacda91f0c",
}


def dump(path: Path, value) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_contract(root: Path = ROOT) -> dict[str, str]:
    result = {}
    for name, expected in OFFICIAL_HASHES.items():
        data = (root / "tools/benchmark" / name).read_bytes()
        # A Windows checkout can convert LF to CRLF; the native Linux run still
        # requires the original LF shell file. No text/content edits are allowed.
        actual = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()
        if actual != expected:
            raise ValueError(f"Official benchmark changed: {name}")
        result[name] = actual
    return result


def official_module():
    verify_contract()
    spec = importlib.util.spec_from_file_location("unchanged_official_fts_score", ROOT / "tools/benchmark/fts_large_bench_score.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def checked_report(path: Path):
    """Reject appended/mixed reports before invoking the original scorer."""
    scorer = official_module()
    baseline = scorer.load_baseline(ROOT / "tools/benchmark/fts_large_bench_baseline.json")
    text = path.read_text(encoding="utf-8")
    if text.count("FTS Large Benchmark Report") != 1:
        raise ValueError("Expected exactly one complete official report, not appended runs")
    required = [*CONFIG, *(metric for group in baseline["score_categories"].values() for metric in group)]
    for name in required:
        if len(re.findall(r"^\s*" + re.escape(name) + r":", text, re.MULTILINE)) != 1:
            raise ValueError(f"Missing/duplicate report field: {name}")
    report = scorer.load_report(path)
    scorer.check_config(report, baseline, False)
    for name in required:
        if name not in report or not math.isfinite(report[name]):
            raise ValueError(f"Invalid numeric report field: {name}")
        if name not in CONFIG and report[name] <= 0:
            raise ValueError(f"Nonpositive duration: {name}")
    return scorer, baseline, report


def run(args) -> int:
    if platform.system() != "Linux":
        raise ValueError("Official benchmark requires the declared native Linux/Bash/MySQL environment; Windows helper tests are not this run")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.label):
        raise ValueError("label must contain 1..40 letters, digits, '_' or '-'")
    source_root = args.source_root.resolve(strict=True)
    hashes = verify_contract(source_root)
    if subprocess.run(["git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--"], check=False).returncode != 0:
        raise ValueError("Commit tracked source changes before a reproducible native run")
    binary = args.server_binary.resolve(strict=True)
    if not binary.is_file():
        raise ValueError("server-binary must be the compiled native executable")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = output / "official-report.txt"
    report.touch(exist_ok=False)
    environment = dict(os.environ)
    environment.update({key.upper(): str(value) for key, value in CONFIG.items()})
    environment.update({"LABEL": args.label, "OUTPUT": str(report)})
    command = ["bash", str(source_root / "tools/benchmark/fts_large_bench.sh")]
    record = {
        "status": "running", "label": args.label, "benchmark_config": CONFIG,
        "official_contract_commit": UPSTREAM, "official_file_sha256": hashes,
        "code_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"]).decode().strip(),
        "source_root": str(source_root),
        "server_binary_sha256": fingerprint(binary), "server_binary_name": binary.name,
        "command": command, "host": {"system": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
        "started_at": datetime.now(timezone.utc).isoformat(),
        "notes": "MYSQL/MYSQL_VERBOSE/MYSQL_PWD are inherited but omitted from evidence; the operator must bind the declared executable to the dedicated running instance.",
    }
    dump(output / "run.json", record)
    try:
        with (output / "stdout.log").open("wb") as stdout, (output / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=source_root, env=environment, stdout=stdout, stderr=stderr, check=False)
        record["benchmark_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError("Official benchmark failed; preserve stdout/stderr and do not calculate a score")
        checked_report(report)
        scoring = [sys.executable, str(source_root / "tools/benchmark/fts_large_bench_score.py"), str(report), "--json"]
        scored = subprocess.run(scoring, cwd=source_root, capture_output=True, check=False)
        (output / "score.stderr.log").write_bytes(scored.stderr)
        if scored.returncode:
            raise RuntimeError("The unchanged official scorer rejected the report")
        score = json.loads(scored.stdout)
        dump(output / "official-score.json", score)
        record.update(status="completed", scoring_command=scoring, score_exit_code=scored.returncode,
                      official_performance_score=score["score"], competition_platform_score=None,
                      correctness_verified=False,
                      meaning="Performance evidence only; run native_correctness.py and relevant upstream regressions before accepting correctness.")
        return 0
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        dump(output / "run.json", record)


def compare(args) -> int:
    runs = []
    for directory in (args.before, args.after):
        record = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if record.get("status") != "completed" or record.get("official_file_sha256") != OFFICIAL_HASHES:
            raise ValueError("Both runs must be complete and use the same unchanged official files")
        scorer, baseline, metrics = checked_report(directory / "official-report.txt")
        runs.append((record, metrics, scorer.calculate_score(metrics, baseline)))
    before, after = runs
    if before[0]["server_binary_sha256"] == after[0]["server_binary_sha256"]:
        raise ValueError("Before and after executable fingerprints are identical")
    if before[0]["code_commit"] == after[0]["code_commit"]:
        raise ValueError("Before and after source revisions are identical")
    baseline["baseline_metrics"] = before[1]
    relative = scorer.calculate_score(after[1], baseline)
    result = {
        "status": "compared", "before_commit": before[0]["code_commit"], "after_commit": after[0]["code_commit"],
        "official_baseline_scores": {"before": before[2], "after": after[2]},
        "same_machine_relative_categories": relative["categories"],
        "same_machine_mean_time_reduction": relative["mean_improvement"],
        "correctness_verified": False, "competition_platform_score": None,
        "limitations": "The operator must verify same hardware/resources/compiler/load and actual deployed binary identity. No local score is a platform grade.",
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "same_machine_mean_time_reduction": result["same_machine_mean_time_reduction"]}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("verify-contract")
    execute = commands.add_parser("run")
    execute.add_argument("--label", required=True)
    execute.add_argument("--source-root", type=Path, default=ROOT, help="Git checkout used to build this running native binary")
    execute.add_argument("--server-binary", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("before", type=Path)
    comparison.add_argument("after", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "verify-contract":
        print(json.dumps({"status": "verified", "official_file_sha256": verify_contract()}, indent=2))
        return 0
    return run(args) if args.action == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

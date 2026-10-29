#!/usr/bin/env python3
"""Compare the results of two or more benchmark_hnsw_build.py runs.

The first file is used as the baseline, every other file is compared against
it. The interesting number is the speedup of the index build time, the recall
has to stay on the same level.

Example:
    python3 compare_benchmark_results.py result_cpu.json result_gpu.json
"""

import argparse
import json
import sys

METRICS = [
    ("build_seconds", "build time (s)", "lower"),
    ("build_rows_per_second", "build rows/s", "higher"),
    ("recall", "recall@k", "higher"),
    ("approx_latency_p50_ms", "approx latency p50 (ms)", "lower"),
    ("approx_latency_p95_ms", "approx latency p95 (ms)", "lower"),
]


def load(path):
    with open(path) as handle:
        return json.load(handle)


def recall_key(result):
    for key in result:
        if key.startswith("recall_at_"):
            return key
    return None


def pick(result, name):
    if name == "recall":
        key = recall_key(result)
        return result.get(key) if key else None
    return result.get(name)


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return "%.3f" % value
    return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="result json files, the first one is the baseline")
    parser.add_argument("--output", default=None, help="write the comparison as json to this file")
    args = parser.parse_args()

    runs = [(path, load(path)) for path in args.results]
    base_path, base = runs[0]

    print("baseline: %s" % base_path)
    width = max(len(str(result.get("label", path))) for path, result in runs)
    header = "metric".ljust(24) + "baseline".rjust(14)
    for path, result in runs[1:]:
        header += str(result.get("label", path)).rjust(max(width, 12))
    print(header)

    comparison = {"baseline": base_path, "runs": []}

    rows = []
    for name, title, direction in METRICS:
        base_value = pick(base, name)
        cells = [fmt(base_value)]
        speeds = []
        for path, result in runs[1:]:
            value = pick(result, name)
            cells.append(fmt(value))
            speed = None
            if isinstance(base_value, (int, float)) and isinstance(value, (int, float)) and base_value and value:
                if direction == "lower":
                    speed = base_value / value
                else:
                    speed = value / base_value
            speeds.append(speed)
        rows.append((title, cells, speeds))

    for title, cells, speeds in rows:
        line = title.ljust(24) + cells[0].rjust(14)
        for index in range(1, len(cells)):
            cell = cells[index].rjust(max(width, 12))
            speed = speeds[index - 1]
            if speed is not None:
                cell += "  x%.2f" % speed
            line += cell
        print(line)

    print("")
    print("xN means the run is N times better than the baseline for that metric")

    for path, result in runs:
        entry = {
            "label": result.get("label"),
            "file": path,
            "rows": result.get("rows"),
            "dim": result.get("dim"),
            "build_seconds": result.get("build_seconds"),
            "build_rows_per_second": result.get("build_rows_per_second"),
            "recall": pick(result, "recall"),
        }
        if path != base_path:
            base_build = base.get("build_seconds")
            this_build = result.get("build_seconds")
            if base_build and this_build:
                entry["build_speedup"] = round(base_build / this_build, 3)
        comparison["runs"].append(entry)

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(comparison, handle, indent=2)
        print("wrote %s" % args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())

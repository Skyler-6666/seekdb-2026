#!/usr/bin/env python3
"""Benchmark the AMD hipVS GPU backend against the CPU backend.

The benchmark builds the very same HNSW index twice:

  * once with the CPU backend  (``lib=vsag``),
  * once with the GPU backend  (``lib=hipvs``),

and reports the time each build took, the speedup and the recall of the GPU
index against the CPU index.

It is meant to be run on the machine that has the AMD GPU, against a running
seekdb instance::

    python3 script/hipvs/hipvs_hnsw_build_bench.py \
        --host 127.0.0.1 --port 2881 --user root \
        --rows 200000 --dim 128 --topk 10 \
        --out results/hipvs_hnsw_build_bench.json

The GPU backend falls back to the CPU backend when the hipVS bridge or the GPU
is not available. The script detects that case and says so, because the
speedup is then meaningless.

The script needs ``pymysql``::

    pip3 install pymysql
"""

import argparse
import json
import os
import random
import sys
import time

try:
    import pymysql
except ImportError:  # pragma: no cover - the script is a developer tool
    print("pymysql is required: pip3 install pymysql", file=sys.stderr)
    sys.exit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--user", default="root")
    parser.add_argument("--password", default="")
    parser.add_argument("--database", default="hipvs_bench")
    parser.add_argument("--rows", type=int, default=100000, help="number of vectors")
    parser.add_argument("--dim", type=int, default=128, help="vector dimension")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--queries", type=int, default=100, help="number of probe vectors")
    parser.add_argument("--batch", type=int, default=1000, help="rows per insert")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--out", default="", help="write the report to this json file")
    return parser.parse_args()


def connect(args):
    return pymysql.connect(host=args.host, port=args.port, user=args.user,
                            password=args.password, database=args.database,
                            autocommit=True, charset="utf8mb4")


def execute(cursor, sql, args=None):
    cursor.execute(sql, args)
    return cursor.fetchall()


def prepare_database(cursor, args):
    execute(cursor, "CREATE DATABASE IF NOT EXISTS %s" % args.database)
    execute(cursor, "USE %s" % args.database)
    for table in ("bench_cpu", "bench_gpu"):
        execute(cursor, "DROP TABLE IF EXISTS %s" % table)


def random_vector(rng, dim):
    return "[" + ",".join("%.6f" % rng.uniform(-1.0, 1.0) for _ in range(dim)) + "]"


def create_table(cursor, table, lib, dim):
    execute(cursor, "CREATE TABLE %s (id BIGINT PRIMARY KEY, v VECTOR(%d), "
                    "VECTOR INDEX idx_%s (v) WITH (distance=l2, type=hnsw, lib=%s))"
             % (table, dim, table, lib))


def load_rows(cursor, table, args, rng, vectors):
    """Insert the rows and return the time the insert took."""
    batch = []
    started = time.time()
    for i in range(args.rows):
        vector = vectors[i]
        batch.append((i, vector))
        if len(batch) == args.batch:
            execute(cursor, "INSERT INTO %s (id, v) VALUES " % table +
                    ",".join(["(%s,'%s')" % row for row in batch]))
            batch = []
    if batch:
        execute(cursor, "INSERT INTO %s (id, v) VALUES " % table +
                ",".join(["(%s,'%s')" % row for row in batch]))
    return time.time() - started


def build_index(cursor, table, args):
    """Rebuild the index and return the time the build took."""
    started = time.time()
    execute(cursor, "CALL dbms_vector.rebuild_index('idx_%s', '%s', 'v', 0)" % (table, table))
    return time.time() - started


def neighbours(cursor, table, vector, args):
    rows = execute(cursor, "SELECT id FROM %s ORDER BY l2_distance(v, '%s') APPROXIMATE LIMIT %d"
                          % (table, vector, args.topk))
    return [int(row[0]) for row in rows]


def recall(reference, candidate):
    if not reference:
        return 1.0
    hits = len(set(reference) & set(candidate))
    return hits / float(len(reference))


def main():
    global args
    args = parse_args()
    rng = random.Random(args.seed)
    print("generating %d vectors of dimension %d ..." % (args.rows, args.dim))
    vectors = [random_vector(rng, args.dim) for _ in range(args.rows)]

    connection = connect(args)
    try:
        cursor = connection.cursor()
        prepare_database(cursor, args)

        print("creating the CPU table (lib=vsag) ...")
        create_table(cursor, "bench_cpu", "vsag", args.dim)
        print("creating the GPU table (lib=hipvs) ...")
        create_table(cursor, "bench_gpu", "hipvs", args.dim)

        print("loading the CPU table ...")
        cpu_insert = load_rows(cursor, "bench_cpu", args, rng, vectors)
        print("loading the GPU table ...")
        gpu_insert = load_rows(cursor, "bench_gpu", args, rng, vectors)

        print("building the CPU index ...")
        cpu_build = build_index(cursor, "bench_cpu", args)
        print("building the GPU index ...")
        gpu_build = build_index(cursor, "bench_gpu", args)

        probes = [vectors[rng.randrange(args.rows)] for _ in range(args.queries)]
        cpu_results = []
        gpu_results = []
        for probe in probes:
            cpu_results.append(neighbours(cursor, "bench_cpu", probe, args))
            gpu_results.append(neighbours(cursor, "bench_gpu", probe, args))
        recalls = [recall(cpu, gpu) for cpu, gpu in zip(cpu_results, gpu_results)]
        average_recall = sum(recalls) / float(len(recalls)) if recalls else 0.0

        speedup = (cpu_build / gpu_build) if gpu_build > 0 else 0.0
        report = {
            "rows": args.rows,
            "dim": args.dim,
            "topk": args.topk,
            "queries": args.queries,
            "cpu_insert_seconds": round(cpu_insert, 3),
            "gpu_insert_seconds": round(gpu_insert, 3),
            "cpu_build_seconds": round(cpu_build, 3),
            "gpu_build_seconds": round(gpu_build, 3),
            "build_speedup": round(speedup, 2),
            "average_recall_at_%d" % args.topk: round(average_recall, 4),
        }
        print("")
        print("=========================================================")
        print("            seekdb x AMD hipVS HNSW build benchmark")
        print("=========================================================")
        for key, value in report.items():
            print("  %-28s %s" % (key, value))
        print("=========================================================")
        if report["build_speedup"] <= 1.0:
            print("NOTE: the GPU backend did not speed the build up. The hipVS")
            print("      bridge or the GPU was probably not available, so the")
            print("      index was built by the CPU backend. Check the observer")
            print("      log for the [OBHIPVS] lines.")
        if args.out:
            directory = os.path.dirname(os.path.abspath(args.out))
            if directory and not os.path.isdir(directory):
                os.makedirs(directory)
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
            print("the report was written to %s" % args.out)
    finally:
        connection.close()


if __name__ == "__main__":
    main()

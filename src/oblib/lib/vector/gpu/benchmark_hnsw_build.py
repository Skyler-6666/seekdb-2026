#!/usr/bin/env python3
"""Measure the HNSW index build time and the recall of seekdb.

The script drives a running seekdb through the mysql client. It creates a
table with a vector column, loads a synthetic dataset, builds the vector index
and compares the approximate top-k results against the exact ones.

Run it once with the observer started without the GPU plugin and once with the
plugin enabled to compare the two builds. The observer decides which path is
taken, the script only measures.

Example:
    python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label cpu
    python3 benchmark_hnsw_build.py --rows 200000 --dim 128 --label gpu
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time

# The SQL distance function of seekdb does not follow the metric name of the
# index, see src/oblib/lib/ob_name_def.h.
DISTANCE_FUNCTIONS = {
    "l2": "l2_distance",
    "ip": "negative_inner_product",
    "cosine": "cosine_distance",
}


def run_mysql(mysql_cmd, sql=None, stdin_file=None, capture=True):
    """Run a mysql client command and return (stdout, stderr, returncode)."""
    cmd = list(mysql_cmd)
    if sql is not None:
        cmd += ["-e", sql]
    proc = subprocess.run(
        cmd,
        stdin=stdin_file,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout, proc.stderr, proc.returncode


def mysql_query(mysql_cmd, sql):
    out, err, code = run_mysql(mysql_cmd, sql=sql)
    if code != 0:
        raise RuntimeError("mysql failed for %r: %s" % (sql, err.strip()))
    return out


def generate_vectors(rows, dim, seed):
    """Yield (id, vector_text) pairs of a random uniform dataset."""
    rng = random.Random(seed)
    for row_id in range(1, rows + 1):
        values = [rng.random() for _ in range(dim)]
        yield row_id, "[" + ",".join("%.6f" % v for v in values) + "]"


def write_dataset(path, rows, dim, seed):
    with open(path, "w") as handle:
        for row_id, vector_text in generate_vectors(rows, dim, seed):
            handle.write("%d\t%s\n" % (row_id, vector_text))


def build_dataset(path, rows, dim, seed):
    """Write the dataset, preferring numpy when it is available."""
    try:
        import numpy as np
    except ImportError:
        write_dataset(path, rows, dim, seed)
        return
    rng = np.random.default_rng(seed)
    block = 10000
    with open(path, "w") as handle:
        written = 0
        while written < rows:
            count = min(block, rows - written)
            matrix = rng.random((count, dim))
            for offset in range(count):
                values = ",".join("%.6f" % v for v in matrix[offset])
                handle.write("%d\t[%s]\n" % (written + offset + 1, values))
            written += count


def parse_ids(output):
    ids = []
    for line in output.strip().splitlines():
        line = line.strip()
        if line:
            ids.append(line.split("\t")[0])
    return ids


def load_with_insert(mysql_cmd, table, data_path, batch):
    """Fallback loader for servers that refuse LOAD DATA LOCAL INFILE."""
    pending = []
    with open(data_path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            pending.append(line)
            if len(pending) >= batch:
                flush_insert(mysql_cmd, table, pending)
                pending = []
    if pending:
        flush_insert(mysql_cmd, table, pending)


def flush_insert(mysql_cmd, table, pending):
    values = ",".join("(%s)" % row.replace("\t", ",") for row in pending)
    sql = "insert into %s (c1, c2) values %s" % (table, values)
    out, err, code = run_mysql(mysql_cmd, stdin_file=_as_stdin(sql))
    if code != 0:
        raise RuntimeError("insert failed: %s" % err.strip())


def _as_stdin(sql):
    import io

    return io.StringIO(sql)


def load_dataset(mysql_cmd, table, data_path, args):
    """Load the generated dataset, preferring LOAD DATA over INSERT."""
    load_sql = (
        "load data local infile '%s' into table %s fields terminated by '\\t' "
        "(c1, c2)" % (data_path, table)
    )
    try:
        mysql_query(mysql_cmd, load_sql)
        return "load data"
    except RuntimeError as error:
        print("LOAD DATA LOCAL INFILE failed (%s), falling back to INSERT" % error)
        load_with_insert(mysql_cmd, table, data_path, args.insert_batch)
        return "insert"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", default=os.environ.get("MYSQL_PORT", "2881"))
    parser.add_argument("--user", default=os.environ.get("MYSQL_USER", "root@test"))
    parser.add_argument("--password", default=os.environ.get("MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql", default=os.environ.get("MYSQL_BIN", "mysql"))
    parser.add_argument("--database", default="gpu_hnsw_bench")
    parser.add_argument("--table", default="t_bench")
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--distance", default="l2")
    parser.add_argument(
        "--distance-function",
        default=None,
        help="distance function used in the queries, defaults to the seekdb "
        "function of the metric (l2_distance, negative_inner_product, cosine_distance)",
    )
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--insert-batch", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", default="run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--keep-table", action="store_true")
    args = parser.parse_args()

    mysql_cmd = [args.mysql, "-h", args.host, "-P", str(args.port), "-u", args.user]
    if args.password:
        mysql_cmd.append("-p" + args.password)
    mysql_cmd += ["--batch", "--raw", "--skip-column-names", "--local-infile=1"]

    table = "%s.%s" % (args.database, args.table)
    index_name = "idx_vec"
    distance_function = args.distance_function or DISTANCE_FUNCTIONS.get(args.distance)
    if distance_function is None:
        raise SystemExit("unknown distance %r, pass --distance-function explicitly" % args.distance)

    mysql_query(mysql_cmd, "create database if not exists %s" % args.database)
    mysql_query(mysql_cmd, "drop table if exists %s" % table)
    mysql_query(
        mysql_cmd,
        "create table %s (c1 bigint primary key, c2 vector(%d))" % (table, args.dim),
    )

    data_path = os.path.join(tempfile.gettempdir(), "gpu_hnsw_bench_%d.tsv" % os.getpid())
    print("generating %d rows of %d dimensions ..." % (args.rows, args.dim))
    build_dataset(data_path, args.rows, args.dim, args.seed)

    print("loading data ...")
    load_start = time.time()
    load_method = load_dataset(mysql_cmd, table, data_path, args)
    load_seconds = time.time() - load_start
    os.unlink(data_path)
    print("loaded in %.2f s via %s" % (load_seconds, load_method))

    ddl = (
        "create vector index %s on %s (c2) with (distance=%s, lib=vsag, type=hnsw, "
        "m=%d, ef_construction=%d)" % (index_name, table, args.distance, args.m, args.ef_construction)
    )
    print("building the vector index ...")
    print("  %s" % ddl)
    build_start = time.time()
    mysql_query(mysql_cmd, ddl)
    build_seconds = time.time() - build_start
    print("index built in %.2f s" % build_seconds)

    rng = random.Random(args.seed + 1)
    queries = []
    for _ in range(args.queries):
        values = [rng.random() for _ in range(args.dim)]
        queries.append("[" + ",".join("%.6f" % v for v in values) + "]")

    for vector_text in queries[: args.warmup]:
        mysql_query(
            mysql_cmd,
            "select c1 from %s order by %s(c2, %s) approximate limit %d"
            % (table, distance_function, vector_text, args.topk),
        )

    overlaps = []
    approx_latency = []
    exact_latency = []
    for vector_text in queries:
        approx_sql = "select c1 from %s order by %s(c2, %s) approximate limit %d" % (
            table,
            distance_function,
            vector_text,
            args.topk,
        )
        exact_sql = "select c1 from %s order by %s(c2, %s) limit %d" % (
            table,
            distance_function,
            vector_text,
            args.topk,
        )

        start = time.time()
        approx_ids = parse_ids(mysql_query(mysql_cmd, approx_sql))
        approx_latency.append(time.time() - start)

        start = time.time()
        exact_ids = parse_ids(mysql_query(mysql_cmd, exact_sql))
        exact_latency.append(time.time() - start)

        if exact_ids:
            hits = len(set(approx_ids) & set(exact_ids))
            overlaps.append(hits / float(len(exact_ids)))

    recall = sum(overlaps) / len(overlaps) if overlaps else 0.0
    approx_latency.sort()
    exact_latency.sort()

    result = {
        "label": args.label,
        "rows": args.rows,
        "dim": args.dim,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "distance": args.distance,
        "distance_function": distance_function,
        "ddl": ddl,
        "load_method": load_method,
        "load_seconds": round(load_seconds, 3),
        "build_seconds": round(build_seconds, 3),
        "build_rows_per_second": round(args.rows / build_seconds, 1) if build_seconds > 0 else None,
        "recall_at_%d" % args.topk: round(recall, 4),
        "approx_latency_p50_ms": round(approx_latency[len(approx_latency) // 2] * 1000, 3),
        "approx_latency_p95_ms": round(approx_latency[int(len(approx_latency) * 0.95)] * 1000, 3),
        "exact_latency_p50_ms": round(exact_latency[len(exact_latency) // 2] * 1000, 3),
    }

    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)
        print("wrote %s" % args.output)

    if not args.keep_table:
        mysql_query(mysql_cmd, "drop table if exists %s" % table)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Native union boundaries: K=1/2/8/9, encoded key shapes, DML and plan reuse.

Run on an isolated real seekdb instance, then repeat with the candidate build:
  python3 doc_only_union_regression.py --port 38680 --output base.json
  python3 doc_only_union_regression.py --port 38680 --output new.json --compare base.json

Each query is checked against both the source corpus and an explicitly scored
MATCH > 0 query. All rows/IDs are retained; no precomputed database result is used.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid

from doc_only_regression import normalize


TERMS = ["quartz", "zephyr", "orchid", "marigold", "saffron", "topaz", "onyx", "cobalt", "willow"]
CORPUS_ROWS = 2113
INSERT_ORDINAL = CORPUS_ROWS + 1000


def query_sets():
    # Present, empty and mixed streams on both sides of the K=8 fast-path bound.
    result = [(f"present_k{k}", TERMS[:k]) for k in (1, 2, 8, 9)]
    result += [
        ("empty_k1", ["absenta"]),
        ("empty_k2", ["absenta", "absentb"]),
        ("dense_k1", ["batchprobe"]),
        ("missing_k2", ["batchprobe", "absenta"]),
        ("missing_k8", ["batchprobe", *TERMS[:5], "absenta", "absentb"]),
        ("missing_k9", ["batchprobe", *TERMS[:6], "absenta", "absentb"]),
        ("sparse_k2", ["sparseprobe", "absenta"]),
        ("sparse_dense_k2", ["sparseprobe", "batchprobe"]),
        ("exact_k1", ["exactprobe"]),
    ]
    return result


def make_corpus():
    data = {}
    for i in range(CORPUS_ROWS):
        words = [term for j, term in enumerate(TERMS) if (i + j) % (j % 4 + 2)]
        if i % 31 == 0:
            words = ["elsewhere"]
        elif i % 37 == 0:
            words = TERMS[:]
        words += words[:1] * (i % 5)
        # More than 2048 postings in one stream, also after UPDATE/DELETE.
        # The sparse stream ends much earlier than its dense siblings.
        words += ["batchprobe"]
        if i % 127 == 0:
            words += ["sparseprobe"]
        if i < 2100:
            # Exactly 300 seven-row batches; DML removes 42, still a multiple.
            words += ["exactprobe"]
        data[i] = " ".join(words)
    return data


def capture(port):
    import pymysql
    database = "fts_union_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    cases = []
    table_defs = [
        ("signed_ids", "key1 BIGINT PRIMARY KEY", "key1", lambda i: (i - 300,)),
        ("unsigned_ids", "key1 BIGINT UNSIGNED PRIMARY KEY", "key1", lambda i: (2**63 + i,)),
        ("string_ids", "key1 VARCHAR(32) COLLATE utf8mb4_bin PRIMARY KEY", "key1",
         lambda i: (f"shared-{i:04d}-中",)),
        ("composite_ids", "key1 INT, key2 VARCHAR(16) COLLATE utf8mb4_bin, PRIMARY KEY(key1,key2)",
         "key1,key2", lambda i: (i % 7 - 3, f"item-{i:04d}")),
    ]
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            for table, key_ddl, key_cols, key in table_defs:
                data = make_corpus()
                cur.execute(f"CREATE TABLE {table}(body TEXT, ordinal INT NOT NULL,{key_ddl})")
                placeholders = ",".join(["%s"] * (2 + len(key(0))))
                cur.executemany(f"INSERT INTO {table} VALUES({placeholders})",
                                [(body, i, *key(i)) for i, body in data.items()])
                cur.execute(f"ALTER TABLE {table} ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
                for phase in ("initial", "after_dml"):
                    if phase == "after_dml":
                        cur.execute(f"UPDATE {table} SET body='willow zephyr' WHERE ordinal BETWEEN 20 AND 40")
                        for i in range(20, 41):
                            data[i] = "willow zephyr"
                        cur.execute(f"DELETE FROM {table} WHERE ordinal BETWEEN 70 AND 90")
                        for i in range(70, 91):
                            del data[i]
                        cur.execute(f"INSERT INTO {table} VALUES({placeholders})",
                                    (" ".join(TERMS + ["batchprobe", "sparseprobe"]),
                                     INSERT_ORDINAL, *key(INSERT_ORDINAL)))
                        data[INSERT_ORDINAL] = " ".join(TERMS + ["batchprobe", "sparseprobe"])
                    for scenario, terms in query_sets():
                        query = " ".join(terms)
                        expected = sorted(i for i, body in data.items() if set(terms).intersection(body.split()))
                        # The ordinary plan and a forced seven-row batch both
                        # retain every ID. A small batch forces many refills and
                        # partial final batches even on machines with different
                        # _rowsets_max_rows defaults.
                        variants = (("", None, 0, None), (" LIMIT 17 OFFSET 3", 17, 3, None),
                                    (" LIMIT 0", 0, 0, None), ("", None, 0, 7))
                        for suffix, limit, offset, batch_size in variants:
                            hint = "" if batch_size is None else f"/*+ OPT_PARAM('_rowsets_max_rows', {batch_size}) */ "
                            sql = f"SELECT {hint}{key_cols},ordinal FROM {table} WHERE MATCH(body) AGAINST(%s) ORDER BY ordinal{suffix}"
                            cur.execute(sql, (query,))
                            rows = normalize(cur.fetchall())
                            selected = expected[offset:] if limit is None else expected[offset:offset + limit]
                            expected_rows = [[*key(i), i] for i in selected]
                            if rows != expected_rows:
                                raise AssertionError(f"Corpus identity mismatch: {table}/{phase}/{scenario}/{suffix}/{batch_size}")
                            cur.execute(sql.replace("AGAINST(%s)", "AGAINST(%s)>0"), (query,))
                            if normalize(cur.fetchall()) != rows:
                                raise AssertionError(f"Scored identity mismatch: {table}/{phase}/{scenario}/{suffix}/{batch_size}")
                            cases.append({"table": table, "phase": phase, "scenario": scenario,
                                          "terms": terms, "batch_size": batch_size,
                                          "limit": limit, "offset": offset, "rows": rows})
                        # No ordering consumer survives COUNT(*). These queries
                        # can use the strict count-only rewrite and exercise the
                        # union's own early LIMIT/OFFSET path. Check zero, exact
                        # and partial batches, late offsets and exhaustion.
                        if scenario in ("empty_k1", "dense_k1", "missing_k2", "missing_k8",
                                        "missing_k9", "sparse_k2"):
                            for limit, offset in ((0, 0), (7, 0), (8, 3), (127, 2040),
                                                  (17, CORPUS_ROWS + 20)):
                                sql = (f"SELECT COUNT(*) AS n FROM (SELECT /*+ OPT_PARAM('_rowsets_max_rows', 7) */ "
                                       f"ordinal FROM {table} WHERE MATCH(body) AGAINST(%s) "
                                       f"LIMIT {limit} OFFSET {offset}) AS limited")
                                cur.execute(sql, (query,))
                                rows = normalize(cur.fetchall())
                                if rows != [[min(limit, max(len(expected) - offset, 0))]]:
                                    raise AssertionError(f"Count/limit mismatch: {table}/{phase}/{scenario}/{limit}/{offset}")
                                cur.execute(sql.replace("AGAINST(%s)", "AGAINST(%s)>0"), (query,))
                                if normalize(cur.fetchall()) != rows:
                                    raise AssertionError(f"Scored count mismatch: {table}/{phase}/{scenario}/{limit}/{offset}")
                                cases.append({"table": table, "phase": phase, "scenario": scenario,
                                              "count_only": True, "batch_size": 7,
                                              "limit": limit, "offset": offset, "rows": rows})
                    # Reusing the statement with token counts crossing the fast
                    # path boundary must not retain cursors or the previous mode.
                    prepared = f"SELECT ordinal FROM {table} WHERE MATCH(body) AGAINST(?) ORDER BY ordinal"
                    cur.execute("PREPARE union_stmt FROM %s", (prepared,))
                    for k in (1, 9, 2, 8, 1):
                        query = " ".join(TERMS[:k])
                        cur.execute("SET @union_term=%s", (query,))
                        cur.execute("EXECUTE union_stmt USING @union_term")
                        rows = normalize(cur.fetchall())
                        expected = [[i] for i, body in sorted(data.items()) if set(TERMS[:k]).intersection(body.split())]
                        if rows != expected:
                            raise AssertionError(f"Prepared identity mismatch: {table}/{phase}/K{k}")
                        cases.append({"table": table, "phase": phase, "prepared_k": k, "rows": rows})
                    cur.execute("DEALLOCATE PREPARE union_stmt")
            return cases
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
        finally:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    cases = capture(args.port)
    report = {"status": "captured", "case_count": len(cases), "cases": cases}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))["cases"]
        report["status"] = "pass" if before == cases else "fail"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(cases)} native union cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

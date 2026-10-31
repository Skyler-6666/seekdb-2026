"""Check FTS cache lookups, score/identity alignment, and partial result batches.

Capture on official and candidate seekdb instances using the same script. The
test keeps complete rows, scores and plans, and changes only a randomly named
database. Implicit-order pages validate every ID and score against a full scored
query; only selection inside a score tie cut by a page boundary is normalized.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from native_extended_regression import exact


def observe(cursor, records, name, sql, params=()):
    cursor.execute(sql, params)
    columns = [item[0] for item in cursor.description]
    rows = cursor.fetchall()
    cursor.execute("SHOW WARNINGS")
    records.append({"name": name, "sql": sql, "params": exact(params),
                    "columns": columns, "rows": exact(rows),
                    "warnings": exact(cursor.fetchall())})
    return list(rows)


def check_page(rows, reference, limit, offset):
    """Check ranking and payload/score alignment, retaining full identities."""
    expected_scores = sorted((row[2] for row in reference.values()), reverse=True)
    expected_scores = expected_scores[offset:offset + limit]
    if len(rows) != len(expected_scores) or len({row[0] for row in rows}) != len(rows):
        raise AssertionError("Incorrect or duplicate implicit-order page identities")
    for row, expected_score in zip(rows, expected_scores):
        wanted = reference.get(row[0])
        if wanted is None or row[:2] != wanted[:2]:
            raise AssertionError("Cached projection is paired with the wrong base row")
        if not math.isclose(row[2], wanted[2], rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Cached score differs from that document's full-query score")
        if not math.isclose(row[2], expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError("Implicit page is not the requested descending score slice")
    all_counts = Counter(row[2] for row in reference.values())
    groups = []
    for ident, payload, score in rows:
        if not groups or score != groups[-1][0]:
            groups.append([score, []])
        groups[-1][1].append([ident, payload])
    # A partial tie at either end may select different valid members. Full tie
    # groups must retain exactly the same document identities and payloads.
    return [[exact(score), sorted(items) if len(items) == all_counts[score]
             else {"partial_tie_count": len(items)}] for score, items in groups]


def capture(port):
    import pymysql
    database = "fts_cache_capacity_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    records, plans = [], []
    data = {}
    for ident in range(2113):
        body = "quartz " * (1 + ident % 13) if ident % 5 else ""
        body += "zephyr " * (1 + ident % 7) if ident % 7 else ""
        body += "filler " * (1 + ident % 19)
        data[ident] = (None if ident % 17 == 0 else ident * 3, body)
    match = "MATCH(body) AGAINST('quartz zephyr')"
    pages = ((0, 0), (1, 0), (7, 3), (20, 0), (63, 0), (64, 0),
             (65, 17), (1001, 997), (20, 2100), (7, 3000))
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(id BIGINT PRIMARY KEY,payload INT,body TEXT)")
            cur.executemany("INSERT INTO docs VALUES(%s,%s,%s)",
                            [(ident, payload, body) for ident, (payload, body) in data.items()])
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER beng")
            for phase in ("initial", "after_dml"):
                if phase == "after_dml":
                    cur.execute("UPDATE docs SET payload=-id,body='quartz updated' WHERE id BETWEEN 20 AND 40")
                    for ident in range(20, 41):
                        data[ident] = (-ident, "quartz updated")
                    cur.execute("DELETE FROM docs WHERE id BETWEEN 1980 AND 2010")
                    for ident in range(1980, 2011):
                        del data[ident]
                    cur.execute("INSERT INTO docs VALUES(3000,9000,'zephyr quartz'),(3001,NULL,'other text')")
                    data.update({3000: (9000, "zephyr quartz"), 3001: (None, "other text")})
                reference_rows = observe(cur, records, f"{phase}_score_reference",
                    f"SELECT id,payload,{match} AS score FROM docs WHERE {match} ORDER BY id")
                expected_ids = [ident for ident, (_, body) in sorted(data.items())
                                if "quartz" in body.split() or "zephyr" in body.split()]
                if [row[0] for row in reference_rows] != expected_ids:
                    raise AssertionError("Full scored query lost or invented a source document")
                reference = {row[0]: row for row in reference_rows}
                for row in reference_rows:
                    if row[1] != data[row[0]][0] or not math.isfinite(row[2]):
                        raise AssertionError("Invalid full-query payload or score")
                for batch in (7, 64, 256):
                    hint = f"/*+ OPT_PARAM('_rowsets_max_rows',{batch}) */"
                    scored = f"SELECT {hint} id,payload,{match} AS score FROM docs WHERE {match}"
                    ids = f"SELECT {hint} id,payload FROM docs WHERE {match}"
                    for limit, offset in pages:
                        suffix = f" LIMIT {limit} OFFSET {offset}"
                        prefix = f"{phase}_batch{batch}_limit{limit}_offset{offset}"
                        rows = observe(cur, records, prefix + "_scored_ids", scored + " ORDER BY id" + suffix)
                        if rows != reference_rows[offset:offset + limit]:
                            raise AssertionError("ID-ordered page changed score/row alignment")
                        rows = observe(cur, records, prefix + "_ids", ids + " ORDER BY id" + suffix)
                        if rows != [row[:2] for row in reference_rows[offset:offset + limit]]:
                            raise AssertionError("Document-only page changed base rows")
                        rows = observe(cur, records, prefix + "_implicit", scored + suffix)
                        records[-1]["comparison_rows"] = check_page(rows, reference, limit, offset)
                        sql = f"SELECT COUNT(*) AS n FROM ({ids}{suffix}) limited"
                        rows = observe(cur, records, prefix + "_count", sql)
                        if rows != [(min(limit, max(len(reference) - offset, 0)),)]:
                            raise AssertionError("Count-only page has incorrect cardinality")
                        if phase == "initial" and (limit, offset) == (20, 0):
                            for kind, plan_sql in (("count", sql), ("implicit", scored + suffix)):
                                cur.execute("EXPLAIN " + plan_sql)
                                plan = exact(cur.fetchall())
                                plan_text = json.dumps(plan, ensure_ascii=False).lower()
                                if "text retrieval" not in plan_text or "is_index_back=true" not in plan_text:
                                    raise AssertionError("Expected FTS index lookup was not selected")
                                plans.append({"name": f"batch{batch}_{kind}", "sql": plan_sql, "rows": plan})
                    ranked_reference = sorted(reference_rows, key=lambda row: (-row[2], row[0]))
                    for limit, offset in ((20, 3), (65, 17)):
                        sql = scored + f" ORDER BY score DESC,id LIMIT {limit} OFFSET {offset}"
                        rows = observe(cur, records, f"{phase}_batch{batch}_stable_rank_{limit}_{offset}", sql)
                        if rows != ranked_reference[offset:offset + limit]:
                            raise AssertionError("Stable ranked page changed document/score alignment")
                    # Residual-filter lookups retain the original large prefetch.
                    for condition, accepts in (("payload IS NULL", lambda p: p is None),
                            ("MOD(payload,97)=0", lambda p: p is not None and p % 97 == 0),
                            ("payload>999999", lambda p: p is not None and p > 999999)):
                        sql = scored + f" AND {condition} ORDER BY id LIMIT 7 OFFSET 3"
                        rows = observe(cur, records, f"{phase}_batch{batch}_{condition}", sql)
                        if rows != [row for row in reference_rows if accepts(row[1])][3:10]:
                            raise AssertionError("Filtered lookup page changed its identities or scores")
                statement = (f"SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ id,payload,{match} AS score "
                             f"FROM docs WHERE {match} ORDER BY id LIMIT ? OFFSET ?")
                cur.execute("PREPARE cache_page FROM %s", (statement,))
                try:
                    for number, (limit, offset) in enumerate(((1, 0), (65, 17), (7, 997), (0, 0), (1, 0))):
                        cur.execute("SET @n=%s,@o=%s", (limit, offset))
                        rows = observe(cur, records, f"{phase}_prepared_{number}", "EXECUTE cache_page USING @n,@o")
                        records[-1]["bindings"] = [limit, offset]
                        if rows != reference_rows[offset:offset + limit]:
                            raise AssertionError("Prepared lookup retained stale scores, rows, or pagination")
                finally:
                    cur.execute("DEALLOCATE PREPARE cache_page")
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
        finally:
            conn.close()
    return records, plans


def comparable(records):
    return [{key: value for key, value in record.items() if key not in ("rows", "comparison_rows")}
            | {"rows": record.get("comparison_rows", record["rows"])} for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    records, plans = capture(args.port)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "captured", "script_sha256": digest, "case_count": len(records),
              "records": records, "plans": plans}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if (before["script_sha256"] == digest
            and comparable(before["records"]) == comparable(records)) else "fail"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} native FTS cache lookup cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

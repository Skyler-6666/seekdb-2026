"""Capture or compare real seekdb results for the predicate-only DAAT path.

Run once against each freshly started build, using the same resource settings:
  python3 doc_only_regression.py --port 38680 --output before.json
  python3 doc_only_regression.py --port 38680 --output after.json --compare before.json

PyMySQL is required. No replacement engine or stored-answer fallback is used.
Only this script's randomly named database is created/dropped. Password, if any,
comes from FTS_TEST_PASSWORD and is never put in the report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import uuid


def query_cases():
    match = "MATCH(body) AGAINST('quartz zephyr')"
    score = "MATCH(body) AGAINST('quartz zephyr' IN NATURAL LANGUAGE MODE)"
    return [
        ("count", f"SELECT COUNT(*) FROM docs WHERE {match}"),
        ("ids", f"SELECT id FROM docs WHERE {match} ORDER BY id"),
        ("other_filter", f"SELECT id FROM docs WHERE {match} AND category=2 ORDER BY id"),
        ("aggregation", f"SELECT COUNT(*),SUM(id),MIN(id),MAX(id) FROM docs WHERE {match}"),
        ("group", f"SELECT category,COUNT(*),SUM(id) FROM docs WHERE {match} GROUP BY category ORDER BY category"),
        ("distinct", f"SELECT DISTINCT category FROM docs WHERE {match} ORDER BY category"),
        ("empty_search", "SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST('')"),
        ("missing_search", "SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST('nonexistentonly')"),
        ("common_term", "SELECT id FROM docs WHERE MATCH(body) AGAINST('commontoken') ORDER BY id"),
        ("score_projection", f"SELECT id,{score} AS score FROM docs WHERE {match} ORDER BY id"),
        ("score_aggregate", f"SELECT COUNT(*) AS n,SUM({score}) AS sum_score,MIN({score}) AS min_score,MAX({score}) AS max_score FROM docs WHERE {match}"),
        ("score_threshold", f"SELECT id FROM docs WHERE {match} AND {score}>0.5 ORDER BY id"),
        ("score_negative", f"SELECT id FROM docs WHERE {score}<=0 ORDER BY id"),
        ("score_group", f"SELECT FLOOR({score}) AS bucket,COUNT(*) AS n FROM docs WHERE {match} GROUP BY FLOOR({score}) ORDER BY 1"),
        ("score_having", f"SELECT id,{score} AS score FROM docs WHERE {match} HAVING score>0.5 ORDER BY id"),
        ("score_rank", f"SELECT id,{score} AS score FROM docs WHERE {match} ORDER BY {score} DESC,id LIMIT 17"),
        ("score_window", f"SELECT id,ROW_NUMBER() OVER(ORDER BY {score} DESC,id) AS n FROM docs WHERE {match} ORDER BY id"),
        ("implicit_rank", f"SELECT id,{score} AS score FROM docs WHERE {match}"),
        ("implicit_ids", f"SELECT id FROM docs WHERE {match}"),
        ("implicit_ids_limit", f"SELECT id FROM docs WHERE {match} LIMIT 17"),
        ("limit_count", f"SELECT COUNT(*) FROM (SELECT id FROM docs WHERE {match} LIMIT 17) q"),
        ("or_filter", f"SELECT id FROM docs WHERE {match} OR id=-1 ORDER BY id"),
        ("boolean_required", "SELECT id FROM docs WHERE MATCH(body) AGAINST('+quartz -zephyr' IN BOOLEAN MODE) ORDER BY id"),
        ("boolean_optional", "SELECT id FROM docs WHERE MATCH(body) AGAINST('quartz zephyr' IN BOOLEAN MODE) ORDER BY id"),
        ("boolean_negative_only", "SELECT id FROM docs WHERE MATCH(body) AGAINST('-quartz' IN BOOLEAN MODE) ORDER BY id"),
        ("multi_column", "SELECT COUNT(*) FROM docs WHERE MATCH(title,body) AGAINST('quartz zephyr')"),
        ("chinese", "SELECT id FROM docs WHERE MATCH(body_cn) AGAINST('数据库 索引') ORDER BY id"),
    ]


def normalize(value):
    from decimal import Decimal
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("Non-finite result from MATCH")
    return value


def equal(left, right):
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return (isinstance(left, (int, float)) and isinstance(right, (int, float))
                and math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10))
    return type(left) is type(right) and left == right


def score_groups(rows):
    """Preserve complete identity and descending score order, normalize ties only."""
    groups = []
    seen = set()
    for ident, score in rows:
        if ident in seen:
            raise AssertionError("Duplicate document in implicit score ordering")
        seen.add(ident)
        if groups and score > groups[-1][0]:
            raise AssertionError("Implicit MATCH ordering is not descending by score")
        if groups and score == groups[-1][0]:
            groups[-1][1].append(ident)
        else:
            groups.append([score, [ident]])
    for _, ids in groups:
        ids.sort()
    return groups


def validate_implicit_ids(rows, reference, limit=None):
    """Check unprojected ranking against independently retrieved true scores."""
    if reference is None:
        raise AssertionError("Missing score-producing reference query")
    ids = [row[0] for row in rows]
    if any(len(row) != 1 for row in rows) or len(ids) != len(set(ids)):
        raise AssertionError("Invalid or duplicate implicit-ranking document IDs")
    if not set(ids).issubset(reference):
        raise AssertionError("Implicit-ranking query returned a nonmatching document")
    count = len(reference) if limit is None else min(limit, len(reference))
    if len(ids) != count:
        raise AssertionError("Implicit-ranking query returned an incorrect row count")
    scored = [[ident, reference[ident]] for ident in ids]
    groups = score_groups(scored)
    expected_scores = sorted(reference.values(), reverse=True)[:count]
    if not equal([score for _, score in scored], expected_scores):
        raise AssertionError("Implicit-ranking IDs do not have the required top scores")
    return groups


def comparison_rows(case):
    rows = case["rows"]
    if case["name"] == "implicit_ids_limit" and rows:
        # Both captures independently verify every actual ID and its top-N rank.
        # Preserve actual IDs in the report; only the last tie group's choice of
        # IDs may differ when LIMIT cuts it. Earlier groups compare IDs exactly.
        return rows[:-1] + [[rows[-1][0], {"tie_count": len(rows[-1][1])}]]
    return rows


def capture(port):
    import pymysql
    database = "fts_doc_only_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    results = []
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(id BIGINT PRIMARY KEY, category INT, title TEXT, body TEXT, body_cn TEXT)")
            rows = []
            for i in range(512):
                text = "commontoken " + ("quartz " * (1 + i % 7) if i % 3 else "")
                text += "zephyr " * (1 + i % 5) if i % 4 else "differentword "
                text += "filler " * (1 + i % 11)
                rows.append((i, i % 5, "quartz title" if i % 2 else "another title", text,
                             "数据库 全文索引" if i % 2 else "其他 内容"))
            rows.extend([(-1, 0, "empty", "", ""), (-2, 0, None, None, None)])
            cur.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s,%s)", rows)
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER beng")
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_multi(title,body) WITH PARSER beng")
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_cn(body_cn) WITH PARSER ik")
            for phase in ("initial", "after_dml"):
                reference_scores = None
                if phase == "after_dml":
                    cur.execute("UPDATE docs SET body='quartz commontoken updated' WHERE id BETWEEN 20 AND 40")
                    cur.execute("DELETE FROM docs WHERE id BETWEEN 60 AND 80")
                    cur.execute("INSERT INTO docs VALUES(1000,2,'fresh','zephyr commontoken','数据库 索引')")
                for name, sql in query_cases():
                    cur.execute(sql)
                    rows = normalize(cur.fetchall())
                    if name == "score_projection":
                        reference_scores = dict(rows)
                        if len(reference_scores) != len(rows):
                            raise AssertionError("Duplicate document in score reference")
                    # Read all rows so a LIMIT boundary cannot select an arbitrary
                    # subset of a tie. Retain all IDs and validate order first.
                    if name == "implicit_rank":
                        rows = score_groups(rows)
                    elif name == "implicit_ids":
                        rows = validate_implicit_ids(rows, reference_scores)
                    elif name == "implicit_ids_limit":
                        rows = validate_implicit_ids(rows, reference_scores, limit=17)
                    results.append({"phase": phase, "name": name, "sql": sql, "rows": rows})
                cur.execute("PREPARE fts_predicate FROM 'SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST(?)'")
                for term in ("quartz", "zephyr", "nonexistentonly", "quartz"):
                    cur.execute("SET @fts_term=%s", (term,))
                    cur.execute("EXECUTE fts_predicate USING @fts_term")
                    results.append({"phase": phase, "name": "prepared_count", "term": term,
                                    "rows": normalize(cur.fetchall())})
                cur.execute("DEALLOCATE PREPARE fts_predicate")
            return results
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
    rows = capture(args.port)
    report = {"status": "captured", "cases": rows, "case_count": len(rows)}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))["cases"]
        failures = []
        if len(before) != len(rows):
            failures.append("case count differs")
        for a, b in zip(before, rows):
            if ({k: v for k, v in a.items() if k != "rows"}
                    != {k: v for k, v in b.items() if k != "rows"}
                    or not equal(comparison_rows(a), comparison_rows(b))):
                failures.append(f"{b['phase']}/{b['name']}")
        report.update(status="fail" if failures else "pass", failures=failures)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(rows)} native cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

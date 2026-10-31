"""Native regression for unobservable implicit MATCH ordering below COUNT(*).

Run against independently started baseline and candidate builds:
  python3 count_only_regression.py --port 38680 --output before.json
  python3 count_only_regression.py --port 38680 --output after.json \
      --compare before.json --require-no-synthetic-sort

Only a fresh random database is created/dropped. Result types, scores and warnings
are compared exactly. Plans are recorded separately because they should change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from native_extended_regression import exact, observe


def record_unsupported(records, name, sql, params, error):
    # Record only recognized SQL/parser capability failures. Connection errors,
    # resource failures and unexpected execution errors must abort the capture.
    # A baseline error and a candidate result (or vice versa) compare unequal.
    if not error.args or error.args[0] not in (1064, 1235, 1295):
        raise error
    records.append({"name": name, "sql": sql, "params": exact(params),
                    "unsupported": {"code": error.args[0], "message": str(error.args[1])}})


def observe_optional(cursor, records, name, sql, params=()):
    import pymysql
    try:
        observe(cursor, records, name, sql, params)
    except pymysql.MySQLError as error:
        record_unsupported(records, name, sql, params, error)
        return False
    return True


def capture_prepared(cursor, records, corpus):
    import pymysql
    statements = [
        ("count_term", "SELECT COUNT(*) AS n FROM (SELECT id FROM docs "
         "WHERE MATCH(body) AGAINST(?) LIMIT 20 OFFSET 3) q",
         [(query, 20, 3) for query in ("quartz", "missingterm", "", "quartz zephyr", "quartz")]),
        ("count_limits", "SELECT COUNT(*) AS n FROM (SELECT id FROM docs "
         "WHERE MATCH(body) AGAINST(?) LIMIT ? OFFSET ?) q",
         [("quartz", 20, 0), ("quartz", 0, 0), ("quartz zephyr", 1, 3),
          ("missingterm", 200, 0), ("", 20, 0), ("quartz", 200, 500),
          ("quartz", 200, 3), ("quartz", 20, 0)]),
    ]
    for name, statement, bindings in statements:
        prepare_sql = f"PREPARE {name} FROM %s"
        try:
            cursor.execute(prepare_sql, (statement,))
        except pymysql.MySQLError as error:
            record_unsupported(records, f"prepare_{name}", prepare_sql, (statement,), error)
            continue
        cursor.execute("SHOW WARNINGS")
        records.append({"name": f"prepare_{name}", "sql": statement,
                        "prepared": True, "warnings": exact(cursor.fetchall())})
        try:
            for index, (query, limit, offset) in enumerate(bindings):
                cursor.execute("SET @fts_term=%s,@fts_limit=%s,@fts_offset=%s", (query, limit, offset))
                variables = "@fts_term" if name == "count_term" else "@fts_term,@fts_limit,@fts_offset"
                succeeded = observe_optional(cursor, records, f"{name}_{index}",
                                             f"EXECUTE {name} USING {variables}")
                records[-1]["bindings"] = [query, limit, offset]
                if succeeded:
                    matches = sum(bool(set(query.split()).intersection((body or "").split()))
                                  for _, _, body in corpus)
                    if records[-1]["rows"] != [[min(limit, max(matches - offset, 0))]]:
                        raise AssertionError(f"Prepared cardinality mismatch: {name}/{index}")
        finally:
            cursor.execute(f"DEALLOCATE PREPARE {name}")


def capture(port, require_no_sort):
    import pymysql

    database = "fts_count_limit_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    records, plans = [], []
    corpus = [(i, None if i % 3 == 0 else i,
               " ".join((["quartz"] * i if i % 7 else ["elsewhere"] * i)
                        + ["zephyr" if i % 3 else "orchid"] * (192 - i)))
              for i in range(1, 98)]
    corpus += [(98, None, ""), (99, None, None)]
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(id INT PRIMARY KEY, nullable_id INT, body TEXT)")
            cur.executemany("INSERT INTO docs VALUES(%s,%s,%s)", corpus)
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
            for query in ("quartz", "quartz zephyr", "missingterm", ""):
                matches = sum(bool(set(query.split()).intersection((body or "").split()))
                              for _, _, body in corpus)
                for limit in (0, 1, 20, 200):
                    for offset in (0, 3, 500):
                        sql = ("SELECT COUNT(*) AS n FROM (SELECT id FROM docs "
                               f"WHERE MATCH(body) AGAINST(%s) LIMIT {limit} OFFSET {offset}) q")
                        observe(cur, records, f"count_{query}_{limit}_{offset}", sql, (query,))
                        if records[-1]["rows"] != [[min(limit, max(matches - offset, 0))]]:
                            raise AssertionError(f"Cardinality mismatch: {query}/{limit}/{offset}")
            match = "MATCH(body) AGAINST('quartz')"
            inner = f"SELECT id,nullable_id FROM docs WHERE {match}"
            negative_cases = [
                ("direct_implicit_ids", inner + " LIMIT 20"),
                ("score_projection", f"SELECT id,{match} AS score FROM docs WHERE {match} LIMIT 20"),
                ("count_nullable", f"SELECT COUNT(nullable_id) FROM ({inner} LIMIT 20) q"),
                ("count_distinct", f"SELECT COUNT(DISTINCT nullable_id) FROM ({inner} LIMIT 20) q"),
                ("outer_filter", f"SELECT COUNT(*) FROM ({inner} LIMIT 20) q WHERE id<30"),
                ("outer_sum", f"SELECT COUNT(*),SUM(id) FROM ({inner} LIMIT 20) q"),
                ("explicit_id_order", f"SELECT COUNT(*) FROM ({inner} ORDER BY id LIMIT 20) q"),
                ("explicit_score_order", f"SELECT COUNT(*) FROM ({inner} ORDER BY {match} DESC LIMIT 20) q"),
                ("child_filter", f"SELECT COUNT(*) FROM ({inner} AND id<30 LIMIT 20) q"),
                ("score_threshold", f"SELECT COUNT(*) FROM (SELECT id FROM docs WHERE {match}>0.5 LIMIT 20) q"),
                ("child_distinct", f"SELECT COUNT(*) FROM (SELECT DISTINCT nullable_id FROM docs WHERE {match} LIMIT 20) q"),
                ("boolean", "SELECT COUNT(*) FROM (SELECT id FROM docs WHERE "
                            "MATCH(body) AGAINST('+quartz -zephyr' IN BOOLEAN MODE) LIMIT 20) q"),
                ("sole_cte_count", f"WITH q AS ({inner} LIMIT 20) SELECT COUNT(*) AS n FROM q"),
                ("cte_observable_sum", f"WITH q AS ({inner} LIMIT 20) SELECT COUNT(*) AS n,SUM(id) AS id_sum FROM q"),
                ("shared_cte", f"WITH q AS ({inner} LIMIT 20) "
                               "SELECT COUNT(*) AS n,SUM(a.id) AS a_sum,SUM(b.id) AS b_sum FROM q a CROSS JOIN q b"),
                ("nested_count", f"SELECT (SELECT COUNT(*) FROM ({inner} LIMIT 20) q) AS n"),
                ("projection_side_effect", "SELECT COUNT(*) FROM (SELECT LAST_INSERT_ID(id) AS value "
                                           f"FROM docs WHERE {match} LIMIT 20) q"),
            ]
            for name, sql in negative_cases:
                if name == "projection_side_effect":
                    observe(cur, records, "last_insert_id_reset", "SELECT LAST_INSERT_ID(0) AS saved_id")
                observe_optional(cur, records, name, sql)
                if name == "projection_side_effect":
                    observe(cur, records, "last_insert_id_after", "SELECT LAST_INSERT_ID() AS saved_id")
            capture_prepared(cur, records, corpus)
            for name, sql in [
                ("count_only", f"SELECT COUNT(*) FROM ({inner} LIMIT 20) q"),
                ("observable_ids", inner + " LIMIT 20"),
                ("nullable_count", f"SELECT COUNT(nullable_id) FROM ({inner} LIMIT 20) q"),
                ("explicit_order", f"SELECT COUNT(*) FROM ({inner} ORDER BY {match} DESC LIMIT 20) q"),
            ]:
                cur.execute("EXPLAIN " + sql)
                rows = exact(cur.fetchall())
                plans.append({"name": name, "sql": sql, "rows": rows})
                if name == "count_only" and require_no_sort:
                    if "SORT" in json.dumps(rows, ensure_ascii=False).upper():
                        raise AssertionError("Count-only plan still contains a synthetic sort")
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
        finally:
            conn.close()
    return records, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--require-no-synthetic-sort", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    records, plans = capture(args.port, args.require_no_synthetic_sort)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "captured", "script_sha256": digest, "case_count": len(records),
              "records": records, "plans": plans}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if (before["script_sha256"] == digest
                                         and before["records"] == records) else "fail"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} native count-limit cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

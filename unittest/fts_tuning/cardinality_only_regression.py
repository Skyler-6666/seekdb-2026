"""Exact native regression for the strict direct-COUNT FTS output shortcut.

Run this unchanged against independently started official and candidate builds.
Pure counts are also checked against a Python token-set corpus and the original
scored MATCH > 0 path. Capability errors in optional SQL shapes are recorded,
including their codes/messages; baseline errors never silently become passes.
Only a new random database is created and dropped. No benchmark is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from native_extended_regression import exact, observe
from count_only_regression import observe_optional, record_unsupported


TERMS = ["quartz", "zephyr", "orchid", "marigold", "saffron", "topaz", "onyx", "cobalt", "willow"]
BOUNDARIES = (0, 1, 7, 8, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1023, 1024, 1025)
MARKER = "enabled fulltext cardinality-only output"


def make_corpus():
    data = {}
    for ordinal in range(2057):
        words = [term for index, term in enumerate(TERMS)
                 if (ordinal + index) % (index % 4 + 2)]
        if ordinal % 29 == 0:
            words = ["elsewhere"]
        elif ordinal % 31 == 0:
            words = TERMS[:]
        words += words[:1] * (ordinal % 7)
        words += ["edge" + chr(97 + index) for index, count in enumerate(BOUNDARIES)
                  if ordinal < count]
        body = " ".join(words)
        if ordinal == 2055:
            body = None
        elif ordinal == 2056:
            body = ""
        data[ordinal - 1028] = (ordinal % 7, None if ordinal % 5 == 0 else ordinal % 11, body)
    return data


def matching(data, query):
    words = set(query.split())
    return {ident: values for ident, values in data.items()
            if words.intersection((values[-1] or "").split())}


def checked_count(cur, records, name, sql, query, expected, optional=False):
    if optional:
        if not observe_optional(cur, records, name, sql, (query,)):
            return
    else:
        observe(cur, records, name, sql, (query,))
    record = records[-1]
    if record["rows"] != [[expected]]:
        raise AssertionError(f"Python cardinality mismatch: {name}: {record['rows']} != {expected}")
    record["python_expected_count"] = expected
    scored_sql = sql.replace("AGAINST(%s)", "AGAINST(%s)>0")
    if scored_sql == sql:
        raise AssertionError("The independent scored reference was not constructed")
    reference = []
    observe(cur, reference, name + "_scored_reference", scored_sql, (query,))
    record["scored_reference"] = reference[0]
    if reference[0]["rows"] != record["rows"]:
        raise AssertionError(f"Original scored MATCH cardinality differs: {name}")


def capture_positive(cur, records, plans, data):
    for k in range(1, 10):
        query = " ".join(TERMS[:k])
        expected = len(matching(data, query))
        for batch in (1, 7, 16, 64, 256):
            sql = (f"SELECT /*+ OPT_PARAM('_rowsets_max_rows',{batch}) */ COUNT(*) AS n "
                   "FROM docs WHERE MATCH(body) AGAINST(%s)")
            checked_count(cur, records, f"initial_k{k}_batch{batch}", sql, query, expected)
            if batch == 64 and k in (1, 2, 8, 9):
                cur.execute("EXPLAIN " + sql, (query,))
                plan = exact(cur.fetchall())
                if "text retrieval" not in json.dumps(plan).lower():
                    raise AssertionError(f"Expected an FTS plan for K={k}")
                plans.append({"name": f"direct_k{k}", "sql": sql, "params": [query], "rows": plan})
    for index, count in enumerate(BOUNDARIES):
        query = "edge" + chr(97 + index)
        expected = len(matching(data, query))
        if expected != count:
            raise AssertionError("The independent batch-boundary corpus is malformed")
        for batch in (16, 64, 256):
            sql = (f"SELECT /*+ OPT_PARAM('_rowsets_max_rows',{batch}) */ COUNT(*) AS n "
                   "FROM docs WHERE MATCH(body) AGAINST(%s)")
            checked_count(cur, records, f"boundary_count{count}_batch{batch}", sql, query, expected)
    for index, query in enumerate(("", "missingcardinalityterm", "quartz quartz zephyr", "quartz missingcardinalityterm")):
        checked_count(cur, records, f"empty_or_duplicate_{index}",
                      "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(%s)",
                      query, len(matching(data, query)))
    # Exact score-bearing rows must survive when a consumer can observe them.
    for k in (1, 2, 8, 9):
        query = " ".join(TERMS[:k])
        observe(cur, records, f"scored_all_ids_k{k}",
                "SELECT id,MATCH(body) AGAINST(%s) AS score FROM docs "
                "WHERE MATCH(body) AGAINST(%s) ORDER BY id", (query, query))
        if [row[0] for row in records[-1]["rows"]] != sorted(matching(data, query)):
            raise AssertionError(f"Score-bearing source IDs differ for K={k}")


def capture_fallbacks(cur, records, data):
    query = "quartz zephyr"
    match = "MATCH(body) AGAINST('quartz zephyr')"
    where = " FROM docs WHERE " + match
    cases = [
        ("count_nullable", "SELECT COUNT(nullable_id) AS n" + where),
        ("count_one", "SELECT COUNT(1) AS n" + where),
        ("count_nonnull_pk", "SELECT COUNT(id) AS n" + where),
        ("count_null_constant", "SELECT COUNT(NULL) AS n" + where),
        ("count_distinct", "SELECT COUNT(DISTINCT nullable_id) AS n" + where),
        ("multiple_aggregates", "SELECT COUNT(*),SUM(id),MIN(id),MAX(id),AVG(nullable_id)" + where),
        ("score_aggregate", "SELECT COUNT(*),SUM(" + match + "),MAX(" + match + ")" + where),
        ("score_projection", "SELECT id," + match + " AS score" + where + " ORDER BY id LIMIT 67"),
        ("score_ranking", "SELECT id," + match + " AS score" + where + " ORDER BY score DESC,id LIMIT 67 OFFSET 13"),
        ("score_threshold_zero", "SELECT COUNT(*) AS n FROM docs WHERE " + match + ">0"),
        ("score_threshold_half", "SELECT COUNT(*) AS n FROM docs WHERE " + match + ">0.5"),
        ("score_threshold_huge", "SELECT COUNT(*) AS n FROM docs WHERE " + match + ">1000000"),
        ("and_payload", "SELECT COUNT(*) AS n" + where + " AND grp=2 AND nullable_id IS NOT NULL"),
        ("and_pk", "SELECT COUNT(*) AS n" + where + " AND id<0"),
        ("or_payload", "SELECT COUNT(*) AS n FROM docs WHERE " + match + " OR grp=2"),
        ("or_pk", "SELECT COUNT(*) AS n FROM docs WHERE " + match + " OR id=-1028"),
        ("and_match", "SELECT COUNT(*) AS n" + where + " AND MATCH(body) AGAINST('orchid')"),
        ("or_match", "SELECT COUNT(*) AS n FROM docs WHERE " + match + " OR MATCH(body) AGAINST('orchid')"),
        ("group_having", "SELECT grp,COUNT(*) AS n,SUM(nullable_id)" + where + " GROUP BY grp HAVING COUNT(*)>17 ORDER BY grp"),
        ("group_nullable", "SELECT nullable_id,COUNT(*) AS n" + where + " GROUP BY nullable_id ORDER BY nullable_id"),
        ("scalar_having_true", "SELECT COUNT(*) AS n" + where + " HAVING COUNT(*)>0"),
        ("scalar_having_false", "SELECT COUNT(*) AS n" + where + " HAVING COUNT(*)>1000000"),
        ("distinct_count", "SELECT DISTINCT COUNT(*) AS n" + where),
        ("distinct_values", "SELECT DISTINCT nullable_id" + where + " ORDER BY nullable_id"),
        ("aggregate_order", "SELECT COUNT(*) AS n" + where + " ORDER BY n"),
        ("limit_zero", "SELECT COUNT(*) AS n" + where + " LIMIT 0"),
        ("limit_one", "SELECT COUNT(*) AS n" + where + " LIMIT 1"),
        ("limit_offset", "SELECT COUNT(*) AS n" + where + " LIMIT 1 OFFSET 1"),
        ("derived_plain", "SELECT COUNT(*) AS n FROM (SELECT id" + where + ") q"),
        ("derived_page", "SELECT COUNT(*) AS n FROM (SELECT id" + where + " ORDER BY id LIMIT 65 OFFSET 17) q"),
        ("derived_empty", "SELECT COUNT(*) AS n FROM (SELECT id" + where + " LIMIT 0) q"),
        ("derived_group", "SELECT COUNT(*),SUM(n) FROM (SELECT grp,COUNT(*) AS n" + where + " GROUP BY grp) q"),
        ("derived_observable", "SELECT COUNT(*),SUM(id),SUM(nullable_id) FROM (SELECT id,nullable_id" + where + " ORDER BY id LIMIT 65) q"),
        ("join_count", "SELECT COUNT(*) AS n FROM docs d JOIN tags t ON d.grp=t.grp "
                       "WHERE MATCH(d.body) AGAINST('quartz zephyr')"),
        ("left_join_count", "SELECT COUNT(*),COUNT(t.tag) FROM docs d LEFT JOIN tags t ON d.grp=t.grp "
                            "WHERE MATCH(d.body) AGAINST('quartz zephyr')"),
        ("join_filter", "SELECT COUNT(*) AS n FROM docs d JOIN tags t ON d.grp=t.grp "
                        "WHERE MATCH(d.body) AGAINST('quartz zephyr') AND t.tag=1"),
        ("cross_join", "SELECT COUNT(*) AS n FROM docs CROSS JOIN (SELECT 1 AS x UNION ALL SELECT 2) q "
                       "WHERE MATCH(body) AGAINST('quartz zephyr')"),
        ("nested_scalar", "SELECT (SELECT COUNT(*)" + where + ") AS n,17 AS marker"),
        ("cte_plain", "WITH q AS (SELECT id" + where + ") SELECT COUNT(*) AS n FROM q"),
        ("cte_observable", "WITH q AS (SELECT id,nullable_id" + where + ") SELECT COUNT(*),SUM(id),SUM(nullable_id) FROM q"),
        ("union_counts", "SELECT COUNT(*) AS n" + where + " UNION ALL SELECT COUNT(*) FROM docs WHERE id<0"),
        ("boolean_count", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST('+quartz -zephyr' IN BOOLEAN MODE)"),
        ("null_query", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(NULL)"),
        ("aliased_count", "SELECT COUNT(*) AS n FROM docs d WHERE MATCH(d.body) AGAINST('quartz zephyr')"),
    ]
    selected = matching(data, query)
    expected = {
        "count_nullable": sum(values[1] is not None for values in selected.values()),
        "count_one": len(selected), "count_nonnull_pk": len(selected), "count_null_constant": 0,
        "count_distinct": len({values[1] for values in selected.values() if values[1] is not None}),
        "score_threshold_zero": len(selected), "score_threshold_huge": 0,
        "and_payload": sum(values[0] == 2 and values[1] is not None for values in selected.values()),
        "and_pk": sum(ident < 0 for ident in selected),
        "or_payload": sum(ident in selected or values[0] == 2 for ident, values in data.items()),
        "or_pk": len(set(selected) | {-1028}),
        "and_match": len(set(selected) & set(matching(data, "orchid"))),
        "or_match": len(set(selected) | set(matching(data, "orchid"))),
        "distinct_count": len(selected), "scalar_having_true": len(selected),
        "aggregate_order": len(selected), "limit_one": len(selected),
        "derived_plain": len(selected), "derived_page": min(65, max(len(selected) - 17, 0)),
        "derived_empty": 0, "join_count": sum(2 for values in selected.values() if values[0] < 6),
        "join_filter": sum(values[0] < 6 for values in selected.values()),
        "cross_join": len(selected) * 2, "cte_plain": len(selected), "aliased_count": len(selected),
        "boolean_count": len(set(matching(data, "quartz")) - set(matching(data, "zephyr"))),
        "null_query": 0,
    }
    for name, sql in cases:
        if observe_optional(cur, records, "fallback_" + name, sql):
            if name in expected and records[-1]["rows"] != [[expected[name]]]:
                raise AssertionError(f"Fallback Python expectation differs: {name}")
            if name in ("limit_zero", "limit_offset", "scalar_having_false") and records[-1]["rows"] != []:
                raise AssertionError(f"Outer aggregate row suppression differs: {name}")


def prepare(cur, records, name, sql):
    import pymysql
    try:
        cur.execute(f"PREPARE {name} FROM %s", (sql,))
    except pymysql.MySQLError as error:
        record_unsupported(records, "prepare_" + name, f"PREPARE {name} FROM %s", (sql,), error)
        return False
    cur.execute("SHOW WARNINGS")
    records.append({"name": "prepare_" + name, "sql": sql, "prepared": True,
                    "warnings": exact(cur.fetchall())})
    return True


def capture_prepared(cur, records, data):
    statements = [
        ("card_direct", "SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ COUNT(*) AS n "
         "FROM docs WHERE MATCH(body) AGAINST(?)", "@q"),
        ("card_nullable", "SELECT COUNT(nullable_id) AS n FROM docs "
         "WHERE MATCH(body) AGAINST(?) AND grp=?", "@q,@g"),
        ("card_threshold", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(?)>?", "@q,@s"),
        ("card_page", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(?) LIMIT ? OFFSET ?", "@q,@n,@o"),
    ]
    queries = [" ".join(TERMS[:k]) for k in range(1, 10)]
    queries += ["missingcardinalityterm", "", "quartz quartz zephyr", "quartz"]
    for name, sql, variables in statements:
        if not prepare(cur, records, name, sql):
            continue
        try:
            for index, query in enumerate(queries):
                group, threshold = index % 7, (0, 0.5, 1000000)[index % 3]
                limit, offset = (0, 1, 7)[index % 3], (0, 1)[index % 2]
                cur.execute("SET @q=%s,@g=%s,@s=%s,@n=%s,@o=%s", (query, group, threshold, limit, offset))
                succeeded = observe_optional(cur, records, f"{name}_{index}", f"EXECUTE {name} USING {variables}")
                records[-1]["bindings"] = [query, group, threshold, limit, offset]
                if not succeeded:
                    continue
                selected = matching(data, query)
                if name == "card_direct":
                    expected = [[len(selected)]]
                elif name == "card_nullable":
                    expected = [[sum(values[0] == group and values[1] is not None for values in selected.values())]]
                elif name == "card_page":
                    expected = [[len(selected)]] if limit and not offset else []
                else:
                    reference = []
                    observe(cur, reference, name + "_reference", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(%s)>%s", (query, threshold))
                    records[-1]["scored_reference"] = reference[0]
                    expected = reference[0]["rows"]
                if records[-1]["rows"] != expected:
                    raise AssertionError(f"Prepared statement retained stale cardinality/state: {name}/{index}")
        finally:
            cur.execute(f"DEALLOCATE PREPARE {name}")


def capture_transactions(cur, records, data):
    direct = "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(%s)"
    prepared = prepare(cur, records, "card_transaction", direct.replace("%s", "?"))

    def snapshot(phase, current, optional=False):
        for index, query in enumerate(("quartz", "quartz zephyr", " ".join(TERMS[:8]), " ".join(TERMS), "edgep", "missingcardinalityterm")):
            checked_count(cur, records, f"{phase}_{index}", direct, query, len(matching(current, query)), optional)
        if prepared:
            query = "quartz zephyr"
            cur.execute("SET @txnq=%s", (query,))
            if observe_optional(cur, records, phase + "_prepared", "EXECUTE card_transaction USING @txnq"):
                records[-1]["bindings"] = [query]
                if records[-1]["rows"] != [[len(matching(current, query))]]:
                    raise AssertionError(f"Prepared transaction visibility mismatch: {phase}")

    def mutate(current, base):
        changed = dict(current)
        cur.execute("UPDATE docs SET body='willow zephyr',nullable_id=NULL WHERE id BETWEEN -100 AND -80")
        for ident, (group, nullable, body) in list(changed.items()):
            if -100 <= ident <= -80:
                changed[ident] = (group, None, "willow zephyr")
        cur.execute("DELETE FROM docs WHERE id BETWEEN 80 AND 100")
        for ident in range(80, 101):
            changed.pop(ident, None)
        additions = [(base + index, index % 7, None if index % 2 else index, "quartz orchid edgep") for index in range(17)]
        cur.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s)", additions)
        changed.update({ident: (group, nullable, body) for ident, group, nullable, body in additions})
        return changed

    try:
        snapshot("before_transaction", data)
        cur.execute("START TRANSACTION")
        rolled_back = mutate(data, 4000)
        snapshot("inside_rollback", rolled_back, optional=True)
        cur.execute("ROLLBACK")
        snapshot("after_rollback", data)
        cur.execute("START TRANSACTION")
        committed = mutate(data, 5000)
        snapshot("inside_commit", committed, optional=True)
        cur.execute("COMMIT")
        snapshot("after_commit", committed)
        return committed
    finally:
        cur.execute("ROLLBACK")
        if prepared:
            cur.execute("DEALLOCATE PREPARE card_transaction")


def capture_binary(cur, records):
    keys = [b"", b"\x00", b"\x00\x00", b"\x00 ", b" ", b"  ", b"a", b"a ", b"a  ",
            b"a\x00", b"a\xff", b"\x7f", b"\x80", b"\xff", b"\xff\x00", b"\xff ", b"\xff\xff"]
    keys += [b"bin" + index.to_bytes(2, "big") + (b" \x00" if index % 2 else b"\xff ") for index in range(1025)]
    if len(keys) != len(set(keys)):
        raise AssertionError("Duplicate binary source keys")
    data = {key: (None if index % 5 == 0 else index,
                  " ".join(term for k, term in enumerate(TERMS) if (index + k) % (k % 4 + 2)))
            for index, key in enumerate(keys)}
    cur.execute("CREATE TABLE binary_docs(key1 VARBINARY(16) PRIMARY KEY,payload INT,body TEXT)")
    items = list(data.items())
    for start in range(0, len(items), 256):
        chunk = items[start:start + 256]
        params = [value for key, (payload, body) in chunk for value in (key.hex(), payload, body)]
        affected = cur.execute("INSERT INTO binary_docs VALUES " + ",".join(["(UNHEX(%s),%s,%s)"] * len(chunk)), params)
        cur.execute("SHOW WARNINGS")
        if affected != len(chunk) or cur.fetchall():
            raise AssertionError("Binary-key insertion changed rows or produced warnings")
    cur.execute("ALTER TABLE binary_docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
    for phase in ("initial", "dml"):
        if phase == "dml":
            cur.execute("DELETE FROM binary_docs WHERE key1=UNHEX(%s)", (b"a ".hex(),))
            del data[b"a "]
            cur.execute("UPDATE binary_docs SET body='quartz zephyr',payload=NULL WHERE key1=UNHEX(%s)", (b"\xff\x00".hex(),))
            data[b"\xff\x00"] = (None, "quartz zephyr")
        for k in range(1, 10):
            query = " ".join(TERMS[:k])
            selected = matching(data, query)
            for batch in (7, 64):
                sql = (f"SELECT /*+ OPT_PARAM('_rowsets_max_rows',{batch}) */ COUNT(*) AS n "
                       "FROM binary_docs WHERE MATCH(body) AGAINST(%s)")
                checked_count(cur, records, f"binary_{phase}_k{k}_batch{batch}", sql, query, len(selected))
            observe(cur, records, f"binary_{phase}_k{k}_observable",
                    "SELECT HEX(key1),payload,MATCH(body) AGAINST(%s) AS score FROM binary_docs "
                    "WHERE MATCH(body) AGAINST(%s) ORDER BY key1", (query, query))
            if [row[:2] for row in records[-1]["rows"]] != [[key.hex().upper(), values[0]] for key, values in sorted(selected.items())]:
                raise AssertionError("Binary ID/payload projection was removed or reordered")
            observe(cur, records, f"binary_{phase}_k{k}_nullable",
                    "SELECT COUNT(payload) AS n FROM binary_docs WHERE MATCH(body) AGAINST(%s)", (query,))
            if records[-1]["rows"] != [[sum(values[0] is not None for values in selected.values())]]:
                raise AssertionError("Binary nullable payload count differs")


def capture(port):
    import pymysql
    database = "fts_cardinality_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180, write_timeout=180)
    records, plans = [], []
    created = False
    data = make_corpus()
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            created = True
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(id BIGINT PRIMARY KEY,grp INT NOT NULL,nullable_id INT,body TEXT)")
            cur.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s)", [(ident, *values) for ident, values in data.items()])
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
            cur.execute("CREATE TABLE tags(grp INT,tag INT,PRIMARY KEY(grp,tag))")
            cur.executemany("INSERT INTO tags VALUES(%s,%s)", [(group, tag) for group in range(6) for tag in range(2)])
            capture_positive(cur, records, plans, data)
            capture_fallbacks(cur, records, data)
            capture_prepared(cur, records, data)
            committed = capture_transactions(cur, records, data)
            for k in range(1, 10):
                query = " ".join(TERMS[:k])
                checked_count(cur, records, f"after_commit_k{k}", "SELECT COUNT(*) AS n FROM docs WHERE MATCH(body) AGAINST(%s)", query, len(matching(committed, query)))
            cur.execute("CREATE TABLE empty_docs(id BIGINT PRIMARY KEY,body TEXT)")
            cur.execute("ALTER TABLE empty_docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
            for index, query in enumerate(("", "quartz", " ".join(TERMS))):
                checked_count(cur, records, f"empty_table_{index}", "SELECT COUNT(*) AS n FROM empty_docs WHERE MATCH(body) AGAINST(%s)", query, 0)
            capture_binary(cur, records)
    finally:
        try:
            conn.rollback()
            if created:
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE `{database}`")
        finally:
            conn.close()
    return records, plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--require-cardinality", action="store_true",
                        help="Require a new codegen marker; runtime correctness is tested independently")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    if args.require_cardinality and not args.server_log:
        parser.error("--require-cardinality needs --server-log")
    offset = args.server_log.stat().st_size if args.server_log else 0
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("native_extended_regression.py", "count_only_regression.py")}
    records, plans = capture(args.port)
    unsupported = [record["name"] for record in records if "unsupported" in record]
    report = {"status": "captured", "script_sha256": digest, "helper_sha256": helper_hashes,
              "case_count": len(records), "unsupported_count": len(unsupported),
              "unsupported_cases": unsupported, "records": records, "plans": plans}
    if args.server_log:
        evidence = []
        with args.server_log.open("rb") as handle:
            handle.seek(offset)
            for raw in handle:
                line = raw.decode("utf-8", errors="replace")
                if MARKER in line:
                    evidence.append({key: int(value) for key, value in re.findall(r"\b(output_count)=(\d+)", line)})
        report["cardinality_codegen_evidence"] = evidence
        report["cardinality_codegen_observed"] = any(item.get("output_count") == 0 for item in evidence)
        report["activation_scope"] = "Codegen eligibility only; no claim of runtime activation for every K or fallback SQL"
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if (before["script_sha256"] == digest and before["helper_sha256"] == helper_hashes and before["records"] == records) else "fail"
    if args.require_cardinality and not report.get("cardinality_codegen_observed", False):
        report["status"] = "fail"
        report["failure"] = "No new cardinality-only codegen marker with output_count=0; activation is unproven"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} cardinality cases; {len(unsupported)} explicitly recorded capability errors")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

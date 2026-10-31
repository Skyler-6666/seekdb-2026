"""Native ordinary-index lookups with partial output batches and residual filters.

Run against separately started baseline and candidate instances. Every ordered
identity and payload is checked against a source corpus; plans and complete rows
are retained. The test creates/drops only its own randomly named database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from native_extended_regression import exact, observe


def capture(port):
    import pymysql
    database = "lookup_capacity_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    records, plans = [], []
    data = {i: (i % 3, None if i % 17 == 0 else i) for i in range(2113)}
    filters = [
        ("unfiltered", "1=1", lambda p: True),
        ("all", "payload IS NOT NULL", lambda p: p is not None),
        ("late", "payload>=1900", lambda p: p is not None and p >= 1900),
        ("sparse", "MOD(payload,47)=0", lambda p: p is not None and p % 47 == 0),
        ("none", "payload<0", lambda p: p is not None and p < 0),
        ("nulls", "payload IS NULL", lambda p: p is None),
    ]
    variants = [(0, 0), (1, 0), (7, 3), (63, 0), (64, 0), (65, 17), (1000, 0), (7, 3000)]
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}`")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(id INT PRIMARY KEY,group_id INT NOT NULL,payload INT,"
                        "INDEX ix_group(group_id,id))")
            cur.executemany("INSERT INTO docs VALUES(%s,%s,%s)",
                            [(i, group, value) for i, (group, value) in data.items()])
            for phase in ("initial", "after_dml"):
                if phase == "after_dml":
                    cur.execute("UPDATE docs SET group_id=1,payload=2048 WHERE id BETWEEN 20 AND 40")
                    for i in range(20, 41):
                        data[i] = (1, 2048)
                    cur.execute("DELETE FROM docs WHERE id BETWEEN 1980 AND 2010")
                    for i in range(1980, 2011):
                        del data[i]
                    cur.execute("INSERT INTO docs VALUES(3000,1,2048),(3001,1,NULL)")
                    data.update({3000: (1, 2048), 3001: (1, None)})
                for name, condition, accepts in filters:
                    expected = [[i, value] for i, (group, value) in sorted(data.items())
                                if group == 1 and accepts(value)]
                    base = ("SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ id,payload "
                            f"FROM docs FORCE INDEX(ix_group) WHERE group_id=1 AND {condition}")
                    for limit, offset in variants:
                        sql = base + f" ORDER BY id LIMIT {limit} OFFSET {offset}"
                        observe(cur, records, f"{phase}_{name}_{limit}_{offset}", sql)
                        if records[-1]["rows"] != expected[offset:offset + limit]:
                            raise AssertionError(f"Lookup identity mismatch: {phase}/{name}/{limit}/{offset}")
                    # COUNT observes only cardinality; without ORDER BY these
                    # also test demand propagation through a derived table.
                    for limit, offset in ((1, 0), (7, 3), (65, 17)):
                        sql = ("SELECT COUNT(*) AS n FROM (" + base
                               + f" LIMIT {limit} OFFSET {offset}) limited")
                        observe(cur, records, f"{phase}_{name}_count_{limit}_{offset}", sql)
                        if records[-1]["rows"] != [[min(limit, max(len(expected) - offset, 0))]]:
                            raise AssertionError(f"Lookup cardinality mismatch: {phase}/{name}/{limit}/{offset}")
                    if phase == "initial" and name in ("unfiltered", "all", "late"):
                        sql = base + " ORDER BY id LIMIT 7 OFFSET 3"
                        cur.execute("EXPLAIN " + sql)
                        rows = exact(cur.fetchall())
                        text = json.dumps(rows, ensure_ascii=False).lower()
                        if "ix_group" not in text or "is_index_back=true" not in text:
                            raise AssertionError("Forced secondary index did not retain the expected base-table lookup")
                        plans.append({"name": name, "sql": sql, "rows": rows})
                statement = ("SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ id,payload "
                             "FROM docs FORCE INDEX(ix_group) WHERE group_id=? AND payload>=? "
                             "ORDER BY id LIMIT ? OFFSET ?")
                cur.execute("PREPARE lookup_stmt FROM %s", (statement,))
                try:
                    for number, bindings in enumerate(((1, 1900, 1, 0), (2, 0, 7, 3),
                                                       (0, 9999, 7, 0), (1, 0, 0, 0),
                                                       (1, 0, 65, 17), (1, 1900, 1, 0))):
                        group, threshold, limit, offset = bindings
                        cur.execute("SET @g=%s,@p=%s,@n=%s,@o=%s", bindings)
                        observe(cur, records, f"{phase}_prepared_{number}",
                                "EXECUTE lookup_stmt USING @g,@p,@n,@o")
                        records[-1]["bindings"] = list(bindings)
                        expected = [[i, value] for i, (g, value) in sorted(data.items())
                                    if g == group and value is not None and value >= threshold]
                        if records[-1]["rows"] != expected[offset:offset + limit]:
                            raise AssertionError(f"Prepared lookup mismatch: {phase}/{number}")
                finally:
                    cur.execute("DEALLOCATE PREPARE lookup_stmt")
                # Unlike the threshold query above, this prepared lookup has
                # no data-table residual filter and exercises the demand cap.
                statement = ("SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ id,payload "
                             "FROM docs FORCE INDEX(ix_group) WHERE group_id=? "
                             "ORDER BY id LIMIT ? OFFSET ?")
                cur.execute("PREPARE unfiltered_lookup FROM %s", (statement,))
                try:
                    for number, bindings in enumerate(((1, 1, 0), (2, 65, 17), (0, 7, 3),
                                                       (1, 0, 0), (1, 1, 0))):
                        group, limit, offset = bindings
                        cur.execute("SET @g=%s,@n=%s,@o=%s", bindings)
                        observe(cur, records, f"{phase}_unfiltered_prepared_{number}",
                                "EXECUTE unfiltered_lookup USING @g,@n,@o")
                        records[-1]["bindings"] = list(bindings)
                        expected = [[i, value] for i, (g, value) in sorted(data.items()) if g == group]
                        if records[-1]["rows"] != expected[offset:offset + limit]:
                            raise AssertionError(f"Unfiltered prepared lookup mismatch: {phase}/{number}")
                finally:
                    cur.execute("DEALLOCATE PREPARE unfiltered_lookup")
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    records, plans = capture(args.port)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "captured", "script_sha256": digest, "case_count": len(records),
              "records": records, "plans": plans}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if before["script_sha256"] == digest and before["records"] == records else "fail"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} native lookup cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Native binary-key differential for document-only comparison and projection.

The original scored MATCH > 0 path is the per-query comparator reference. A
separate source corpus validates every returned key/payload and pagination.
Capture the same script against official and candidate builds; existing FTS
collectors and their baseline hashes are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from native_extended_regression import exact


TERMS = ["quartz", "zephyr", "orchid", "marigold", "saffron", "topaz", "onyx", "cobalt", "willow"]


def make_corpus():
    special = [b"", b"\x00", b"\x00\x00", b"\x00 ", b" ", b"  ", b"a", b"a ",
               b"a  ", b"a\x00", b"a\x00\x00", b"a\xff", b"\x7f", b"\x80", b"\xff",
               b"\xff\x00", b"\xff ", b"\xff\xff", b"p", b"p\x00", b"p\x00\x00"]
    keys = special + [b"p" + ident.to_bytes(2, "big") + bytes([[0, 32, 128, 255][ident % 4]])
                      + (b"  " if ident % 5 == 0 else b"") for ident in range(2113)]
    if len(keys) != len(set(keys)):
        raise AssertionError("Binary-key source corpus accidentally contains duplicate keys")
    if max(map(len, keys)) > 16:
        raise AssertionError("Binary-key source corpus exceeds the VARBINARY(16) fixture schema")
    data = {}
    for ordinal, key in enumerate(keys):
        words = [term for index, term in enumerate(TERMS) if (ordinal + index) % (index % 4 + 2)]
        if ordinal % 31 == 0:
            words = ["elsewhere"]
        elif ordinal % 37 == 0:
            words = TERMS[:]
        words += words[:1] * (ordinal % 5)
        words += ["batchprobe"]
        body = " ".join(words)
        if ordinal in (4, 7):
            body = None if ordinal == 4 else ""
        data[key] = (ordinal, None if ordinal % 17 == 0 else ordinal * 3, body)
    return data


def expected_rows(data, terms):
    return [(key.hex().upper(), ordinal, payload)
            for key, (ordinal, payload, body) in sorted(data.items())
            if set(terms).intersection((body or "").split())]


def observe(cur, records, name, sql, params=()):
    cur.execute(sql, params)
    rows = list(cur.fetchall())
    columns = [column[0] for column in cur.description]
    cur.execute("SHOW WARNINGS")
    records.append({"name": name, "sql": sql, "params": exact(params),
                    "columns": columns, "rows": exact(rows), "warnings": exact(cur.fetchall())})
    return rows


def insert_binary_rows(cur, rows):
    # Send binary keys as ASCII hex so the connection's utf8mb4 text charset
    # never decodes embedded NUL/high bytes. Check every statement's warnings.
    for start in range(0, len(rows), 256):
        chunk = rows[start:start + 256]
        params = [value for key, ordinal, payload, body in chunk
                  for value in (key.hex(), ordinal, payload, body)]
        sql = "INSERT INTO docs VALUES " + ",".join(["(UNHEX(%s),%s,%s,%s)"] * len(chunk))
        affected = cur.execute(sql, params)
        cur.execute("SHOW WARNINGS")
        warnings = cur.fetchall()
        if affected != len(chunk) or warnings:
            raise AssertionError(f"Binary corpus insertion changed rows or produced warnings: {affected}/{len(chunk)}, {warnings!r}")


def capture(port):
    import pymysql
    database = "fts_binary_ids_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=180)
    records, plans = [], []
    data = make_corpus()
    scenarios = [(f"k{k}", TERMS[:k]) for k in (1, 2, 8, 9)]
    scenarios += [("empty", ["absenta", "absentb"]), ("mixed", ["batchprobe", "absenta"])]
    pages = ((None, 0), (0, 0), (1, 0), (7, 3), (65, 17), (129, 997), (7, 3000))
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE docs(key1 VARBINARY(16) PRIMARY KEY,ordinal INT NOT NULL,payload INT,body TEXT)")
            insert_binary_rows(cur, [(key, *values) for key, values in data.items()])
            cur.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft_body(body) WITH PARSER space")
            for phase in ("initial", "after_dml"):
                if phase == "after_dml":
                    cur.execute("UPDATE docs SET payload=-ordinal,body='willow zephyr' WHERE ordinal BETWEEN 20 AND 40")
                    for key, (ordinal, payload, body) in list(data.items()):
                        if 20 <= ordinal <= 40:
                            data[key] = (ordinal, -ordinal, "willow zephyr")
                        elif 70 <= ordinal <= 90:
                            del data[key]
                    cur.execute("DELETE FROM docs WHERE ordinal BETWEEN 70 AND 90")
                    additions = [(b"a\x00\xff ", 9000, 27, "quartz zephyr batchprobe"),
                                 (b"\xff\x00\xff", 9001, None, "willow batchprobe")]
                    insert_binary_rows(cur, additions)
                    data.update({key: (ordinal, payload, body) for key, ordinal, payload, body in additions})
                for scenario, terms in scenarios:
                    query = " ".join(terms)
                    wanted = expected_rows(data, terms)
                    for batch in (7, 64, 256):
                        base = (f"SELECT /*+ OPT_PARAM('_rowsets_max_rows',{batch}) */ "
                                "HEX(key1) AS key_hex,ordinal,payload FROM docs WHERE MATCH(body) AGAINST(%s)")
                        for limit, offset in pages:
                            suffix = "" if limit is None else f" LIMIT {limit} OFFSET {offset}"
                            sql = base + " ORDER BY key1" + suffix
                            name = f"{phase}_{scenario}_batch{batch}_limit{limit}_offset{offset}"
                            rows = observe(cur, records, name, sql, (query,))
                            expected = wanted[offset:] if limit is None else wanted[offset:offset + limit]
                            if rows != expected:
                                raise AssertionError(f"Source binary identity/payload mismatch: {name}")
                            scored_sql = sql.replace("AGAINST(%s)", "AGAINST(%s)>0")
                            cur.execute(scored_sql, (query,))
                            scored_rows = list(cur.fetchall())
                            cur.execute("SHOW WARNINGS")
                            records[-1]["scored_rows"] = exact(scored_rows)
                            records[-1]["scored_warnings"] = exact(cur.fetchall())
                            if scored_rows != rows:
                                raise AssertionError(f"Original scored comparator differs: {name}")
                        if phase == "initial" and scenario == "k2":
                            cur.execute("EXPLAIN " + base + " ORDER BY key1 LIMIT 65 OFFSET 17", (query,))
                            plan = exact(cur.fetchall())
                            if "text retrieval" not in json.dumps(plan).lower():
                                raise AssertionError("Binary-key query did not use FTS")
                            plans.append({"batch_size": batch, "rows": plan})
                    for limit, offset in ((7, 0), (65, 17), (17, 3000)):
                        sql = ("SELECT COUNT(*) AS n FROM (SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ "
                               "key1 FROM docs WHERE MATCH(body) AGAINST(%s) "
                               f"LIMIT {limit} OFFSET {offset}) limited")
                        rows = observe(cur, records, f"{phase}_{scenario}_count_{limit}_{offset}", sql, (query,))
                        if rows != [(min(limit, max(len(wanted) - offset, 0)),)]:
                            raise AssertionError("Binary-key count lost final-batch/offset rows")
                prepared = ("SELECT /*+ OPT_PARAM('_rowsets_max_rows',64) */ HEX(key1) AS key_hex,ordinal,payload "
                            "FROM docs WHERE MATCH(body) AGAINST(?) ORDER BY key1 LIMIT ? OFFSET ?")
                cur.execute("PREPARE binary_union FROM %s", (prepared,))
                try:
                    for number, (k, limit, offset) in enumerate(((1, 1, 0), (9, 65, 17), (2, 129, 997),
                                                               (8, 7, 3), (1, 0, 0), (1, 1, 0))):
                        query = " ".join(TERMS[:k])
                        cur.execute("SET @q=%s,@n=%s,@o=%s", (query, limit, offset))
                        rows = observe(cur, records, f"{phase}_prepared_{number}", "EXECUTE binary_union USING @q,@n,@o")
                        records[-1]["bindings"] = [query, limit, offset]
                        if rows != expected_rows(data, TERMS[:k])[offset:offset + limit]:
                            raise AssertionError("Prepared binary-key query retained stale comparator/cursor state")
                finally:
                    cur.execute("DEALLOCATE PREPARE binary_union")
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
    print(f"{report['status']}: {len(records)} native binary-key cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

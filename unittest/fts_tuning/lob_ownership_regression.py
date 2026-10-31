"""Native FTS input-buffer ownership and reused-row TOKENIZE regression.

Creates only a fresh random database. LOB_INROW_THRESHOLD=0 forces the DDL scan
to materialize text outside the row. Each parser's index is created after data
insertion, then checked against source tokens through updates/deletes/inserts.
TOKENIZE results from a multirow scan must equal isolated scalar evaluations.

  python3 lob_ownership_regression.py --port 38680 --output results.json
Optional --compare before.json compares exact native results, never scores rounded
or sorted away. No service is started or stopped by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from native_extended_regression import exact, observe


def marker(index):
    return "artifact" + chr(ord("a") + index // 26) + chr(ord("a") + index % 26)


def document(index):
    return (f"{marker(index)} quartz zephyr cobalt ({marker(index).upper()}) " * 400).strip()


def capture(port):
    import pymysql

    database = "fts_lob_ownership_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=240)
    records = []
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cur.execute(f"USE `{database}`")
            cur.execute("CREATE TABLE inputs(id INT PRIMARY KEY, body VARCHAR(2048))")
            inputs = [(i, f"row{i} quartz zephyr MIXED_{i} café CAFÉ 中 (#orchid#) " * (1 + i % 11))
                      for i in range(64)]
            cur.executemany("INSERT INTO inputs VALUES(%s,%s)", inputs)
            for mode, suffix in (("default", ""), ("all", ",'[{\"output\":\"all\"}]'")):
                sql = (f"SELECT id,TOKENIZE(body,'beng'{suffix}),"
                       f"TOKENIZE(CONCAT(body,' final',id),'beng'{suffix}) FROM inputs ORDER BY id")
                observe(cur, records, f"multirow_{mode}", sql)
                rows = records[-1]["rows"]
                for (index, body), row in zip(inputs, rows, strict=True):
                    cur.execute(f"SELECT TOKENIZE(%s,'beng'{suffix}),TOKENIZE(%s,'beng'{suffix})",
                                (body, body + " final" + str(index)))
                    if exact(cur.fetchone()) != row[1:]:
                        raise AssertionError(f"TOKENIZE input/result lifetime mismatch: {mode}/{index}")
            for parser in ("space", "beng", "ik"):
                table = "lob_" + parser
                data = {i: document(i) for i in range(16)}
                cur.execute(f"CREATE TABLE {table}(id INT PRIMARY KEY, body LONGTEXT) LOB_INROW_THRESHOLD=0")
                cur.executemany(f"INSERT INTO {table} VALUES(%s,%s)", sorted(data.items()))
                # Building after insertion exercises ObFTIndexRowCache and its
                # caller's temporary out-of-row LOB materialization allocator.
                cur.execute(f"ALTER TABLE {table} ADD FULLTEXT INDEX ft_body(body) WITH PARSER {parser}")
                for phase in ("after_build", "after_dml"):
                    if phase == "after_dml":
                        data[3] = document(20)
                        cur.execute(f"UPDATE {table} SET body=%s WHERE id=3", (data[3],))
                        del data[5]
                        cur.execute(f"DELETE FROM {table} WHERE id=5")
                        data[30] = document(21)
                        cur.execute(f"INSERT INTO {table} VALUES(30,%s)", (data[30],))
                    for index in list(range(16)) + [20, 21]:
                        word = marker(index)
                        observe(cur, records, f"{parser}_{phase}_{word}",
                                f"SELECT id FROM {table} WHERE MATCH(body) AGAINST(%s) ORDER BY id", (word,))
                        expected = [[ident] for ident, body in sorted(data.items()) if word in body.split()]
                        if records[-1]["rows"] != expected:
                            raise AssertionError(f"LOB index lost/corrupted token: {parser}/{phase}/{word}")
                    observe(cur, records, f"{parser}_{phase}_common",
                            f"SELECT id FROM {table} WHERE MATCH(body) AGAINST('quartz zephyr') ORDER BY id")
                    if records[-1]["rows"] != [[ident] for ident in sorted(data)]:
                        raise AssertionError(f"LOB common-token mismatch: {parser}/{phase}")
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS `{database}`")
        finally:
            conn.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    records = capture(args.port)
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {"status": "captured", "script_sha256": digest, "case_count": len(records), "records": records}
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if before["script_sha256"] == digest and before["records"] == records else "fail"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} native input-ownership cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

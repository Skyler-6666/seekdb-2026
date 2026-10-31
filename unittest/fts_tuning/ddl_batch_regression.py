"""Compare exact native FTS DDL results before/after token-output batching.

Only a fresh random database is changed. No services, benchmark files, server
configuration or timers are changed. Run against separately started builds:

  python3 ddl_batch_regression.py --port 38680 --output before.json
  python3 ddl_batch_regression.py --port 38680 --output after.json --compare before.json

The document containing 4,105 distinct words exceeds the usual expression batch
capacity; the following documents detect an arena reuse or projected-flag leak.
Both primary-key (DDL sampling) and heap scan plans, out-of-row LOB materialization,
multi-column indexes, empty/NULL documents, DML and index rebuild are exercised.
Exact scores are retained. Successful ALTER INDEX completion also exercises the
existing DDL source checksum reporting; no internal checksum value is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from native_extended_regression import observe


def word(number):
    suffix = ""
    for _ in range(4):
        suffix = chr(ord("a") + number % 26) + suffix
        number //= 26
    return "boundary" + suffix


def source_rows():
    long_document = " ".join(word(i) for i in range(4105))
    rows = [
        (0, None, None),
        (1, "", ""),
        (2, "punctuation", "!!! --- ... ,,,"),
        (3, "quartz", "quartz quartz quartz zephyr"),
        (4, "long document", long_document),
        (5, "empty body", ""),
        (6, "null body", None),
        (7, None, "zephyr cobalt"),
        (8, "Unicode CAFÉ", "café CAFÉ café 中文 数据库"),
        (9, "the and a", "the and a is of to"),
    ]
    rows.extend((i, "title " + word(5000 + i),
                 (word(5000 + i) + " quartz ") * (1 + i % 9)) for i in range(10, 139))
    return rows


def capture(port):
    import pymysql

    database = "fts_ddl_batch_" + uuid.uuid4().hex
    conn = pymysql.connect(host="127.0.0.1", port=port, user="root",
                           password=os.environ.get("FTS_TEST_PASSWORD", ""),
                           charset="utf8mb4", autocommit=True, read_timeout=300)
    records = []
    created = False
    shapes = (
        ("single_space", "space", "id INT PRIMARY KEY,title VARCHAR(128),body LONGTEXT", "", "body"),
        ("lob_beng", "beng", "id INT PRIMARY KEY,title VARCHAR(128),body LONGTEXT",
         " LOB_INROW_THRESHOLD=0", "body"),
        ("multi_space", "space", "id INT PRIMARY KEY,title VARCHAR(128),body LONGTEXT", "", "title,body"),
        ("heap_space", "space", "id INT,title VARCHAR(128),body LONGTEXT", "", "body"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            created = True
            cur.execute(f"USE `{database}`")
            for table, parser, columns, options, indexed in shapes:
                cur.execute(f"CREATE TABLE {table}({columns}){options}")
                cur.executemany(f"INSERT INTO {table} VALUES(%s,%s,%s)", source_rows())
                build = f"ALTER TABLE {table} ADD FULLTEXT INDEX ft_body({indexed}) WITH PARSER {parser}"
                cur.execute(build)
                for phase in ("build", "dml", "rebuild"):
                    if phase == "dml":
                        cur.execute(f"UPDATE {table} SET body=%s WHERE id=3",
                                    ("cobalt " + word(4104),))
                        cur.execute(f"DELETE FROM {table} WHERE id=12")
                        cur.execute(f"INSERT INTO {table} VALUES(%s,%s,%s)",
                                    (200, "new title", "quartz " + word(4096)))
                    elif phase == "rebuild":
                        cur.execute(f"ALTER TABLE {table} DROP INDEX ft_body")
                        cur.execute(build)
                    prefix = f"{table}_{phase}"
                    observe(cur, records, prefix + "_source_checksum",
                            "SELECT COUNT(*),SUM(id),SUM(OCTET_LENGTH(body)),"
                            "BIT_XOR(CRC32(CONCAT(id,':',COALESCE(title,'<NULL>'),':',"
                            f"COALESCE(body,'<NULL>')))) FROM {table}")
                    queries = ("quartz", "zephyr", "quartz cobalt", word(0),
                               word(255), word(256), word(1024), word(4096), word(4104),
                               word(5010), word(5138), "nonexistentword")
                    for n, query in enumerate(queries):
                        match = f"MATCH({indexed}) AGAINST(%s IN NATURAL LANGUAGE MODE)"
                        observe(cur, records, f"{prefix}_score_{n}",
                                f"SELECT id,{match} AS score FROM {table} WHERE {match} ORDER BY id",
                                (query, query))
                        if n in (3, 4, 5, 6):
                            # Unique interior words of the long document remain
                            # searchable across exact and partial batch boundaries.
                            if [row[0] for row in records[-1]["rows"]] != [4]:
                                raise AssertionError(f"Missing/extra long-document token: {prefix}/{query}")
                    observe(cur, records, prefix + "_boolean",
                            f"SELECT id,MATCH({indexed}) AGAINST('+quartz -cobalt' IN BOOLEAN MODE) AS score "
                            f"FROM {table} WHERE MATCH({indexed}) AGAINST('+quartz -cobalt' IN BOOLEAN MODE) "
                            "ORDER BY id")
                    observe(cur, records, prefix + "_ranked",
                            f"SELECT id,MATCH({indexed}) AGAINST('quartz zephyr') AS score "
                            f"FROM {table} WHERE MATCH({indexed}) AGAINST('quartz zephyr') "
                            "ORDER BY score DESC,id LIMIT 9 OFFSET 3")
                    observe(cur, records, prefix + "_count",
                            f"SELECT COUNT(*) FROM {table} WHERE MATCH({indexed}) AGAINST('quartz zephyr')")
    finally:
        try:
            if created:
                with conn.cursor() as cur:
                    cur.execute(f"DROP DATABASE `{database}`")
        finally:
            conn.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--server-log", type=Path,
                        help="Optional local seekdb.log to verify actual token-batch execution")
    parser.add_argument("--require-batch", action="store_true",
                        help="Require a newly logged token batch containing more than one row")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    if args.require_batch and not args.server_log:
        parser.error("--require-batch needs --server-log")
    log_offset = args.server_log.stat().st_size if args.server_log else 0
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    records = capture(args.port)
    report = {"status": "captured", "script_sha256": script_hash,
              "case_count": len(records), "records": records}
    if args.server_log:
        evidence = []
        with args.server_log.open("rb") as handle:
            handle.seek(log_offset)
            for raw in handle:
                line = raw.decode("utf-8", errors="replace")
                if "FTS DDL token batch scan opened" in line or "FTS DDL first token batch" in line:
                    # Save only the new numeric diagnostic fields, never SQL,
                    # host metadata, input documents or unrelated log content.
                    fields = {key: int(value) for key, value in re.findall(
                        r"\b(row_count|batch_capacity|scan_type|output_count|is_fts_index_aux)=(\d+)", line)}
                    fields["event"] = "batch" if "FTS DDL first token batch" in line else "open"
                    evidence.append(fields)
        report["batch_evidence"] = evidence
        report["multiple_rows_observed"] = any(item.get("row_count", 0) > 1 for item in evidence)
    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        report["status"] = "pass" if (before["script_sha256"] == script_hash
                                              and before["records"] == records) else "fail"
    if args.require_batch and not report.get("multiple_rows_observed", False):
        report["status"] = "fail"
        report["failure"] = "No newly logged token batch larger than one row; activation is unproven"
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {len(records)} native FTS DDL cases")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Capture exact native SQL results, then compare independently started builds.

Only a fresh, randomly named database is created/dropped. SQL failures abort the
capture; no mock results, ignored errors, rounded scores or unordered result
normalization are used. The official benchmark and its data are not modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def exact(value):
    """Preserve SQL types and complete values in a portable JSON representation."""
    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AssertionError("Unexpected non-finite native SQL value")
        return {"double_hex": value.hex()}
    if isinstance(value, (tuple, list)):
        return [exact(item) for item in value]
    raise TypeError(f"Unexpected SQL result type: {type(value).__name__}")


def observe(cursor, records, name, sql, params=()):
    try:
        cursor.execute(sql, params)
    except Exception as error:
        raise RuntimeError(f"{name} failed: {sql!r} params={params!r}") from error
    columns = [item[0] for item in cursor.description]
    rows = exact(cursor.fetchall())
    cursor.execute("SHOW WARNINGS")
    warnings = exact(cursor.fetchall())
    records.append({"name": name, "sql": sql, "params": exact(params),
                    "columns": columns, "rows": rows, "warnings": warnings})


def capture_collation(cursor, records):
    cursor.execute("CREATE TABLE cmp (id INT PRIMARY KEY, s VARCHAR(128)) "
                   "DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci")
    values = [None, ""] + [chr(code) for code in range(128)]
    values += ["aA", "Aa", "a[", "a`", "a{", "a_", "a0", "a ", "a  ",
               "aa", "aa ", "a\t", "é", "E", "É", "café", "cafe", "中", "文",
               "中文a", "A中", "中A", "ｚ", "Ｚ", "ａ", "Ａ", "，", "。", "𠮷", "😀"]
    cursor.executemany("INSERT INTO cmp VALUES(%s,%s)", enumerate(values))
    for collation in ("utf8mb4_general_ci", "utf8mb4_bin", "utf8mb4_unicode_ci"):
        lhs, rhs = f"a.s COLLATE {collation}", f"b.s COLLATE {collation}"
        observe(cursor, records, f"collation_pairs_{collation}",
                "SELECT a.id,b.id," + ",".join(
                    f"({lhs} {op} {rhs})" for op in ("<", "=", ">", "<=", ">=", "<=>"))
                + ",STRCMP(" + lhs + "," + rhs + ") FROM cmp a CROSS JOIN cmp b ORDER BY a.id,b.id")
        observe(cursor, records, f"collation_sort_{collation}",
                f"SELECT id,HEX(s) FROM cmp ORDER BY s COLLATE {collation},id")
        observe(cursor, records, f"collation_group_{collation}",
                "SELECT MIN(id),COUNT(*),GROUP_CONCAT(id ORDER BY id) FROM cmp "
                f"GROUP BY s COLLATE {collation} ORDER BY MIN(id)")


def capture_tokenization(cursor, records):
    texts = ["lowercase ascii words repeated repeated", "UPPERCASE ASCII WORDS REPEATED REPEATED",
             "Case CASE cAse café cafe ÉCLAIR eclair", "abc\x00def 123\t456\r\n789",
             "ＡＢＣ ａｂｃ １２３ 中文全角。半角,", "中华人民共和国人民大会堂 研究生命起源",
             "三十公斤 10千米 20个 2000年", "αβ İ I i ı 𠮷😀", "尾随空格 trailing  ",
             ("数据库 OceanBase running 123，" * 256)]
    number = 0
    for parser in ("ik", "beng", "space", "ngram", "ngram2"):
        for collation in ("utf8mb4_general_ci", "utf8mb4_bin"):
            for text in texts:
                for full in (False, True):
                    value = f"CONVERT(%s USING utf8mb4) COLLATE {collation}"
                    args = (text, parser, '[{"output":"all"}]') if full else (text, parser)
                    sql = f"SELECT TOKENIZE({value},%s" + (",%s)" if full else ")") + " AS tokens"
                    observe(cursor, records, f"tokenize_{number}", sql, args)
                    number += 1
    # Alternate explicit max_word/default calls in the same connection. The native
    # GTest separately covers reuse of the lower-level property object itself.
    for mode in ("max_word", None, "smart", "max_word", None):
        options = [{"output": "all"}]
        if mode is not None:
            options.append({"additional_args": [{"ik_mode": mode}]})
        observe(cursor, records, f"ik_default_after_explicit_{number}",
                "SELECT TOKENIZE(%s,'ik',%s) AS tokens",
                ("研究生命起源 中华人民共和国人民大会堂", json.dumps(options)))
        number += 1


def capture_match(cursor, records):
    cursor.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category INT NOT NULL, "
                   "title VARCHAR(256), content TEXT, english TEXT) DEFAULT CHARSET=utf8mb4")
    chinese = ["数据库 全文索引 检索 OceanBase", "中文分词 标题 正文 人民大会堂",
               "内存 分配 排序 归并 数据库", "学习 研究 SQL 123 中英文混合"]
    english = ["Full text indexing builds searchable inverted lists.",
               "Running runners run toward a better database.",
               "The quick brown fox jumps over the lazy dog.",
               "Case CASE case; repeated repeated words and 123 values."]
    documents = [(i - 300, i % 7, f"文档{i} {chinese[i % 4]}",
                  (chinese[(i * 3) % 4] + " café αβ ＡＢＣ 𠮷😀 ") * (1 + i % 7),
                  english[(i * 3) % 4] * (1 + i % 9)) for i in range(600)]
    documents += [(400, 8, None, None, None), (401, 8, "", "", ""),
                  (402, 8, "the and of", "the and of", "the and of")]
    cursor.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s,%s)", documents)
    for name, columns, parser in (("all_ik", "title,content", "ik"),
                                   ("content_ik", "content", "ik"),
                                   ("english_beng", "english", "beng")):
        cursor.execute(f"ALTER TABLE docs ADD FULLTEXT INDEX {name}({columns}) WITH PARSER {parser}")
    observe(cursor, records, "all_documents", "SELECT * FROM docs ORDER BY id")
    queries = [("title,content", "数据库 OceanBase", ""),
               ("content", "内存 排序", ""), ("title,content", "missingtokenzz", ""),
               ("english", "database running", ""), ("english", "full text indexing", ""),
               ("title,content", "+数据库 -全文索引", " IN BOOLEAN MODE"),
               ("title,content", '"全文索引"', " IN BOOLEAN MODE"),
               ("english", "+database -running", " IN BOOLEAN MODE")]
    for number, (columns, text, mode) in enumerate(queries):
        match = f"MATCH({columns}) AGAINST(%s{mode})"
        observe(cursor, records, f"match_ids_{number}",
                f"SELECT id FROM docs WHERE {match} ORDER BY id", (text,))
        observe(cursor, records, f"match_count_{number}",
                f"SELECT COUNT(*) FROM docs WHERE {match}", (text,))
        observe(cursor, records, f"match_scores_{number}",
                f"SELECT id,{match} AS score FROM docs WHERE {match} ORDER BY id", (text, text))
        observe(cursor, records, f"match_score_order_{number}",
                f"SELECT id,{match} AS score FROM docs WHERE {match} ORDER BY score DESC,id LIMIT 20",
                (text, text))
        for threshold in (0, 0.5, 2, 10):
            observe(cursor, records, f"match_threshold_{number}_{threshold}",
                    f"SELECT id FROM docs WHERE {match}>%s ORDER BY id", (text, threshold))
        for suffix in (" AND id<0", " OR id=401", " AND category=3"):
            observe(cursor, records, f"match_compound_{number}_{suffix}",
                    f"SELECT id FROM docs WHERE {match}{suffix} ORDER BY id", (text,))
        observe(cursor, records, f"match_group_{number}",
                f"SELECT category,COUNT(*),MIN(id),MAX(id) FROM docs WHERE {match} "
                "GROUP BY category ORDER BY category", (text,))
        observe(cursor, records, f"match_distinct_{number}",
                f"SELECT DISTINCT category FROM docs WHERE {match} ORDER BY category", (text,))
        observe(cursor, records, f"match_limit_{number}",
                f"SELECT id FROM docs WHERE {match} ORDER BY id LIMIT 3,20", (text,))


def capture(args):
    import pymysql
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new evidence output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    database = "fts_extended_" + uuid.uuid4().hex
    report = {"schema": 1, "complete": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "records": []}
    handle = pymysql.connect(host=args.host, port=args.port, user=args.user,
                             password=os.getenv("FTS_TEST_PASSWORD", ""), charset="utf8mb4",
                             autocommit=True, connect_timeout=15, read_timeout=180, write_timeout=180)
    created = False
    try:
        with handle.cursor() as cursor:
            cursor.execute("SELECT VERSION()")
            report["server_version"] = exact(cursor.fetchone()[0])
            cursor.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            created = True
            cursor.execute(f"USE `{database}`")
            capture_collation(cursor, report["records"])
            capture_tokenization(cursor, report["records"])
            capture_match(cursor, report["records"])
            report["complete"] = True
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            if created:
                with handle.cursor() as cursor:
                    cursor.execute(f"DROP DATABASE `{database}`")
        finally:
            handle.close()
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "cases": len(report["records"]), "output": str(output)}))


def compare(args):
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    for value in (before, after):
        if value.get("schema") != 1 or value.get("complete") is not True or not value.get("records"):
            raise AssertionError("Incomplete or unrecognized native capture")
    if before["script_sha256"] != after["script_sha256"]:
        raise AssertionError("Captures used different regression scripts")
    left, right = before["records"], after["records"]
    if len(left) != len(right):
        raise AssertionError(f"Different case counts: {len(left)} != {len(right)}")
    mismatches = [a["name"] for a, b in zip(left, right) if a != b]
    if mismatches:
        raise AssertionError("Native result mismatch: " + ", ".join(mismatches))
    digest = hashlib.sha256(json.dumps(left, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    print(json.dumps({"equal": True, "cases": len(left), "records_sha256": digest}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    runner = modes.add_parser("capture")
    runner.add_argument("--host", default="127.0.0.1")
    runner.add_argument("--port", type=int, required=True)
    runner.add_argument("--user", default="root")
    runner.add_argument("--output", required=True)
    runner.set_defaults(run=capture)
    checker = modes.add_parser("compare")
    checker.add_argument("before")
    checker.add_argument("after")
    checker.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

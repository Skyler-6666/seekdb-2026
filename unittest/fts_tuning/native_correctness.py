"""Compare complete native TOKENIZE/MATCH results on two compiled seekdb builds.

Requires real servers and PyMySQL. There is no mock, SQLite or precomputed-result
fallback. Each side uses a fresh, uniquely named test database, never user tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


def jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def normalize_token_json(value, *, full_output=False, allow_null=False):
    """Validate the real native shape and preserve every array item/order.

    The full form is {"tokens": [{"word": frequency}, ...], "doc_len": n},
    not a list of {"word": ..., "freq": ...} records. SQL NULL is tolerated
    only for the explicitly tested NULL/empty input behavior.
    """
    value = jsonable(value)
    value = json.loads(value) if isinstance(value, str) else value
    if value is None and allow_null:
        return value
    if full_output:
        if not isinstance(value, dict) or set(value) != {"tokens", "doc_len"}:
            raise ValueError("Full TOKENIZE output must contain tokens and doc_len")
        if type(value["doc_len"]) is not int or value["doc_len"] < 0 or not isinstance(value["tokens"], list):
            raise ValueError("Invalid TOKENIZE document length or token array")
        for token in value["tokens"]:
            if not isinstance(token, dict) or len(token) != 1:
                raise ValueError("Full TOKENIZE token must be a single word/frequency pair")
            word, frequency = next(iter(token.items()))
            if not isinstance(word, str) or not word or type(frequency) is not int or frequency <= 0:
                raise ValueError("Invalid TOKENIZE word or integer frequency")
    elif not isinstance(value, list) or any(not isinstance(token, str) or not token for token in value):
        raise ValueError("Default TOKENIZE output must be an array of token strings")
    return value


def equivalent_rows(before, after, *, relative=1e-8, absolute=1e-10):
    """Document identity is exact; only the finite MATCH score gets tolerance."""
    if len(before) != len(after):
        return False
    for left, right in zip(before, after):
        if len(left) != 2 or len(right) != 2 or type(left[0]) is not int or type(right[0]) is not int or left[0] != right[0]:
            return False
        lhs, rhs = left[1], right[1]
        if type(lhs) not in (int, float) or type(rhs) not in (int, float):
            return False
        if not math.isfinite(lhs) or not math.isfinite(rhs) or not math.isclose(lhs, rhs, rel_tol=relative, abs_tol=absolute):
            return False
    return True


def corpus():
    chinese = ["数据库 全文索引 检索 OceanBase", "中文分词 标题 正文 人民大会堂", "内存 分配 排序 归并 数据库", "学习 研究 SQL 123 中英文混合"]
    english = ["Full text indexing builds searchable inverted lists.", "Running runners run toward a better database.", "The quick brown fox jumps over the lazy dog.", "Case CASE case; repeated repeated words and 123 values."]
    result = []
    for i in range(128):
        title = f"文档{i} {chinese[i % len(chinese)]}"
        content = (chinese[(i * 3) % 4] + "，。！ αβ café ＡＢＣ 𠮷😀\t") * (1 + i % 7)
        result.append((i - 64, i % 5, title, content, f"Document {i} {english[i % 4]}", english[(i * 3) % 4] * (1 + i % 9)))
    return result


def token_cases():
    texts = [None, "", " \t\r\n", "The the THE running runners runs", "hello hello HELLO", "OceanBase是一款非常稳定的数据库，在110个星球上都著名", "中华人民共和国人民大会堂 人际关系", "abc英文空格 中文分号；英文分号;全角空格　全角逗号，等", "café naïve αβ 𠮷😀 ocean_base v1.2.3 foo-bar O'Reilly", ("数据库 OceanBase running 123，" * 512)]
    profiles = [(parser, None) for parser in ("ik", "beng", "space", "ngram", "ngram2")]
    profiles += [(parser, '[{"output":"all"}]') for parser in ("ik", "beng", "space", "ngram", "ngram2")]
    profiles += [("ik", '[{"output":"all"},{"additional_args":[{"ik_mode":"smart"}]}]'), ("ik", '[{"output":"all"},{"additional_args":[{"ik_mode":"max_word"}]}]')]
    cases = [(parser, options, text, None) for parser, options in profiles for text in texts]
    # The real group-by map compares using the input collation. Validate full
    # frequency/last-word bytes under equivalent accented/case variants too.
    cases += [(parser, '[{"output":"all"}]', text, collation)
              for parser in ("space", "beng", "ik")
              for collation in ("utf8mb4_general_ci", "utf8mb4_bin")
              for text in ("caféx cafex caféx", "Case CASE cAse", "éclairx eclairx ÉCLAIRX")]
    return cases


def query_cases():
    profiles = (
        ("title,content", ["数据库", "OceanBase", "中文分词 人民大会堂", "SQL 中文", "unittestfreshsignal"]),
        ("content", ["数据库", "内存 排序", "café αβ", "unittestfreshsignal"]),
        ("title_en,content_en", ["database", "running", "full text indexing", "case 123", "unittestfreshsignal"]),
    )
    return [(columns, query) for columns, queries in profiles for query in queries]


def non_relevance_result(cursor, columns, query, scored_rows):
    """Check full identity before counts on the unprojected MATCH path.

    The predicate uses the official benchmark's implicit boolean MATCH form;
    neither its ID projection nor either COUNT projects/orders by relevance.
    """
    if not equivalent_rows(scored_rows, scored_rows):
        raise AssertionError("Invalid full relevance result used as identity reference")
    predicate = f"MATCH({columns}) AGAINST(%s)"
    cursor.execute(f"SELECT id FROM docs WHERE {predicate} ORDER BY id", (query,))
    rows = jsonable(cursor.fetchall())
    if any(len(row) != 1 or type(row[0]) is not int for row in rows):
        raise AssertionError("Non-relevance query returned invalid document IDs")
    ids = [row[0] for row in rows]
    if ids != sorted(set(ids)) or ids != sorted(row[0] for row in scored_rows):
        raise AssertionError("Complete non-relevance identity differs from relevance projection")

    counts = []
    for sql, expected in (
        (f"SELECT COUNT(*) FROM docs WHERE {predicate}", len(ids)),
        (f"SELECT COUNT(*) FROM (SELECT id FROM docs WHERE {predicate} LIMIT 20) t", min(len(ids), 20)),
    ):
        cursor.execute(sql, (query,))
        count_rows = jsonable(cursor.fetchall())
        if len(count_rows) != 1 or len(count_rows[0]) != 1 or type(count_rows[0][0]) is not int or count_rows[0][0] != expected:
            raise AssertionError("Non-relevance COUNT/LIMIT count differs from the full document set")
        counts.append(count_rows[0][0])
    return {"ids": ids, "count": counts[0], "limit20_count": counts[1]}


def connection(prefix):
    import pymysql
    host = os.getenv(prefix + "_HOST", "127.0.0.1")
    port = int(os.getenv(prefix + "_PORT", "2881" if prefix == "FTS_BEFORE" else "3881"))
    handle = pymysql.connect(host=host, port=port, user=os.getenv(prefix + "_USER", "root"), password=os.getenv(prefix + "_PASSWORD", ""), charset="utf8mb4", autocommit=True, connect_timeout=15, read_timeout=180, write_timeout=180)
    return handle, {"host": host, "port": port}


def is_expected_tokenize_argument_error(text, code):
    # src/share/ob_errno.def maps OB_INVALID_ARGUMENT to ER_WRONG_ARGUMENTS
    # (1210) and OB_NOT_SUPPORTED to ER_NOT_SUPPORTED_YET (1235). NULL/empty
    # input can take the legacy argument/charset path. Resource failures,
    # missing dictionary tables and every other server/client error must stop
    # validation, even when both instances happen to fail the same way.
    return text in (None, "") and code in {1210, 1235}


def native_side(handle, database, keep, data):
    import pymysql
    # Mutate the report object so failed cases retain all completed native
    # observations instead of losing partial evidence on an exception.
    data.update(tokenize=[], queries=[], non_relevance_queries=[], rows={})
    with handle.cursor() as cursor:
        cursor.execute("SELECT VERSION()")
        data["server_version"] = jsonable(cursor.fetchone()[0])
        cursor.execute("SHOW VARIABLES LIKE 'version%'")
        data["server_version_variables"] = jsonable(cursor.fetchall())
        try:
            for number, (parser, options, text, collation) in enumerate(token_cases()):
                try:
                    expression = "%s" if collation is None else f"CONVERT(%s USING utf8mb4) COLLATE {collation}"
                    if options is None:
                        cursor.execute(f"SELECT TOKENIZE({expression},%s) AS tokens", (text, parser))
                    else:
                        cursor.execute(f"SELECT TOKENIZE({expression},%s,%s) AS tokens", (text, parser, options))
                    raw_value = cursor.fetchone()[0]
                    result = {"status": "success", "value": normalize_token_json(raw_value, full_output=options is not None, allow_null=text in (None, ""))}
                except pymysql.MySQLError as error:
                    # Only the NULL/empty inputs intentionally permit a SQL
                    # argument error. Environment failures are never acceptance.
                    code = int(error.args[0]) if error.args else 0
                    if not is_expected_tokenize_argument_error(text, code):
                        raise
                    result = {"status": "sql_argument_error", "code": int(error.args[0])}
                data["tokenize"].append({"case": number, "parser": parser, "options": options, "collation": collation, "text_sha256": None if text is None else hashlib.sha256(text.encode()).hexdigest(), **result})
            cursor.execute(f"CREATE DATABASE `{database}` DEFAULT CHARACTER SET utf8mb4")
            cursor.execute(f"USE `{database}`")
            cursor.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category INT NOT NULL, title VARCHAR(256) NOT NULL, content TEXT NOT NULL, title_en VARCHAR(256) NOT NULL, content_en TEXT NOT NULL) DEFAULT CHARSET=utf8mb4")
            rows = corpus()
            expected_rows = {row[0]: list(row) for row in rows}
            cursor.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s,%s,%s)", rows)
            for name, columns, parser in (("fti_all_ik", "title,content", "ik"), ("fti_content_ik", "content", "ik"), ("fti_en_beng", "title_en,content_en", "beng")):
                cursor.execute(f"ALTER TABLE docs ADD FULLTEXT INDEX {name}({columns}) WITH PARSER {parser}")
            for phase in ("built", "after_dml"):
                if phase == "after_dml":
                    cursor.execute("UPDATE docs SET content=CONCAT(content,%s),content_en=CONCAT(content_en,%s) WHERE id BETWEEN 0 AND 9", (" 更新信号freshsignal", " freshsignal updated"))
                    cursor.execute("DELETE FROM docs WHERE id BETWEEN -64 AND -60")
                    extra = [(1000+i, i%5, "新增freshsignal", "全文新增 unittestfreshsignal", "Fresh insertion", "unittestfreshsignal new native document") for i in range(5)]
                    cursor.executemany("INSERT INTO docs VALUES(%s,%s,%s,%s,%s,%s)", extra)
                    for key in range(0, 10):
                        expected_rows[key][3] += " 更新信号freshsignal"
                        expected_rows[key][5] += " freshsignal updated"
                    for key in range(-64, -59):
                        del expected_rows[key]
                    expected_rows.update({row[0]: list(row) for row in extra})
                cursor.execute("SELECT id,category,title,content,title_en,content_en FROM docs ORDER BY id")
                actual = jsonable(cursor.fetchall())
                data["rows"][phase] = actual
                if actual != [expected_rows[key] for key in sorted(expected_rows)]:
                    raise AssertionError("Full corpus readback differs from the imported/updated source")
                for columns, query in query_cases():
                    full_scored_rows = None
                    for limit in (None, 1, 5, 20):
                        sql = f"SELECT id,MATCH({columns}) AGAINST(%s IN NATURAL LANGUAGE MODE) AS score FROM docs WHERE MATCH({columns}) AGAINST(%s IN NATURAL LANGUAGE MODE)>0 ORDER BY score DESC,id ASC"
                        if limit is not None:
                            sql += f" LIMIT {limit}"
                        cursor.execute(sql, (query, query))
                        result = jsonable(cursor.fetchall())
                        if any(not math.isfinite(row[1]) or row[1] <= 0 for row in result):
                            raise AssertionError("MATCH produced an invalid score")
                        data["queries"].append({"phase": phase, "columns": columns, "query": query, "limit": limit, "rows": result})
                        if limit is None:
                            full_scored_rows = result
                    plain = non_relevance_result(cursor, columns, query, full_scored_rows)
                    data["non_relevance_queries"].append({"phase": phase, "columns": columns, "query": query, **plain})
                if phase == "after_dml":
                    anchors = [case for case in data["queries"] if case["phase"] == phase and case["query"] == "unittestfreshsignal" and case["limit"] is None]
                    if any({row[0] for row in case["rows"]} != set(range(1000, 1005)) for case in anchors):
                        raise AssertionError("Newly inserted exact-term documents are absent from native MATCH")
        finally:
            if not keep:
                # Only this generated identifier can be removed, never a
                # caller-supplied schema or the official benchmark database.
                assert re.fullmatch(r"fts_tuning_verify_[a-f0-9]{16}", database)
                cursor.execute(f"DROP DATABASE IF EXISTS `{database}`")
    return data


def compare(before, after):
    failures = []
    if before["tokenize"] != after["tokenize"]:
        failures.append("Complete TOKENIZE JSON/ordering/frequency/error behavior differs")
    if before["rows"] != after["rows"]:
        failures.append("Full stored corpus readback differs")
    if before["non_relevance_queries"] != after["non_relevance_queries"]:
        failures.append("Non-relevance full document identity/COUNT/LIMIT count differs")
    if len(before["queries"]) != len(after["queries"]):
        failures.append("MATCH case count differs")
    else:
        for left, right in zip(before["queries"], after["queries"]):
            if {key: value for key, value in left.items() if key != "rows"} != {key: value for key, value in right.items() if key != "rows"} or not equivalent_rows(left["rows"], right["rows"]):
                failures.append(f"MATCH differs: {left['phase']}/{left['columns']}/{left['query']}/limit={left['limit']}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-revision", required=True)
    parser.add_argument("--after-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-databases", action="store_true")
    args = parser.parse_args()
    for revision in (args.before_revision, args.after_revision):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
            raise ValueError("Declare full 40-character source revisions")
    args.before_revision = args.before_revision.lower()
    args.after_revision = args.after_revision.lower()
    if args.before_revision == args.after_revision:
        raise ValueError("Before and after revisions must differ")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "declared_revisions": {"before": args.before_revision, "after": args.after_revision}, "performance_or_score_measured": False}
    handles = []
    try:
        before, before_endpoint = connection("FTS_BEFORE")
        handles.append(before)
        after, after_endpoint = connection("FTS_AFTER")
        handles.append(after)
        if before_endpoint == after_endpoint:
            raise ValueError("Use separate native before/after instances")
        database = "fts_tuning_verify_" + uuid.uuid4().hex[:16]
        report["test_database"] = database
        report["before"] = {}
        native_side(before, database, args.keep_databases, report["before"])
        report["after"] = {}
        native_side(after, database, args.keep_databases, report["after"])
        failures = compare(report["before"], report["after"])
        report.update(status="passed" if not failures else "failed", failures=failures,
                      compared_tokenize_cases=len(report["before"]["tokenize"]),
                      compared_match_cases=len(report["before"]["queries"]),
                      compared_non_relevance_cases=len(report["before"]["non_relevance_queries"]),
                      compared_non_relevance_statements=3 * len(report["before"]["non_relevance_queries"]),
                      compared_match_statements=len(report["before"]["queries"]) + 3 * len(report["before"]["non_relevance_queries"]))
        return 0 if not failures else 1
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for handle in handles:
            handle.close()
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        (args.output / "native-correctness.json").write_bytes((json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    raise SystemExit(main())

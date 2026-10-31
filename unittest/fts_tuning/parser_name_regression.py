"""Capture exact native TOKENIZE parser-name and prepared-parameter behavior.

Run unchanged against official and candidate binaries. Complete JSON strings,
SQL errors and SHOW WARNINGS are preserved; expected-error inputs are never
silently discarded. This collector creates no tables and changes no benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from native_extended_regression import exact


PARSERS = ("space", "ngram", "beng", "ik", "ngram2")
TEXTS = ("Case CASE cAse running runners 123 abcdef", "中华人民共和国人民大会堂 研究生命起源", None, "")
ALL = '[{"output":"all"}]'
SQL_TWO = "SELECT TOKENIZE(%s,%s) AS tokens"
SQL_THREE = "SELECT TOKENIZE(%s,%s,%s) AS tokens"
# Transport/resource failures must abort even for a deliberately invalid name.
FATAL_CODES = {1021, 1037, 1038, 1040, 1041, 1043, 1047, 1053, 1114,
               1158, 1159, 1160, 1161, 1203, 1205, 1213, 1317, 3024}


def observe_outcome(cur, records, name, sql, params=(), *, may_error=False):
    import pymysql
    record = {"name": name, "sql": sql, "params": exact(params)}
    try:
        cur.execute(sql, params)
        record["columns"] = [item[0] for item in cur.description] if cur.description else []
        record["rows"] = exact(cur.fetchall()) if cur.description else []
    except pymysql.MySQLError as error:
        code = error.args[0] if error.args else None
        if not isinstance(code, int) or code in FATAL_CODES or 2000 <= code < 3000:
            raise
        record["error"] = {"code": code, "message": str(error.args[1])}
    cur.execute("SHOW WARNINGS")
    record["warnings"] = exact(cur.fetchall())
    records.append(record)
    if "error" in record and not may_error:
        raise AssertionError(f"Required native statement failed: {name}: {record['error']}")
    return record


def validate_tokens(record):
    if "error" in record:
        return
    if record["columns"] != ["tokens"] or len(record["rows"]) != 1 or len(record["rows"][0]) != 1:
        raise AssertionError(f"Invalid TOKENIZE result shape: {record['name']}")
    raw = record["rows"][0][0]
    if isinstance(raw, dict) and set(raw) == {"bytes_hex"}:
        raw = bytes.fromhex(raw["bytes_hex"]).decode("utf-8")
    if not isinstance(raw, str):
        raise AssertionError(f"TOKENIZE did not return complete JSON: {record['name']}")
    parsed = json.loads(raw)
    if not isinstance(parsed, (list, dict)):
        raise AssertionError(f"Unexpected TOKENIZE JSON root: {record['name']}")


def token(cur, records, name, text, parser, options=None, *, may_error=False):
    sql, args = (SQL_TWO, (text, parser)) if options is None else (SQL_THREE, (text, parser, options))
    record = observe_outcome(cur, records, name, sql, args, may_error=may_error)
    validate_tokens(record)
    return record


def same_result(left, right):
    # Keep array/token order and JSON bytes; do not normalize or sort output.
    keys = ("columns", "rows", "error", "warnings")
    return all(left.get(key) == right.get(key) for key in keys)


def check_raw_empty_capability(record, text):
    # Official SQL NULL can retain binary metadata even though TOKENIZE treats
    # its contents as empty. Preserve that parser capability error exactly.
    if "error" in record and not (text in (None, "") and record["error"]["code"] == 1235
                                   and "binary charset" in record["error"]["message"]):
        raise AssertionError(f"Unexpected canonical parser failure: {record['name']}: {record['error']}")


def properties(**values):
    return json.dumps([{"output": "all"}, {"additional_args": [{key: value} for key, value in values.items()]}])


def property_cases():
    return [
        ("space", properties(min_token_size=1, max_token_size=84), True),
        ("space", properties(min_token_size=5, max_token_size=12), True),
        ("beng", properties(min_token_size=1, max_token_size=84), True),
        ("beng", properties(min_token_size=5, max_token_size=12), True),
        ("ngram", properties(ngram_token_size=1), True),
        ("ngram", properties(ngram_token_size=3), True),
        ("ngram2", properties(min_ngram_size=1, max_ngram_size=3), True),
        ("ngram2", properties(min_ngram_size=2, max_ngram_size=4), True),
        ("ik", properties(ik_mode="smart"), True),
        ("ik", properties(ik_mode="max_word"), True),
        ("ik", ALL, True),
        ("ik", properties(ik_mode="unknown"), False),
        ("ngram", properties(ngram_token_size=0), False),
        ("space", properties(min_token_size=12, max_token_size=1), False),
        ("beng", '[{"output":"unsupported"}]', False),
        ("ngram2", '[{"additional_args":[{"unknown_property":1}]}]', False),
        ("ik", '{"output":"all"}', False),
        ("space", '[{"output":', False),
    ]


def capture_names(cur, records):
    for parser in PARSERS:
        mixed = "".join(letter.upper() if index % 2 else letter for index, letter in enumerate(parser))
        names = (parser.upper(), mixed, "  " + parser + "  ", "\t" + parser.upper() + "\r\n")
        for text_index, text in enumerate(TEXTS):
            for mode, options in (("default", None), ("all", ALL)):
                reference = token(cur, records, f"canonical_{parser}_{text_index}_{mode}", text, parser, options,
                                  may_error=text in (None, ""))
                check_raw_empty_capability(reference, text)
                for index, name in enumerate(names):
                    result = token(cur, records, f"variant_{parser}_{text_index}_{mode}_{index}", text, name, options,
                                   may_error=text in (None, ""))
                    check_raw_empty_capability(result, text)
                    if not same_result(reference, result):
                        raise AssertionError(f"Canonical parser-name equivalence changed: {result['name']}")
    for index, text in enumerate(TEXTS):
        reference = observe_outcome(cur, records, f"omitted_parser_{index}", "SELECT TOKENIZE(%s) AS tokens", (text,),
                                    may_error=text in (None, ""))
        validate_tokens(reference)
        check_raw_empty_capability(reference, text)
        result = token(cur, records, f"null_parser_{index}", text, None, may_error=text in (None, ""))
        check_raw_empty_capability(result, text)
        if not same_result(reference, result):
            raise AssertionError("NULL parser no longer selects the omitted-parser default")
    for parser in PARSERS:
        for index, text in enumerate((None, "")):
            for mode, options in (("default", None), ("all", ALL)):
                expression = "CONVERT(%s USING utf8mb4) COLLATE utf8mb4_general_ci"
                sql = f"SELECT TOKENIZE({expression},%s" + (",%s)" if options else ")") + " AS tokens"
                reference = None
                for variant, name in enumerate((parser, parser.upper())):
                    args = (text, name, options) if options else (text, name)
                    result = observe_outcome(cur, records, f"typed_empty_{parser}_{index}_{mode}_{variant}", sql, args)
                    validate_tokens(result)
                    if reference is not None and not same_result(reference, result):
                        raise AssertionError("Explicit UTF-8 empty-input parser dispatch differs")
                    reference = result
    names = ["", " ", "\t\r\n", "unknown", "UNKNOWN", "beng.1", "IK.1", ".beng", "beng.",
             "beng..1", "space_name", "ngram3", "i k", "ＢＥＮＧ", "İK", "\x00", "\x00beng",
             "beng\x00", "beng\x00unknown", "ik\x00.1", "beng\nunknown"]
    names += ["b" * length for length in (63, 64, 65)]
    names += ["beng" + " " * length for length in (60, 61)]
    for index, name in enumerate(names):
        for mode, options in (("default", None), ("all", ALL)):
            token(cur, records, f"boundary_{index}_{mode}", TEXTS[0], name, options, may_error=True)
    for index, (parser, options, valid) in enumerate(property_cases()):
        reference = token(cur, records, f"property_lower_{index}", TEXTS[index % 2], parser, options, may_error=not valid)
        result = token(cur, records, f"property_upper_{index}", TEXTS[index % 2], parser.upper(), options, may_error=not valid)
        if not same_result(reference, result):
            raise AssertionError(f"Property/parser-case dispatch changed: {index}")


def capture_prepared(cur, records):
    statements = [("parser_two", "SELECT TOKENIZE(?,?) AS tokens", SQL_TWO),
                  ("parser_three", "SELECT TOKENIZE(?,?,?) AS tokens", SQL_THREE)]
    for statement, sql, direct_sql in statements:
        observe_outcome(cur, records, "prepare_" + statement, f"PREPARE {statement} FROM %s", (sql,))
        if statement == "parser_two":
            names = list(PARSERS) + [name.upper() for name in reversed(PARSERS)]
            names += [None, "unknown", "beng", "b" * 64, "ik", "beng\x00tail", "ngram2", "", "space"]
            rows = [(TEXTS[index % 2], name) for index, name in enumerate(names)]
            required = [name is None or name.lower() in PARSERS for name in names]
        else:
            profiles = property_cases() + [("UNKNOWN", ALL, False), ("IK", ALL, True),
                                         ("beng", ALL, True), (None, ALL, True), ("ngram2", ALL, True)]
            rows = [(TEXTS[index % 2], parser.upper() if parser and index % 2 else parser, options)
                    for index, (parser, options, valid) in enumerate(profiles)]
            required = [valid for parser, options, valid in profiles]
        try:
            for index, (bindings, must_succeed) in enumerate(zip(rows, required)):
                variables = ["@parser_text", "@parser_name"] + (["@parser_options"] if len(bindings) == 3 else [])
                observe_outcome(cur, records, f"bind_{statement}_{index}",
                                "SET " + ",".join(value + "=%s" for value in variables), bindings)
                result = observe_outcome(cur, records, f"execute_{statement}_{index}",
                                         f"EXECUTE {statement} USING " + ",".join(variables), may_error=not must_succeed)
                result["bindings"] = exact(bindings)
                validate_tokens(result)
                reference = observe_outcome(cur, records, f"direct_{statement}_{index}",
                                            direct_sql, bindings, may_error=not must_succeed)
                validate_tokens(reference)
                if not same_result(result, reference):
                    raise AssertionError(f"Prepared parser/property state differs from direct execution: {statement}/{index}")
        finally:
            observe_outcome(cur, records, "deallocate_" + statement, f"DEALLOCATE PREPARE {statement}")


def capture(port, records):
    import pymysql
    connection = pymysql.connect(host="127.0.0.1", port=port, user="root",
                                 password=os.environ.get("FTS_TEST_PASSWORD", ""),
                                 charset="utf8mb4", autocommit=True, read_timeout=180, write_timeout=180)
    try:
        with connection.cursor() as cur:
            capture_names(cur, records)
            capture_prepared(cur, records)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    report = {"status": "captured", "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "helper_sha256": hashlib.sha256(Path(__file__).with_name("native_extended_regression.py").read_bytes()).hexdigest(),
              "records": []}
    try:
        capture(args.port, report["records"])
        if args.compare:
            before = json.loads(args.compare.read_text(encoding="utf-8"))
            same = all(before[key] == report[key] for key in ("script_sha256", "helper_sha256", "records"))
            report["status"] = "pass" if same else "fail"
    except Exception as error:
        report["status"] = "fail"
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["case_count"] = len(report["records"])
        report["error_count"] = sum("error" in record for record in report["records"])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['case_count']} parser-name records, {report['error_count']} explicit SQL errors")
    if report["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

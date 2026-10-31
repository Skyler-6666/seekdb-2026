"""Regression for parser reuse across repeated TOKENIZE/build/query calls.

The test checks every call against an independently recorded result for the
same input and stresses interleaved parser configurations. It never caches an
expected SQL answer in the server.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import uuid


PARSERS = ("ik", "beng", "space", "ngram", "ngram2")


def tokenize(connection, text, parser, options=None):
    with connection.cursor() as cursor:
        if options is None:
            cursor.execute("SELECT TOKENIZE(%s,%s)", (text, parser))
        else:
            arguments = options if isinstance(options, list) else [options]
            cursor.execute("SELECT TOKENIZE(%s,%s,%s)",
                           (text, parser, json.dumps(arguments)))
        value = cursor.fetchone()[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int,
                        default=int(os.getenv("FTS_TEST_PORT", "2881")))
    parser.add_argument("--user", default=os.getenv("FTS_TEST_USER", "root"))
    args = parser.parse_args()

    import pymysql

    database = "fts_parser_reuse_" + uuid.uuid4().hex[:16]
    handle = pymysql.connect(
        host=args.host, port=args.port, user=args.user,
        password=os.getenv("FTS_TEST_PASSWORD", ""),
        charset="utf8mb4", autocommit=True)
    created = False
    try:
        with handle.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{database}` DEFAULT CHARSET utf8mb4")
            created = True
            cursor.execute(f"USE `{database}`")
            cursor.execute(
                "CREATE TABLE docs(id BIGINT PRIMARY KEY, title TEXT, body TEXT) "
                "DEFAULT CHARSET=utf8mb4")
            rows = []
            for i in range(320):
                marker = f"marker{i:04d}"
                chinese = "数据库 全文索引 分词器 性能 优化"
                english = f"Database retrieval {marker} repeated token"
                rows.append((i, f"{marker} {chinese}", f"{chinese} {english}"))
            cursor.executemany("INSERT INTO docs VALUES(%s,%s,%s)", rows)
            cursor.execute(
                "ALTER TABLE docs ADD FULLTEXT INDEX ft_all(title,body) WITH PARSER ik")
            cursor.execute(
                "ALTER TABLE docs ADD FULLTEXT INDEX ft_en(body) WITH PARSER beng")

        texts = [
            "数据库 全文索引 分词器 性能 优化",
            "Database retrieval performance and tokenizer",
            "OceanBase 中文 English 123，混合检索",
            "三十公斤 10千米 20个 2000年",
            "ＡＢＣ ａｂｃ １２３ 全角与半角",
        ]
        configurations = []
        for text_index, text in enumerate(texts):
            for parser_name in PARSERS:
                configurations.append((text_index, text, parser_name, None))
                configurations.append((text_index, text, parser_name,
                                       {"output": "all"}))
            for mode in ("smart", "max_word", None):
                options = [{"output": "all"}]
                if mode:
                    options.append({"additional_args": [{"ik_mode": mode}]})
                configurations.append((text_index, text, "ik", options))

        expected = {}
        order = list(range(len(configurations)))
        random.Random(20261030).shuffle(order)
        for index in order:
            _, text, parser_name, options = configurations[index]
            expected[index] = tokenize(handle, text, parser_name, options)

        # Repeat in different orders. A cache keyed only by parser, or a reused
        # parser retaining the prior document, changes at least one result.
        random.Random(42).shuffle(order)
        for index in order + order[::-1]:
            _, text, parser_name, options = configurations[index]
            actual = tokenize(handle, text, parser_name, options)
            if actual != expected[index]:
                raise AssertionError(
                    f"TOKENIZE changed after parser reuse: case={index} parser={parser_name}")

        # Independent connections exercise the same code concurrently.
        failures = []

        def worker(worker_index):
            local = pymysql.connect(
                host=args.host, port=args.port, user=args.user,
                password=os.getenv("FTS_TEST_PASSWORD", ""),
                charset="utf8mb4", autocommit=True)
            try:
                for offset in range(20):
                    index = (worker_index * 17 + offset * 7) % len(configurations)
                    _, text, parser_name, options = configurations[index]
                    actual = tokenize(local, text, parser_name, options)
                    if actual != expected[index]:
                        failures.append(index)
                        return
            finally:
                local.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise AssertionError(f"Concurrent parser reuse mismatch: {failures[:4]}")

        with handle.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM docs WHERE MATCH(title,body) "
                "AGAINST('数据库 全文索引')")
            if cursor.fetchone()[0] != 320:
                raise AssertionError("IK index build/query mismatch")
            cursor.execute(
                "SELECT id FROM docs WHERE MATCH(body) AGAINST('marker0319') "
                "ORDER BY id")
            if [row[0] for row in cursor.fetchall()] != [319]:
                raise AssertionError("BENG marker query mismatch")
            cursor.execute("UPDATE docs SET body='fresh marker9999' WHERE id=7")
            cursor.execute(
                "SELECT id FROM docs WHERE MATCH(body) AGAINST('marker9999') "
                "ORDER BY id")
            if [row[0] for row in cursor.fetchall()] != [7]:
                raise AssertionError("Updated document was not searchable")

        print(json.dumps({
            "status": "passed",
            "tokenize_cases": len(configurations),
            "repeat_calls": len(configurations) * 2,
            "concurrent_threads": len(threads),
        }, sort_keys=True))
    finally:
        try:
            if created:
                with handle.cursor() as cursor:
                    cursor.execute(f"DROP DATABASE `{database}`")
        finally:
            handle.close()


if __name__ == "__main__":
    main()

"""Offline tests of evidence guards, not native database acceptance."""
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import native_correctness as native
import official_benchmark as official


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.scorer = official.official_module()
        self.baseline = self.scorer.load_baseline(official.ROOT / "tools/benchmark/fts_large_bench_baseline.json")

    def report(self):
        values = {**official.CONFIG, **self.baseline["baseline_metrics"]}
        return "FTS Large Benchmark Report\n" + "\n".join(f"{key}: {value}" for key, value in values.items())

    def check_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-unit-fixture.txt"
            path.write_text(text, encoding="utf-8")
            return official.checked_report(path)

    def test_frozen_contract_and_all_fields(self):
        self.assertEqual(official.verify_contract(), official.OFFICIAL_HASHES)
        _, _, report = self.check_text(self.report())
        self.assertEqual(report["rows"], 20000)

    def test_appended_report_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.check_text(self.report() + "\n" + self.report())

    def test_missing_duplicate_and_invalid_duration_are_rejected(self):
        name = "tokenize_ik_avg_ms"
        for text in (
            "\n".join(line for line in self.report().splitlines() if not line.startswith(name + ":")),
            self.report() + f"\n{name}: 0.1",
            self.report().replace(f"{name}: {self.baseline['baseline_metrics'][name]}", f"{name}: -1"),
        ):
            with self.assertRaises(ValueError):
                self.check_text(text)

    def test_changed_parameters_are_rejected_by_actual_official_scorer(self):
        with self.assertRaises(SystemExit):
            self.check_text(self.report().replace("rows: 20000", "rows: 19999"))

    def test_real_official_score_keeps_negative_category_and_equal_weights(self):
        metrics = dict(self.baseline["baseline_metrics"])
        factors = {"build": 0.6, "tokenize": 1.5, "query": 0.8}
        for group, names in self.baseline["score_categories"].items():
            for name in names:
                metrics[name] *= factors[group]
        result = self.scorer.calculate_score(metrics, self.baseline)
        self.assertAlmostEqual(result["categories"]["tokenize"]["improvement"], -0.5)
        self.assertAlmostEqual(result["mean_improvement"], (0.4 - 0.5 + 0.2) / 3)
        self.assertAlmostEqual(result["score"], 100 * ((0.4 - 0.5 + 0.2) / 3) / 0.5)

    def test_real_official_score_caps_only_overall_result(self):
        for factor, expected in ((0.5, 100), (0.1, 100), (2, 0)):
            metrics = {key: value * factor for key, value in self.baseline["baseline_metrics"].items()}
            self.assertAlmostEqual(self.scorer.calculate_score(metrics, self.baseline)["score"], expected)

    def test_token_json_preserves_order_and_frequency(self):
        first = native.normalize_token_json('{"doc_len":4,"tokens":[{"caféx":3},{"other":1}]}', full_output=True)
        same = native.normalize_token_json('{"tokens":[{"caféx":3},{"other":1}],"doc_len":4}', full_output=True)
        self.assertEqual(first, same)
        self.assertNotEqual(first, {**first, "tokens": list(reversed(first["tokens"]))})
        changed = json.loads(json.dumps(first))
        changed["tokens"][0]["caféx"] = 2
        self.assertNotEqual(first, changed)

    def test_token_json_rejects_malformed_or_type_changed_native_results(self):
        self.assertEqual(native.normalize_token_json('["caféx","other"]'), ["caféx", "other"])
        self.assertEqual(native.normalize_token_json('{"doc_len":0,"tokens":[]}', full_output=True), {"doc_len": 0, "tokens": []})
        self.assertEqual(native.normalize_token_json('[]'), [])
        self.assertIsNone(native.normalize_token_json(None, allow_null=True))
        self.assertIsNone(native.normalize_token_json('null', full_output=True, allow_null=True))
        for value, full in (
            (None, False), ('null', True), ('{"doc_len":0,"tokens":null}', True),
            ('{"doc_len":true,"tokens":[]}', True), ('{"doc_len":1.0,"tokens":[]}', True),
            ('{"doc_len":-1,"tokens":[]}', True), ('{"doc_len":1,"tokens":[{"x":true}]}', True),
            ('{"doc_len":1,"tokens":[{"x":1.0}]}', True), ('{"doc_len":1,"tokens":[{"x":0}]}', True),
            ('{"doc_len":1,"tokens":[{"word":"x","freq":1}]}', True),
            ('{"doc_len":1,"tokens":[{"x":1,"y":1}]}', True), ('["x",1]', False),
        ):
            with self.subTest(value=value, full=full), self.assertRaises(ValueError):
                native.normalize_token_json(value, full_output=full)

    def test_match_full_identity_order_and_scores(self):
        self.assertTrue(native.equivalent_rows([[1, 0.3], [2, 0.2]], [[1, 0.3 + 1e-11], [2, 0.2]]))
        for rows in ([[2, 0.3], [1, 0.2]], [[1, 0.3]], [[1, 0.5], [2, 0.2]], [[1.0, 0.3], [2, 0.2]], [[1, math.nan], [2, 0.2]]):
            self.assertFalse(native.equivalent_rows([[1, 0.3], [2, 0.2]], rows))

    def test_fixtures_cover_collation_repeated_and_supplementary_text(self):
        rows = native.corpus()
        self.assertEqual(len(rows), 128)
        self.assertEqual(len({row[0] for row in rows}), 128)
        self.assertTrue(any("😀" in row[3] for row in rows))
        cases = native.token_cases()
        self.assertEqual(len(cases), 138)
        self.assertTrue(any(text == "caféx cafex caféx" and collation == "utf8mb4_general_ci" for _, _, text, collation in cases))
        self.assertEqual(len(native.query_cases()), 14)

    def test_comparison_fails_when_only_hit_count_matches(self):
        before = {"tokenize": [], "rows": {}, "non_relevance_queries": [], "queries": [{"phase": "built", "columns": "content", "query": "database", "limit": None, "rows": [[1, 0.5], [2, 0.4]]}]}
        after = json.loads(json.dumps(before))
        after["queries"][0]["rows"] = [[3, 0.5], [4, 0.4]]
        self.assertTrue(native.compare(before, after))

    def test_non_relevance_identity_and_both_count_shapes_are_consistent(self):
        class ResultCursor:
            def __init__(self, results):
                self.results = iter(results)
                self.commands = []

            def execute(self, sql, params):
                self.commands.append((sql, params))

            def fetchall(self):
                return next(self.results)

        ids = list(range(-10, 13))
        scored = [[identifier, 0.3] for identifier in reversed(ids)]
        cursor = ResultCursor([[[identifier] for identifier in ids], [[23]], [[20]]])
        self.assertEqual(native.non_relevance_result(cursor, "content", "database", scored), {"ids": ids, "count": 23, "limit20_count": 20})
        self.assertEqual(len(cursor.commands), 3)
        self.assertEqual(cursor.commands[0][0], "SELECT id FROM docs WHERE MATCH(content) AGAINST(%s) ORDER BY id")
        self.assertEqual(cursor.commands[1][0], "SELECT COUNT(*) FROM docs WHERE MATCH(content) AGAINST(%s)")
        self.assertEqual(cursor.commands[2][0], "SELECT COUNT(*) FROM (SELECT id FROM docs WHERE MATCH(content) AGAINST(%s) LIMIT 20) t")
        self.assertTrue(all(params == ("database",) for _, params in cursor.commands))
        self.assertEqual(native.non_relevance_result(ResultCursor([[], [[0]], [[0]]]), "content", "absent", []), {"ids": [], "count": 0, "limit20_count": 0})
        self.assertEqual(native.non_relevance_result(ResultCursor([[[1]], [[1]], [[1]]]), "content", "database", [[1, 0.3]]), {"ids": [1], "count": 1, "limit20_count": 1})
        invalid = (
            ([[[0], [1]]], [[2, 0.3], [3, 0.2]]),  # Same count, different IDs.
            ([[[1], [1]]], [[1, 0.3], [1, 0.2]]),  # Duplicate IDs.
            ([[[1.0]]], [[1, 0.3]]),
            ([[[1]], [[1.0]]], [[1, 0.3]]),
            ([[[1]], [[0]]], [[1, 0.3]]),
            ([[[1]], [[1]], [[0]]], [[1, 0.3]]),
            ([[[identifier] for identifier in ids], [[23]], [[23]]], scored),
        )
        for results, reference in invalid:
            with self.subTest(results=results), self.assertRaises(AssertionError):
                native.non_relevance_result(ResultCursor(results), "content", "database", reference)

    def test_non_relevance_before_after_comparison_does_not_accept_equal_counts(self):
        before = {"tokenize": [], "rows": {}, "queries": [], "non_relevance_queries": [{"phase": "built", "columns": "content", "query": "database", "ids": [1, 2], "count": 2, "limit20_count": 2}]}
        after = json.loads(json.dumps(before))
        self.assertEqual(native.compare(before, after), [])
        after["non_relevance_queries"][0]["ids"] = [3, 4]
        self.assertTrue(native.compare(before, after))

    def test_only_declared_null_argument_errors_are_tolerated(self):
        for text in (None, ""):
            for code in (1210, 1235):
                self.assertTrue(native.is_expected_tokenize_argument_error(text, code))
            for code in (0, 1045, 1064, 1146, 2003, 2013, 4012, 4013, 4016):
                self.assertFalse(native.is_expected_tokenize_argument_error(text, code))
        for code in (1210, 1235):
            self.assertFalse(native.is_expected_tokenize_argument_error("database", code))

    def test_native_error_handler_does_not_record_resource_failures_as_arguments(self):
        # Fault injection tests the actual native_side exception path. This is
        # an offline guard test, never native database/correctness evidence.
        class InjectedSqlError(Exception):
            pass

        class FaultCursor:
            def __init__(self, code):
                self.code = code

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                if sql.startswith("SELECT TOKENIZE("):
                    raise InjectedSqlError(self.code, "synthetic audit fault")

            def fetchone(self):
                return ("offline-error-handler-test",)

            def fetchall(self):
                return []

        for code in (4012, 4013, 1146):
            data = {}
            handle = types.SimpleNamespace(cursor=lambda: FaultCursor(code))
            with self.subTest(code=code), mock.patch.dict("sys.modules", {"pymysql": types.SimpleNamespace(MySQLError=InjectedSqlError)}):
                with self.assertRaises(InjectedSqlError):
                    native.native_side(handle, "fts_tuning_verify_0123456789abcdef", True, data)
            self.assertEqual(data["tokenize"], [], "Unexpected error was accepted as a completed NULL/empty-input case")


if __name__ == "__main__":
    unittest.main()

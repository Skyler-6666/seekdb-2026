"""Offline plan-query extraction guards; these do not connect to a server."""
import unittest

import capture_native_plans as plans


class PlanQueryTests(unittest.TestCase):
    def test_extracts_exact_unchanged_official_queries(self):
        plans.verify_contract()
        source = (plans.ROOT / 'tools/benchmark/fts_large_bench.sh').read_text(encoding='utf-8')
        queries = plans.official_queries(source)
        self.assertEqual(len(queries), 4)
        self.assertIn('LIMIT 20', queries[-1][1])
        self.assertTrue(all('MATCH(' in sql for _, sql in queries))

    def test_missing_duplicate_or_changed_shape_is_rejected(self):
        source = (plans.ROOT / 'tools/benchmark/fts_large_bench.sh').read_text(encoding='utf-8')
        for altered in (
            source.replace('run_query_bench "query_cn"', 'run_query_bench "unexpected"'),
            source + '\n' + source,
            source.replace('SELECT COUNT(*) FROM fts_large_bench.docs WHERE MATCH(title, content)',
                           'DELETE FROM fts_large_bench.docs WHERE MATCH(title, content)'),
        ):
            with self.assertRaises(ValueError):
                plans.official_queries(altered)


if __name__ == '__main__':
    unittest.main()

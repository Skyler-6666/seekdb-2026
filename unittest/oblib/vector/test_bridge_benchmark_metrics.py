import unittest
from benchmark_cuvs_bridge import recall_at_k


class RecallTests(unittest.TestCase):
    def test_exact_and_partial_recall(self):
        self.assertEqual(1.0, recall_at_k([[2, 1], [4, 3]], [[1, 2, 0], [3, 4, 5]], 6, 2))
        self.assertEqual(0.75, recall_at_k([[0, 1], [4, 3]], [[1, 2, 0], [3, 4, 5]], 6, 2))

    def test_duplicate_or_missing_ids_are_not_counted_as_success(self):
        for result in ([[1, 1]], [[1]], [[]]):
            with self.assertRaises(ValueError):
                recall_at_k(result, [[1, 2]], 5, 2)

    def test_out_of_range_and_query_count(self):
        for result in ([[1, 5]], [[-1, 2]], [], [[1, 2], [1, 2]]):
            with self.assertRaises(ValueError):
                recall_at_k(result, [[1, 2]], 5, 2)


if __name__ == "__main__":
    unittest.main()

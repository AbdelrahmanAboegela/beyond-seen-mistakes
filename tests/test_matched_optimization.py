import unittest

import numpy as np

from matched_composition import optimize_replacement


class OptimizedReplacementTest(unittest.TestCase):
    def test_is_deterministic_and_minimax_balanced(self):
        target = np.array([1, 1, 0])
        labels = np.array([
            [1, 0, 0], [0, 1, 0], [1, 1, 1],
            [0, 0, 0], [1, 0, 1], [0, 1, 1],
        ])
        indices = np.arange(len(labels))
        first, audit = optimize_replacement(labels, indices, target, 3)
        second, _ = optimize_replacement(labels, indices, target, 3)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(first), 3)
        self.assertTrue(audit["success"])
        self.assertEqual(audit["max_positive_count_difference"], 1)
        self.assertEqual(audit["l1_positive_count_difference"], 3)


if __name__ == "__main__":
    unittest.main()

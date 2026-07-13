import importlib.util
from pathlib import Path
import unittest

import numpy as np


RUNNER = Path(__file__).resolve().parents[2] / "o6_dimension_runner.py"
SPEC = importlib.util.spec_from_file_location("o6_dimension_runner", RUNNER)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class FeedbackPolicyTest(unittest.TestCase):
    def test_rank_correlation(self) -> None:
        self.assertAlmostEqual(
            runner.rank_correlation(
                np.array([1, 2, 3]),
                np.array([10, 20, 30]),
            ),
            1.0,
        )
        self.assertAlmostEqual(
            runner.rank_correlation(
                np.array([1, 2, 3]),
                np.array([30, 20, 10]),
            ),
            -1.0,
        )

    def test_increase_requires_improvement(self) -> None:
        scores = {9: 0.10, 15: 0.50, 20: 0.49, 30: 0.20}
        self.assertEqual(
            runner.choose_feedback_depth(scores, 9, 200, 0),
            (15, True),
        )

    def test_dwell_time_blocks_switch(self) -> None:
        scores = {9: 0.10, 15: 0.50, 20: 0.49, 30: 0.20}
        self.assertEqual(
            runner.choose_feedback_depth(scores, 9, 300, 200),
            (9, False),
        )

    def test_prefers_simpler_near_best_depth(self) -> None:
        scores = {9: 0.48, 15: 0.50, 20: 0.30, 30: 0.20}
        self.assertEqual(
            runner.choose_feedback_depth(scores, 15, 400, 200),
            (9, True),
        )


if __name__ == "__main__":
    unittest.main()

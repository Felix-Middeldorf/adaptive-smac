import unittest

from o1_selection_rotating_runner import (
    CANDIDATE_DEPTHS,
    SelectionRotatingCallback,
    rank_depths,
)


class FakeRunHistory:
    def __init__(self) -> None:
        self.n = 10
        self.best = 100.0

    def __len__(self) -> int:
        return self.n

    def get_configs(self) -> list[int]:
        return [0]

    def get_cost(self, config: int) -> float:
        return self.best


class FakeSelector:
    def __init__(self, runhistory: FakeRunHistory) -> None:
        self._runhistory = runhistory
        self._model = type("Model", (), {"_rf_opts": {"max_depth": 5}})()


class FakeSmbo:
    def __init__(self, runhistory: FakeRunHistory) -> None:
        self.runhistory = runhistory


class RankDepthsTest(unittest.TestCase):
    def test_uses_cumulative_improvement(self) -> None:
        improvements = {
            5: [0.0, 1.0],
            10: [0.6, 0.6],
            15: [0.0, 0.0],
            20: [0.4, 0.4],
        }
        self.assertEqual(
            rank_depths(CANDIDATE_DEPTHS, improvements),
            [10, 5, 20, 15],
        )

    def test_breaks_ties_by_smaller_depth(self) -> None:
        improvements = {depth: [0.0] for depth in CANDIDATE_DEPTHS}
        self.assertEqual(
            rank_depths(CANDIDATE_DEPTHS, improvements),
            list(CANDIDATE_DEPTHS),
        )


class SelectionCallbackTest(unittest.TestCase):
    def test_rotates_selects_two_then_selects_one(self) -> None:
        runhistory = FakeRunHistory()
        selector = FakeSelector(runhistory)
        smbo = FakeSmbo(runhistory)
        callback = SelectionRotatingCallback()

        exploration_depths = []
        for trial in range(10, 250, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)
            exploration_depths.append(selector._model._rf_opts["max_depth"])
        self.assertEqual(exploration_depths[:8], [5, 10, 15, 20] * 2)

        # Only depth 20 improves in the final scored exploration segment.
        runhistory.n = 250
        runhistory.best = 98.0
        callback.on_tell_end(smbo, None, None)
        callback.on_next_configurations_start(selector)
        self.assertEqual(selector._model._rf_opts["max_depth"], 20)

        for trial in range(260, 300, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)
        runhistory.n = 300
        callback.on_tell_end(smbo, None, None)
        callback.on_next_configurations_start(selector)
        self.assertEqual(callback.selected_two, (20, 5))

        top_two_depths = [selector._model._rf_opts["max_depth"]]
        for trial in range(310, 500, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)
            top_two_depths.append(selector._model._rf_opts["max_depth"])
        self.assertEqual(top_two_depths[:6], [20, 5] * 3)

        # The last top-two segment uses depth 5, so it wins phase two.
        runhistory.n = 500
        runhistory.best = 97.0
        callback.on_tell_end(smbo, None, None)
        callback.on_next_configurations_start(selector)
        self.assertEqual(callback.selected_depth, 5)
        self.assertEqual(selector._model._rf_opts["max_depth"], 5)


if __name__ == "__main__":
    unittest.main()

import unittest

from o1_selection_rotating_force20_runner import (
    SelectionRotatingCallback,
    policy_spec,
)
from submit_selection_rotating_force20 import (
    BENCHMARK_SEEDS,
    SMAC_SEEDS,
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


class ForceDepth20PolicyTest(unittest.TestCase):
    def test_policy_metadata_and_job_matrix(self) -> None:
        spec = policy_spec()
        self.assertEqual(
            spec["selection_delay_policy"],
            "continue_exploration_rotation_unscored",
        )
        self.assertEqual(
            spec["final_depth_policy"],
            "force_depth_20_regardless_of_second_ranking",
        )
        self.assertEqual(len(BENCHMARK_SEEDS) * len(SMAC_SEEDS), 70)

    def test_rotates_during_delay_and_forces_depth_20(self) -> None:
        runhistory = FakeRunHistory()
        selector = FakeSelector(runhistory)
        smbo = FakeSmbo(runhistory)
        callback = SelectionRotatingCallback()

        for trial in range(10, 250, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)

        # Only depth 20 improves in the scored 0--250 exploration phase.
        runhistory.n = 250
        runhistory.best = 98.0
        callback.on_tell_end(smbo, None, None)

        delay_depths = []
        for trial in range(250, 300, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)
            delay_depths.append(selector._model._rf_opts["max_depth"])
        self.assertEqual(delay_depths, [5, 10, 15, 20, 5])

        runhistory.n = 300
        callback.on_tell_end(smbo, None, None)
        callback.on_next_configurations_start(selector)
        self.assertEqual(callback.selected_two, (20, 5))

        # Make depth 5 win the scored 300--500 phase.
        for trial in range(310, 500, 10):
            runhistory.n = trial
            callback.on_next_configurations_start(selector)
        runhistory.n = 500
        runhistory.best = 97.0
        callback.on_tell_end(smbo, None, None)
        self.assertEqual(callback.second_ranking[0], 5)

        callback.on_next_configurations_start(selector)
        self.assertEqual(callback.selected_depth, 20)
        self.assertEqual(selector._model._rf_opts["max_depth"], 20)


if __name__ == "__main__":
    unittest.main()

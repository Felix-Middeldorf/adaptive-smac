#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Fast checks that do not load the large ACLib surrogate."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
EXPECTED_LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"
sys.path.insert(0, str(EXPECTED_LOCAL_SMAC_ROOT))

from ConfigSpace import ConfigurationSpace
from smac import AlgorithmConfigurationFacade, Scenario
from smac.main.config_selector import ConfigSelector

from run_compare import (
    LOCAL_SMAC_ROOT,
    MODES,
    N_INSTANCES,
    N_TRIALS,
    OriginalSingletonConfigSelector,
    SMAC_SEEDS,
)
from submit_experiment import experiment_jobs


class RecordingModel:
    def __init__(self):
        self.calls = []

    def predict_marginalized(self, X):
        self.calls.append(X.copy())
        means = X[:, :1]
        return means, np.ones_like(means)


def _check_selector(selector: ConfigSelector, expected_calls: int) -> None:
    X = np.array([[0.3, 1.0], [0.1, 2.0], [0.1, 3.0]])
    model = RecordingModel()
    selector._model = model
    x_best, cost = selector._get_x_best(X)
    assert len(model.calls) == expected_calls
    np.testing.assert_array_equal(x_best, X[1])
    assert cost == 0.1


def main() -> None:
    assert LOCAL_SMAC_ROOT.resolve() == EXPECTED_LOCAL_SMAC_ROOT.resolve()
    selector_source = Path(inspect.getfile(ConfigSelector)).resolve()
    assert LOCAL_SMAC_ROOT.resolve() in selector_source.parents
    assert N_TRIALS == 3_000
    assert N_INSTANCES == 150
    assert SMAC_SEEDS == (0, 1)
    assert MODES == ("original_singleton", "fixed_batched")

    scenario = Scenario(ConfigurationSpace())
    _check_selector(
        AlgorithmConfigurationFacade.get_config_selector(scenario),
        expected_calls=1,
    )
    _check_selector(
        OriginalSingletonConfigSelector(scenario),
        expected_calls=3,
    )

    jobs = experiment_jobs()
    assert len(jobs) == 4
    assert len(set(jobs)) == len(jobs)
    print(f"PASS: local SMAC imported from {selector_source}")
    print("PASS: batched selector makes one call; reference makes one per row.")
    print("PASS: 4 unique paired comparison jobs.")


if __name__ == "__main__":
    main()

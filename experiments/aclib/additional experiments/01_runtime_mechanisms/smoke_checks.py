#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Static and asset-level checks for the runtime profiling experiment."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from smac.model.random_forest.random_forest import RandomForest

from profile_runtime import (
    BENCHMARKS,
    DEFAULT_DEPTH,
    LOCAL_SMAC_ROOT,
    MAIN_EXPERIMENTS,
    N_TRIALS,
    VARIANTS,
    load_benchmark_data,
)
from submit_jobs import (
    SLURM_ACCOUNT,
    jobs_for_suite,
    slurm_parameters,
)


def main() -> None:
    assert N_TRIALS == 256
    assert DEFAULT_DEPTH == 10
    assert len(BENCHMARKS) == 5
    assert len(VARIANTS) == 8
    assert len(jobs_for_suite("baseline")) == 15
    assert len(jobs_for_suite("mechanisms")) == 35
    assert len(jobs_for_suite("depth")) == 30
    assert len(jobs_for_suite("all")) == 80
    assert SLURM_ACCOUNT == "lect0190"
    assert slurm_parameters("all")["slurm_account"] == "lect0190"
    assert (
        slurm_parameters("all")["slurm_setup"][0]
        == "export PYTHONHASHSEED=0"
    )

    source = Path(inspect.getfile(RandomForest)).resolve()
    assert LOCAL_SMAC_ROOT.resolve() in source.parents, source
    for key, experiment in MAIN_EXPERIMENTS.items():
        assert (experiment / "initial_config.json").is_file(), key
        data = load_benchmark_data(key)
        spec = BENCHMARKS[key]
        assert len(data.configspace) == spec.expected_hyperparameters
        assert len(data.training_instances) == spec.expected_training_instances
        assert len(data.test_instances) == spec.expected_test_instances
        assert not set(data.training_instances) & set(data.test_instances)
        assert len(next(iter(data.features.values()))) == spec.expected_features
        print(
            f"PASS {key}: {len(data.configspace)} hyperparameters, "
            f"{len(data.training_instances)} training instances, "
            f"{spec.expected_features} features."
        )

    print("PASS job suites: baseline=15, mechanisms=35, depth=30, all=80.")
    print("PASS local SMAC, project lect0190, and PYTHONHASHSEED=0 setup.")
    print(f"Worker PYTHONHASHSEED currently: {os.environ.get('PYTHONHASHSEED')!r}")


if __name__ == "__main__":
    main()

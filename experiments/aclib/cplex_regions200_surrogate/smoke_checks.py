"""Opt-in checks; run through run_in_env.sh, not repository-wide pytest."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from aclib_benchmark import (
    EXPECTED_CONDITIONS,
    EXPECTED_FEATURES,
    EXPECTED_HYPERPARAMETERS,
    EXPECTED_INSTANCES_PER_SPLIT,
    TIMEOUT_COST,
    CplexRegions200Benchmark,
    load_benchmark_data,
    par10_cost,
    required_assets,
)
from run_smac import _acquire_run_lock, run_experiment
from submit_experiment import MAX_SUBMITTED_JOBS, experiment_jobs


def test_assets_and_benchmark_layout() -> None:
    assert all(path.is_file() for path in required_assets())
    data = load_benchmark_data()
    assert len(data.configspace) == EXPECTED_HYPERPARAMETERS
    assert len(data.configspace.conditions) == EXPECTED_CONDITIONS
    assert len(data.training_instances) == EXPECTED_INSTANCES_PER_SPLIT
    assert len(data.test_instances) == EXPECTED_INSTANCES_PER_SPLIT
    assert not set(data.training_instances) & set(data.test_instances)
    assert all(len(data.features[name]) == EXPECTED_FEATURES for name in data.features)


def test_par10_timeout_mapping() -> None:
    assert par10_cost(1.25, "TRUE") == 1.25
    assert par10_cost(10_000.0, "CUTOFF") == TIMEOUT_COST == 100_000.0


def test_submit_matrix() -> None:
    seeds = tuple(range(7))
    jobs = experiment_jobs(
        smac_seeds=seeds,
        n_trials=100,
        n_instances=20,
        output_root=Path("/tmp/aclib-test-output"),
        validate_test=False,
    )
    assert len(jobs) == len(seeds)
    assert len(set(jobs)) == len(jobs)
    assert len(jobs) <= MAX_SUBMITTED_JOBS


def test_exclusive_run_lock() -> None:
    with tempfile.TemporaryDirectory(prefix="aclib-lock-test-") as directory:
        path = Path(directory) / "run"
        first = _acquire_run_lock(path)
        try:
            try:
                _acquire_run_lock(path)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Duplicate lock acquisition unexpectedly succeeded.")
        finally:
            first.close()
        second = _acquire_run_lock(path)
        second.close()


def run_model_integration_test() -> None:
    data = load_benchmark_data()
    benchmark = CplexRegions200Benchmark()
    config = data.configspace.get_default_configuration()
    instance = data.training_instances[0]
    first_cost, first_info = benchmark.evaluate(config, instance)
    second_cost, second_info = benchmark.evaluate(config, instance)
    assert abs(first_cost - 1.92172) < 1e-4, first_cost
    assert first_cost == second_cost
    assert first_info["surrogate_status"] == second_info["surrogate_status"] == "TRUE"
    print(f"Default-config golden prediction: {first_cost:.8f}")


def run_smac_integration_test() -> None:
    with tempfile.TemporaryDirectory(prefix="aclib-smac-smoke-") as directory:
        summary = run_experiment(
            smac_seed=0,
            n_trials=10,
            n_instances=3,
            output_root=Path(directory),
            validate_test=False,
            overwrite=False,
        )
        assert summary["finished_trials"] == 10
        output = Path(summary["output_directory"])
        assert (output / "completed.json").is_file()
        assert (output / "runhistory.json").is_file()
        assert summary["incumbent_cost"] >= 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="store_true",
        help="Also load the 1 GB PyRFR model and check a golden prediction.",
    )
    parser.add_argument(
        "--smac",
        action="store_true",
        help="Also perform a disposable ten-trial end-to-end SMAC run.",
    )
    args = parser.parse_args()

    static_tests = (
        test_assets_and_benchmark_layout,
        test_exclusive_run_lock,
        test_par10_timeout_mapping,
        test_submit_matrix,
    )
    for test in static_tests:
        test()
        print(f"PASS {test.__name__}")
    if args.model:
        run_model_integration_test()
        print("PASS run_model_integration_test")
    if args.smac:
        run_smac_integration_test()
        print("PASS run_smac_integration_test")
    print("All requested checks passed.")


if __name__ == "__main__":
    main()

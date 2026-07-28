from __future__ import annotations

from o1_3500_uncertainty_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    DIMENSION,
    N_INSTANCES,
    N_TRIALS,
    OUTPUT_DIRECTORY,
    SMAC_SEEDS,
    SOURCE_RUNNER,
    _SHARED,
    make_instance_map,
)
from submit_uncertainty import (
    MAX_SUBMITTED_JOBS,
    RUNS_PER_JOB,
    experiment_packs,
    experiment_runs,
)


def test_requested_experiment_matrix() -> None:
    assert DEPTHS == (5, 10, 15, 20, 25, 30)
    assert BENCHMARK_SEEDS == (40, 42)
    assert SMAC_SEEDS == tuple(range(10))
    assert DIMENSION == 15
    assert N_INSTANCES == 15
    assert N_TRIALS == 3_500


def test_all_runs_are_packed_once_under_job_limit() -> None:
    runs = experiment_runs()
    packs = experiment_packs()
    assert len(runs) == 6 * 2 * 10 == 120
    assert len(set(runs)) == len(runs)
    assert RUNS_PER_JOB == 2
    assert len(packs) == 60
    assert len(packs) <= MAX_SUBMITTED_JOBS == 80
    assert tuple(run for pack in packs for run in pack.runs) == runs


def test_shared_runner_is_configured_for_this_output() -> None:
    assert SOURCE_RUNNER.is_file()
    assert _SHARED.DEPTHS == DEPTHS
    assert _SHARED.BENCHMARK_SEEDS == BENCHMARK_SEEDS
    assert _SHARED.SMAC_SEEDS == SMAC_SEEDS
    assert _SHARED.DIMENSION == DIMENSION
    assert _SHARED.N_INSTANCES == N_INSTANCES
    assert _SHARED.N_TRIALS == N_TRIALS
    assert _SHARED.OUTPUT_DIRECTORY == OUTPUT_DIRECTORY


def test_instance_map_is_shared_and_has_fifteen_instances() -> None:
    first = make_instance_map()
    second = make_instance_map()
    assert first == second
    assert list(first) == [f"i{index}" for index in range(N_INSTANCES)]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")

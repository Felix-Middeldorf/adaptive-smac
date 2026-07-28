from __future__ import annotations

from o1_5000_fixed_depth_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    DIMENSION,
    N_INSTANCES,
    N_TRIALS,
    SMAC_SEEDS,
    make_instance_map,
)
from submit_fixed_depths import MAX_SUBMITTED_JOBS, experiment_jobs


def test_requested_matrix() -> None:
    assert DEPTHS == (5, 10, 15, 20, 30, 40, 50)
    assert BENCHMARK_SEEDS == (40,)
    assert SMAC_SEEDS == tuple(range(5))
    assert DIMENSION == 10
    assert N_INSTANCES == 10
    assert N_TRIALS == 5_000


def test_job_matrix_is_complete_and_unique() -> None:
    jobs = experiment_jobs()
    assert len(jobs) == 7 * 1 * 5 == 35
    assert len(set(jobs)) == len(jobs)
    assert len(jobs) <= MAX_SUBMITTED_JOBS
    assert {
        (job.benchmark_seed, job.smac_seed, job.depth)
        for job in jobs
    } == {
        (40, smac_seed, depth)
        for smac_seed in SMAC_SEEDS
        for depth in DEPTHS
    }


def test_instance_map_is_reused_and_has_ten_instances() -> None:
    first = make_instance_map()
    second = make_instance_map()
    assert first == second
    assert list(first) == [f"i{index}" for index in range(10)]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")

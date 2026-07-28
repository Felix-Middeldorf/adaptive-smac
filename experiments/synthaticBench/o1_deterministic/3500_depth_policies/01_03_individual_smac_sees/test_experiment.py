from __future__ import annotations

from o1_7000_fixed_depth_runner import (
    BENCHMARK_SEEDS,
    DIMENSION,
    FIXED_DEPTHS,
    N_INSTANCES,
    N_TRIALS,
    SMAC_SEEDS,
    make_instance_map,
    policy_name,
    policy_spec,
)
from submit_experiment import MAX_SUBMITTED_JOBS, experiment_jobs


def test_requested_experiment_matrix() -> None:
    assert BENCHMARK_SEEDS == (40,)
    assert SMAC_SEEDS == tuple(range(7))
    assert FIXED_DEPTHS == (5, 10, 15, 20, 25, 30, 40, 50)
    assert DIMENSION == 15
    assert N_INSTANCES == 20
    assert N_TRIALS == 7_000


def test_one_run_per_job_and_job_limit() -> None:
    jobs = experiment_jobs()
    expected = len(FIXED_DEPTHS) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    assert len(jobs) == expected == 56
    assert len(set(jobs)) == expected
    assert len(jobs) <= MAX_SUBMITTED_JOBS == 70
    assert {
        (job.benchmark_seed, job.smac_seed, job.depth)
        for job in jobs
    } == {
        (40, smac_seed, depth)
        for smac_seed in SMAC_SEEDS
        for depth in FIXED_DEPTHS
    }


def test_instance_map_and_policy_metadata() -> None:
    first = make_instance_map()
    second = make_instance_map()
    assert first == second
    assert len(first) == N_INSTANCES
    assert list(first) == [f"i{index}" for index in range(N_INSTANCES)]
    for depth in FIXED_DEPTHS:
        assert policy_name(depth) == f"fixed_depth_{depth}"
        assert policy_spec(depth)["fixed_depth"] == depth


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")

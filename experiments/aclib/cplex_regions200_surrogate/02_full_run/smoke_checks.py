#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Fast, non-submitting validation of the requested experiment matrix."""

from run_fixed_depth import (
    DEPTHS,
    N_INSTANCES,
    N_TRIALS,
    PCA_COMPONENTS,
    SMAC_SEEDS,
)
from submit_experiment import MAX_SUBMITTED_JOBS, experiment_jobs


def main() -> None:
    assert DEPTHS == (5, 10, 15, 20, 25, 30, 40)
    assert SMAC_SEEDS == tuple(range(5))
    assert N_INSTANCES == 150
    assert N_TRIALS == 10_000
    assert PCA_COMPONENTS is None
    jobs = experiment_jobs()
    assert len(jobs) == 35
    assert len(set(jobs)) == 35
    assert len(jobs) <= MAX_SUBMITTED_JOBS
    assert {(job.depth, job.smac_seed) for job in jobs} == {
        (depth, seed) for depth in DEPTHS for seed in SMAC_SEEDS
    }
    print("PASS: requested 35-run matrix is complete and unique.")


if __name__ == "__main__":
    main()

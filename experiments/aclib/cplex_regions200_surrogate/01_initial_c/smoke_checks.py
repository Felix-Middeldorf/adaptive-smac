#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Small opt-in checks; this filename intentionally avoids pytest discovery."""

from run_fixed_depth import DEPTHS, N_INSTANCES, N_TRIALS, SMAC_SEEDS
from submit_experiment import MAX_SUBMITTED_JOBS, experiment_jobs


def main() -> None:
    assert DEPTHS == (5, 10, 20, 30)
    assert SMAC_SEEDS == tuple(range(5))
    assert N_INSTANCES == 100
    assert N_TRIALS == 1_000
    jobs = experiment_jobs()
    assert len(jobs) == 20
    assert len(set(jobs)) == 20
    assert len(jobs) <= MAX_SUBMITTED_JOBS
    assert {(job.depth, job.smac_seed) for job in jobs} == {
        (depth, seed) for depth in DEPTHS for seed in SMAC_SEEDS
    }
    print("PASS: requested 20-run matrix is complete and unique.")


if __name__ == "__main__":
    main()

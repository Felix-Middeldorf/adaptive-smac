#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Fast checks that do not load the large ACLib surrogate."""

from profile_runtime import DEPTH, N_INSTANCES, N_TRIALS, SMAC_SEEDS, VARIANTS
from submit_experiment import experiment_jobs


def main() -> None:
    assert DEPTH == 20
    assert N_INSTANCES == 150
    assert N_TRIALS == 1_000
    assert SMAC_SEEDS == (0, 1)
    assert tuple(VARIANTS) == (
        "baseline",
        "no_instance_features",
        "challengers_500",
        "local_search_1",
        "random_search_only",
        "retrain_after_64",
        "no_periodic_save",
    )
    jobs = experiment_jobs()
    assert len(jobs) == len(VARIANTS) * len(SMAC_SEEDS) == 14
    assert len(set(jobs)) == len(jobs)
    print("PASS: 14 unique runtime-profiling jobs.")


if __name__ == "__main__":
    main()

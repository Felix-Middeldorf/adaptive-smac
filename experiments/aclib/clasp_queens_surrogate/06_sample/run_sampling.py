#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Run the Clasp Queens random-configuration sample."""

import argparse

from experiment import DEFINITION
from random_sampling_experiment import (
    CONFIGSPACE_SEED,
    N_CONFIGURATIONS,
    QUANTILE_SEEDS,
    run_random_sampling,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-configurations", type=int, default=N_CONFIGURATIONS)
    parser.add_argument("--configspace-seed", type=int, default=CONFIGSPACE_SEED)
    parser.add_argument(
        "--quantile-seeds",
        type=int,
        nargs="+",
        default=QUANTILE_SEEDS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_random_sampling(
        DEFINITION,
        n_configurations=args.n_configurations,
        configspace_seed=args.configspace_seed,
        quantile_seeds=tuple(args.quantile_seeds),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()


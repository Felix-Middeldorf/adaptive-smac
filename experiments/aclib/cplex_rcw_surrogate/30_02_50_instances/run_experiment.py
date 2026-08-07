#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Run one CPLEX RCW 50-instance depth/seed/PCA combination."""

import argparse

from experiment import DEFINITION
from raw_smac_experiment import DEPTHS, N_TRIALS, SMAC_SEEDS, run_raw_smac


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, choices=DEPTHS, required=True)
    parser.add_argument("--smac-seed", type=int, choices=SMAC_SEEDS, required=True)
    parser.add_argument("--pca-components", type=int, choices=(0, 4), required=True)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_raw_smac(
        definition=DEFINITION,
        depth=args.depth,
        smac_seed=args.smac_seed,
        pca_components=(
            None if args.pca_components == 0 else args.pca_components
        ),
        n_trials=args.n_trials,
        overwrite=args.overwrite,
    )

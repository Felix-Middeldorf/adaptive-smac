#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Run one SMAC seed of the Codex-controlled RF experiment."""

import argparse
from pathlib import Path

from experiment import DEFINITION
from llm_policy import N_TRIALS, SMAC_SEEDS, run_llm_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smac-seed", type=int, choices=SMAC_SEEDS, required=True)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_llm_policy(
        definition=DEFINITION,
        smac_seed=args.smac_seed,
        n_trials=args.n_trials,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

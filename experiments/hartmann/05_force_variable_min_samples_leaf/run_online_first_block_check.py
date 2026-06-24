from __future__ import annotations

import json
from pathlib import Path

from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from master_utils.benchmarks import hartmann


SEED = 0
N_TRIALS = 100
N_INITIAL_DESIGN_CONFIGS = 25

ROOT = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/05_force_variable_min_samples_leaf"
)
BRANCH_SELECT_DIRECTORY = ROOT / "hartmann_6d_branch_select_min_samples_leaf"
OUTPUT_DIRECTORY = ROOT / "hartmann_6d_online_first_block_check"


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


def selected_first_leaf_for_seed(seed: int) -> int:
    summary_path = BRANCH_SELECT_DIRECTORY / f"seed_{seed}" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing offline policy summary for seed {seed}: {summary_path}. "
            "Run run_branch_select_min_samples_leaf.py first."
        )

    with open(summary_path) as fh:
        summary = json.load(fh)

    return int(summary["selected_leaf_values"][0])


def run_online_first_block(seed: int = SEED):
    min_samples_leaf = selected_first_leaf_for_seed(seed)

    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])

    scenario = Scenario(
        name=f"online_first_block_leaf_{min_samples_leaf}",
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model = HPOFacade.get_model(
        scenario=scenario,
        min_samples_leaf=min_samples_leaf,
    )
    initial_design = HPOFacade.get_initial_design(
        scenario=scenario,
        n_configs=N_INITIAL_DESIGN_CONFIGS,
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        model=model,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = smac.runhistory.get_cost(incumbent)

    print(
        f"seed={seed}, min_samples_leaf={min_samples_leaf}, "
        f"n_trials={N_TRIALS}, incumbent_cost={incumbent_cost}"
    )
    print(incumbent)
    return incumbent


if __name__ == "__main__":
    run_online_first_block(SEED)

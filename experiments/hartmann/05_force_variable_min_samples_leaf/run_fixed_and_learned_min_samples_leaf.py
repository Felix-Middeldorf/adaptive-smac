from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback import Callback

from master_utils.benchmarks import hartmann


MIN_SAMPLES_LEAF_VALUES = [1, 2, 3, 4, 5]
SEEDS = range(5)
N_TRIALS = 500
TRIALS_PER_BLOCK = 100
N_INITIAL_DESIGN_CONFIGS = 25

BRANCH_SELECT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/05_force_variable_min_samples_leaf/"
    "hartmann_6d_branch_select_min_samples_leaf"
)
OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/05_force_variable_min_samples_leaf/"
    "hartmann_6d_fixed_vs_learned_min_samples_leaf"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


class LearnedBlockMinSamplesLeafCallback(Callback):
    """Apply the per-block min_samples_leaf sequence discovered offline for this seed."""

    def __init__(self, selected_leaf_values: List[int]) -> None:
        super().__init__()
        if len(selected_leaf_values) != 5:
            raise ValueError(f"Expected 5 selected leaf_values, got {selected_leaf_values!r}")
        self._selected_leaf_values = selected_leaf_values
        self._last_min_samples_leaf = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        block_idx = min(n_trials // TRIALS_PER_BLOCK, len(self._selected_leaf_values) - 1)
        min_samples_leaf = self._selected_leaf_values[block_idx]
        config_selector._model._rf_opts["min_samples_leaf"] = min_samples_leaf

        if min_samples_leaf != self._last_min_samples_leaf:
            print(
                f"[LearnedBlockMinSamplesLeafCallback] trials={n_trials}, "
                f"block={block_idx + 1}, min_samples_leaf={min_samples_leaf}"
            )
            self._last_min_samples_leaf = min_samples_leaf


def selected_leaf_values_for_seed(seed: int) -> List[int]:
    summary_path = BRANCH_SELECT_DIRECTORY / f"seed_{seed}" / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing offline policy summary for seed {seed}: {summary_path}. "
            "Run run_branch_select_min_samples_leaf.py first."
        )

    with open(summary_path) as fh:
        return [int(min_samples_leaf) for min_samples_leaf in json.load(fh)["selected_leaf_values"]]


def policy_components(
    scenario: Scenario,
    policy: str,
    seed: int,
) -> Tuple[Optional[object], List[Callback]]:
    if policy.startswith("leaf_"):
        min_samples_leaf = int(policy.removeprefix("leaf_"))
        return HPOFacade.get_model(scenario=scenario, min_samples_leaf=min_samples_leaf), []

    if policy == "learned_branch_select_policy":
        selected_leaf_values = selected_leaf_values_for_seed(seed)
        model = HPOFacade.get_model(scenario=scenario, min_samples_leaf=selected_leaf_values[0])
        return model, [LearnedBlockMinSamplesLeafCallback(selected_leaf_values)]

    raise ValueError(f"Unknown policy: {policy!r}")


def run_smac(seed: int, policy: str):
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])

    scenario = Scenario(
        name=policy,
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model, callbacks = policy_components(scenario, policy, seed)
    initial_design = HPOFacade.get_initial_design(
        scenario=scenario,
        n_configs=N_INITIAL_DESIGN_CONFIGS,
    )
    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        model=model,
        initial_design=initial_design,
        callbacks=callbacks,
        overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = smac.runhistory.get_cost(incumbent)

    print(f"policy={policy}, seed={seed}, incumbent_cost={incumbent_cost}")
    print(incumbent)
    return incumbent


def submit_jobs() -> None:
    policies = [f"leaf_{min_samples_leaf}" for min_samples_leaf in MIN_SAMPLES_LEAF_VALUES] + ["learned_branch_select_policy"]

    executor = submitit.AutoExecutor(
        folder="logs_fixed_vs_learned",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_array_parallelism=40,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannLeafFixedVsLearned",
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for policy in policies:
            for seed in SEEDS:
                job = executor.submit(run_smac, seed, policy)
                jobs.append((policy, seed, job))

    print("submitted_jobs:")
    for policy, seed, job in jobs:
        print(f"{policy}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

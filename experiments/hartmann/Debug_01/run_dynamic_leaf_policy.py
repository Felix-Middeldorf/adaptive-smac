from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback import Callback

from master_utils.benchmarks import hartmann


POLICY = (1, 2, 3, 4)
SEED = 0
N_TRIALS = 400
TRIALS_PER_BLOCK = 100
N_INITIAL_DESIGN_CONFIGS = 25
SLURM_PARTITION = "c23ms"

OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/Debug/"
    "debug_dynamic_leaf_policy_1_2_3_4"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


class DebugLeafPolicyCallback(Callback):
    """Switch min_samples_leaf every 100 trials: 1 -> 2 -> 3 -> 4."""

    def __init__(self, policy: Tuple[int, ...]) -> None:
        super().__init__()
        self._policy = policy
        self._last_leaf = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        block_idx = min(n_trials // TRIALS_PER_BLOCK, len(self._policy) - 1)
        min_samples_leaf = self._policy[block_idx]
        config_selector._model._rf_opts["min_samples_leaf"] = min_samples_leaf

        if min_samples_leaf != self._last_leaf:
            print(
                f"[DebugLeafPolicyCallback] trials={n_trials}, "
                f"block={block_idx + 1}, min_samples_leaf={min_samples_leaf}"
            )
            self._last_leaf = min_samples_leaf


def make_configspace(seed: int) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])
    return configspace


def ordered_costs_from_runhistory(runhistory) -> List[float]:
    ordered_trials = sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )
    return [float(value.cost) for _, value in ordered_trials]


def best_so_far(costs: List[float]) -> List[float]:
    best = []
    current = float("inf")
    for cost in costs:
        current = min(current, cost)
        best.append(current)
    return best


def run_dynamic_leaf_policy(seed: int = SEED):
    configspace = make_configspace(seed)

    scenario = Scenario(
        name="dynamic_leaf_policy_1_2_3_4",
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model = HPOFacade.get_model(
        scenario=scenario,
        min_samples_leaf=POLICY[0],
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
        callbacks=[DebugLeafPolicyCallback(POLICY)],
        overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = float(smac.runhistory.get_cost(incumbent))

    costs = ordered_costs_from_runhistory(smac.runhistory)
    trajectory = best_so_far(costs)
    summary = {
        "run_type": "dynamic_leaf_policy",
        "seed": seed,
        "policy": list(POLICY),
        "n_trials": N_TRIALS,
        "trials_per_block": TRIALS_PER_BLOCK,
        "n_initial_design_configs": N_INITIAL_DESIGN_CONFIGS,
        "final_incumbent_cost": incumbent_cost,
        "best_at_100": trajectory[99],
        "best_at_200": trajectory[199],
        "best_at_300": trajectory[299],
        "best_at_400": trajectory[399],
        "incumbent": dict(incumbent),
        "run_dir": str(scenario.output_directory),
    }
    with open(scenario.output_directory / "debug_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"dynamic policy={POLICY}, seed={seed}, incumbent_cost={incumbent_cost}")
    print(incumbent)
    return incumbent


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder="logs_debug_dynamic_leaf_policy",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannDebugDynamicLeaf",
        slurm_additional_parameters={"requeue": True},
    )

    job = executor.submit(run_dynamic_leaf_policy, SEED)
    print(f"submitted dynamic leaf policy seed={SEED}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

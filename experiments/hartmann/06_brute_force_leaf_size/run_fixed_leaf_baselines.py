from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from master_utils.benchmarks import hartmann


LEAF_VALUES = [1, 2, 3, 5]
SEEDS = range(5)
N_TRIALS = 400
N_INITIAL_DESIGN_CONFIGS = 25
SLURM_PARTITION = "c23ms"

OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/06_brute_force_leaf_size/"
    "hartmann_6d_fixed_leaf_baselines_400_trials_leaf_1_2_3_5"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


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


def run_smac(seed: int, min_samples_leaf: int):
    configspace = make_configspace(seed)
    name = f"fixed_leaf_{min_samples_leaf}"

    scenario = Scenario(
        name=name,
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
    incumbent_cost = float(smac.runhistory.get_cost(incumbent))

    costs = ordered_costs_from_runhistory(smac.runhistory)
    trajectory = best_so_far(costs)
    milestones = {
        str(n): trajectory[n - 1]
        for n in [100, 200, 300, 400]
        if len(trajectory) >= n
    }

    summary = {
        "seed": seed,
        "min_samples_leaf": min_samples_leaf,
        "policy_name": name,
        "n_trials": N_TRIALS,
        "n_initial_design_configs": N_INITIAL_DESIGN_CONFIGS,
        "final_incumbent_cost": incumbent_cost,
        "milestone_best_costs": milestones,
        "incumbent": dict(incumbent),
        "run_dir": str(scenario.output_directory),
    }

    with open(scenario.output_directory / "policy_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"fixed_leaf={min_samples_leaf}, seed={seed}, incumbent_cost={incumbent_cost}")
    print(incumbent)
    return incumbent


def submit_jobs() -> None:
    job_specs: List[Tuple[int, int]] = [
        (seed, min_samples_leaf)
        for min_samples_leaf in LEAF_VALUES
        for seed in SEEDS
    ]
    print(f"Submitting {len(job_specs)} fixed-leaf baseline jobs.")

    executor = submitit.AutoExecutor(
        folder="logs_fixed_leaf_baselines",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannFixedLeaf",
        slurm_setup=["export PYTHONHASHSEED=0"],
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for seed, min_samples_leaf in job_specs:
            job = executor.submit(run_smac, seed, min_samples_leaf)
            jobs.append((seed, min_samples_leaf, job))

    print("submitted_jobs:")
    for seed, min_samples_leaf, job in jobs:
        print(f"fixed_leaf_{min_samples_leaf}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

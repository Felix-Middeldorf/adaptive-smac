from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from master_utils.benchmarks import hartmann


RUN_LABELS = ["fixed_hashseed_run_a", "fixed_hashseed_run_b"]
SEED = 0
PYTHONHASHSEED = "12345"
N_TRIALS = 400
N_INITIAL_DESIGN_CONFIGS = 25
SLURM_PARTITION = "c23ms"

OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "fixed_pythonhashseed_two_jobs"
LOG_DIRECTORY = Path(__file__).resolve().parent / "logs_fixed_pythonhashseed_two_jobs"


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


def run_smac(run_label: str):
    if run_label not in RUN_LABELS:
        raise ValueError(f"Unknown run_label={run_label!r}; expected one of {RUN_LABELS!r}")

    configspace = make_configspace(SEED)

    scenario = Scenario(
        name=run_label,
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=SEED,
    )

    initial_design = HPOFacade.get_initial_design(
        scenario=scenario,
        n_configs=N_INITIAL_DESIGN_CONFIGS,
    )

    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = float(smac.runhistory.get_cost(incumbent))

    costs = ordered_costs_from_runhistory(smac.runhistory)
    trajectory = best_so_far(costs)
    summary = {
        "run_label": run_label,
        "seed": SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "n_trials": N_TRIALS,
        "n_initial_design_configs": N_INITIAL_DESIGN_CONFIGS,
        "final_incumbent_cost": incumbent_cost,
        "best_at_end": trajectory[-1],
        "incumbent": dict(incumbent),
        "run_dir": str(scenario.output_directory),
    }
    with open(scenario.output_directory / "debug_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"run_label={run_label}, seed={SEED}, "
        f"PYTHONHASHSEED={summary['pythonhashseed']}, "
        f"incumbent_cost={incumbent_cost}"
    )
    print(incumbent)
    return summary


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannDebug06FixedHash",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for run_label in RUN_LABELS:
            job = executor.submit(run_smac, run_label)
            jobs.append((run_label, job))

    print("submitted_jobs:")
    for run_label, job in jobs:
        print(f"{run_label}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

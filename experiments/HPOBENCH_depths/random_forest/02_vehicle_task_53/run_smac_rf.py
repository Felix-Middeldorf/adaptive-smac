from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import submitit
from ConfigSpace import Configuration
from hpobench.container.benchmarks.ml.rf_benchmark import RandomForestBenchmarkBB
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

TASK_ID = 53
SMAC_SEEDS = range(10)
BENCHMARK_SEED = 0
MAX_DEPTH_VALUES = (2, 4, 8, 12, 20)
N_TRIALS = 200
PYTHONHASHSEED = "12345"
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs"
CONTAINER_PATCH = (
    f"{REPOSITORY_ROOT / 'hpobench_container_patches/ml_init_patch.py'}:"
    "/usr/local/lib/python3.8/site-packages/hpobench/benchmarks/ml/__init__.py"
)


def _ordered_costs(runhistory: Any) -> list[float]:
    trials = sorted(runhistory.items(), key=lambda item: (item[1].starttime, item[1].endtime))
    return [float(value.cost) for _, value in trials]


def _best_so_far(costs: list[float]) -> list[float]:
    best, incumbent = [], float("inf")
    for cost in costs:
        incumbent = min(incumbent, cost)
        best.append(incumbent)
    return best


def run_smac(smac_seed: int, max_depth: int) -> dict[str, Any]:
    benchmark = RandomForestBenchmarkBB(
        task_id=TASK_ID, rng=BENCHMARK_SEED, bind_str=CONTAINER_PATCH
    )

    def target_function(config: Configuration, seed: int = 0) -> float:
        del seed
        result = benchmark.objective_function(configuration=config, rng=BENCHMARK_SEED)
        return float(result["function_value"])

    scenario = Scenario(
        name=f"depth_{max_depth}",
        output_directory=OUTPUT_DIRECTORY,
        configspace=benchmark.get_configuration_space(seed=smac_seed),
        deterministic=True,
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = HPOFacade.get_model(scenario=scenario, max_depth=max_depth)
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        overwrite=True,
    )
    incumbent = smac.optimize()
    costs = _ordered_costs(smac.runhistory)
    result = {
        "benchmark": "RandomForestBenchmarkBB",
        "task_id": TASK_ID,
        "task_name": "vehicle",
        "smac_seed": smac_seed,
        "benchmark_seed": BENCHMARK_SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "max_depth": max_depth,
        "n_trials": N_TRIALS,
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(costs) + 1)),
        "cost": costs,
        "best_so_far": _best_so_far(costs),
    }
    result_path = scenario.output_directory / "trajectory.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"seed={smac_seed}, max_depth={max_depth}, incumbent_cost={result['incumbent_cost']}")
    return result


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="HPOBench_RF_DepthsVehicle",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for max_depth in MAX_DEPTH_VALUES:
            for smac_seed in SMAC_SEEDS:
                jobs.append((max_depth, smac_seed, executor.submit(run_smac, smac_seed, max_depth)))

    print(f"Submitted {len(jobs)} jobs:")
    for depth, seed, job in jobs:
        print(f"depth={depth}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

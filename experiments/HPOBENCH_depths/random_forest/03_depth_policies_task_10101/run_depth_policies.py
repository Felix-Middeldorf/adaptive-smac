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
from smac.callback import Callback

TASK_ID = 10101
SMAC_SEEDS = range(10)
BENCHMARK_SEED = 0
N_TRIALS = 200
PYTHONHASHSEED = "12345"
POLICIES = ("rotate_10", "staged_80_50_rest")
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs"
CONTAINER_PATCH = (
    f"{REPOSITORY_ROOT / 'hpobench_container_patches/ml_init_patch.py'}:"
    "/usr/local/lib/python3.8/site-packages/hpobench/benchmarks/ml/__init__.py"
)


def depth_for_trial(policy: str, completed_trials: int) -> int:
    if policy == "rotate_10":
        return (4, 8, 20)[(completed_trials // 10) % 3]
    if policy == "staged_80_50_rest":
        if completed_trials < 80:
            return 4
        if completed_trials < 130:
            return 8
        return 20
    raise ValueError(f"Unknown policy: {policy}")


class DepthPolicyCallback(Callback):
    def __init__(self, policy: str) -> None:
        super().__init__()
        self._policy = policy
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = depth_for_trial(self._policy, completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            print(f"[DepthPolicy] policy={self._policy}, completed_trials={completed_trials}, max_depth={depth}")
            self._last_depth = depth


def _ordered_costs(runhistory: Any) -> list[float]:
    trials = sorted(runhistory.items(), key=lambda item: (item[1].starttime, item[1].endtime))
    return [float(value.cost) for _, value in trials]


def _best_so_far(costs: list[float]) -> list[float]:
    best, incumbent = [], float("inf")
    for cost in costs:
        incumbent = min(incumbent, cost)
        best.append(incumbent)
    return best


def run_smac(smac_seed: int, policy: str) -> dict[str, Any]:
    benchmark = RandomForestBenchmarkBB(task_id=TASK_ID, rng=BENCHMARK_SEED, bind_str=CONTAINER_PATCH)

    def target_function(config: Configuration, seed: int = 0) -> float:
        del seed
        return float(benchmark.objective_function(configuration=config, rng=BENCHMARK_SEED)["function_value"])

    scenario = Scenario(
        name=policy,
        output_directory=OUTPUT_DIRECTORY,
        configspace=benchmark.get_configuration_space(seed=smac_seed),
        deterministic=True,
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = HPOFacade.get_model(scenario=scenario, max_depth=depth_for_trial(policy, 0))
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        callbacks=[DepthPolicyCallback(policy)],
        overwrite=True,
    )
    incumbent = smac.optimize()
    costs = _ordered_costs(smac.runhistory)
    result = {
        "benchmark": "RandomForestBenchmarkBB",
        "task_id": TASK_ID,
        "task_name": "blood-transfusion-service-center",
        "smac_seed": smac_seed,
        "benchmark_seed": BENCHMARK_SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "policy": policy,
        "n_trials": N_TRIALS,
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(costs) + 1)),
        "cost": costs,
        "best_so_far": _best_so_far(costs),
    }
    (scenario.output_directory / "trajectory.json").write_text(json.dumps(result, indent=2))
    print(f"policy={policy}, seed={smac_seed}, incumbent_cost={result['incumbent_cost']}")
    return result


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000)
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="HPOBench_RF_DepthPolicies",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for policy in POLICIES:
            for seed in SMAC_SEEDS:
                jobs.append((policy, seed, executor.submit(run_smac, seed, policy)))
    print(f"Submitted {len(jobs)} jobs:")
    for policy, seed, job in jobs:
        print(f"{policy}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

from __future__ import annotations

import itertools
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


TASK_ID = 10101  # blood-transfusion-service-center
SEED = 0
PYTHONHASHSEED = "12345"
N_TRIALS = 300
TRIALS_PER_BLOCK = 100
N_INITIAL_DESIGN_CONFIGS = 25
MIN_SAMPLES_LEAF_VALUES = (1, 2, 3)
POLICIES = tuple(itertools.product(MIN_SAMPLES_LEAF_VALUES, repeat=3))
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
OUTPUT_DIRECTORY = HERE / "adaptive_output"
LOG_DIRECTORY = HERE / "adaptive_submitit_logs"
CONTAINER_PATCH = (
    f"{REPOSITORY_ROOT / 'hpobench_container_patches/ml_init_patch.py'}:"
    "/usr/local/lib/python3.8/site-packages/hpobench/benchmarks/ml/__init__.py"
)


class BlockMinSamplesLeafCallback(Callback):
    """Switch the SMAC surrogate's leaf size after trials 100 and 200."""

    def __init__(self, policy: tuple[int, int, int]) -> None:
        super().__init__()
        self._policy = policy
        self._last_value: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        block = min(n_trials // TRIALS_PER_BLOCK, len(self._policy) - 1)
        value = self._policy[block]
        config_selector._model._rf_opts["min_samples_leaf"] = value

        if value != self._last_value:
            print(
                f"[BlockMinSamplesLeafCallback] completed_trials={n_trials}, "
                f"block={block + 1}, min_samples_leaf={value}"
            )
            self._last_value = value


def _ordered_costs(runhistory: Any) -> list[float]:
    trials = sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )
    return [float(value.cost) for _, value in trials]


def _best_so_far(costs: list[float]) -> list[float]:
    best = []
    incumbent_cost = float("inf")
    for cost in costs:
        incumbent_cost = min(incumbent_cost, cost)
        best.append(incumbent_cost)
    return best


def run_policy(policy: tuple[int, int, int]) -> dict[str, Any]:
    policy = tuple(int(value) for value in policy)
    if policy not in POLICIES:
        raise ValueError(f"Invalid policy: {policy}")

    policy_name = "adaptive_" + "_".join(map(str, policy))
    benchmark = RandomForestBenchmarkBB(
        task_id=TASK_ID,
        rng=SEED,
        bind_str=CONTAINER_PATCH,
    )

    def target_function(config: Configuration, seed: int = SEED) -> float:
        result = benchmark.objective_function(configuration=config, rng=seed)
        return float(result["function_value"])

    scenario = Scenario(
        name=policy_name,
        output_directory=OUTPUT_DIRECTORY,
        configspace=benchmark.get_configuration_space(seed=SEED),
        deterministic=True,
        n_trials=N_TRIALS,
        seed=SEED,
    )
    model = HPOFacade.get_model(
        scenario=scenario,
        min_samples_leaf=policy[0],
    )
    initial_design = HPOFacade.get_initial_design(
        scenario=scenario,
        n_configs=N_INITIAL_DESIGN_CONFIGS,
    )
    smac = HPOFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        initial_design=initial_design,
        callbacks=[BlockMinSamplesLeafCallback(policy)],
        overwrite=True,
    )

    incumbent = smac.optimize()
    costs = _ordered_costs(smac.runhistory)
    result = {
        "policy_type": "adaptive",
        "policy_name": policy_name,
        "min_samples_leaf_by_block": list(policy),
        "task_id": TASK_ID,
        "task_name": "blood-transfusion-service-center",
        "seed": SEED,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "n_trials": N_TRIALS,
        "trials_per_block": TRIALS_PER_BLOCK,
        "n_initial_design_configs": N_INITIAL_DESIGN_CONFIGS,
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(costs) + 1)),
        "cost": costs,
        "best_so_far": _best_so_far(costs),
        "smac_run_directory": str(scenario.output_directory),
    }
    result_path = scenario.output_directory / "trajectory.json"
    result_path.write_text(json.dumps(result, indent=2))

    print(f"Finished {policy_name}: {result['incumbent_cost']}")
    print(f"Trajectory: {result_path}")
    return result


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="HPOBench_RF_AdaptiveLeaf",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for policy in POLICIES:
            jobs.append((policy, executor.submit(run_policy, policy)))

    print(f"Submitted {len(jobs)} adaptive-policy jobs:")
    for policy, job in jobs:
        print(f"{policy}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

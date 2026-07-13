from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import submitit
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

SMAC_SEED = 0
PROBLEM_SEED = 52
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_TRIALS = 500
LEAF_SIZES = (1, 2, 3)
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs" / "fixed"


def make_instance_map() -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, N_INSTANCES))
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def best_so_far(values: list[float]) -> list[float]:
    return np.minimum.accumulate(values).astype(float).tolist()


def run_fixed_policy(min_samples_leaf: int) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = PROBLEM_SEED
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    policy = f"fixed_leaf_{min_samples_leaf}"
    scenario = Scenario(
        name=policy,
        output_directory=OUTPUT_DIRECTORY / policy,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=SMAC_SEED,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        min_samples_leaf=min_samples_leaf,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        overwrite=True,
    )
    incumbent = smac.optimize()

    trials = ordered_trials(smac.runhistory)
    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    result = {
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "policy": policy,
        "min_samples_leaf": min_samples_leaf,
        "smac_seed": SMAC_SEED,
        "problem_seed": PROBLEM_SEED,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": N_INSTANCES,
        "instance_map": instance_map,
        "n_trials": len(trials),
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "best_so_far": best_so_far(objective_values),
        "trials_per_config": {
            str(config_id): count
            for config_id, count in sorted(trials_per_config.items())
        },
    }
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"policy={policy}, seed={SMAC_SEED}, "
        f"incumbent_cost={result['incumbent_cost']}, output={output_path}"
    )
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
        slurm_array_parallelism=len(LEAF_SIZES),
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O1_FixedLeaf",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for leaf_size in LEAF_SIZES:
            jobs.append(
                (leaf_size, executor.submit(run_fixed_policy, leaf_size))
            )

    print(f"Submitted {len(jobs)} fixed-policy jobs:")
    for leaf_size, job in jobs:
        print(f"min_samples_leaf={leaf_size}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

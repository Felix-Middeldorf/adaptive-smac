from __future__ import annotations

import json
import os
import sys
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
from smac.callback import Callback
from smac.initial_design import RandomInitialDesign

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.c2_parameter_interactions.c2_runner import (
    INSTANCE_SEED,
    N_INITIAL_CONFIGS,
    PROBLEM_CONFIG,
    PYTHONHASHSEED,
    make_instance_map,
    ordered_trials,
)

POLICY = "staged_depth_9_15_after_500"
STAGE_BOUNDARIES = (500,)
DEPTH_SCHEDULE = (9, 15)
SMAC_SEEDS = range(10)
PROBLEM_SEED = 52
N_INSTANCES = 10
N_TRIALS = 1000
DIMENSION = 10
FUNCTION_NAME = "ackley"
OUTPUT_DIRECTORY = HERE / "smac_output"


def depth_for_completed_trials(completed_trials: int) -> int:
    if completed_trials < STAGE_BOUNDARIES[0]:
        return DEPTH_SCHEDULE[0]
    return DEPTH_SCHEDULE[1]


class StagedDepthCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = depth_for_completed_trials(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[StagedDepth9To15] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


def run_staged_depth_9_15(
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DIMENSION,
    function_name: str = FUNCTION_NAME,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = problem_seed
    problem_cfg.problem.function.dim = dimension
    problem_cfg.problem.function.name = function_name
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances)
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        cost = np.asarray(problem.evaluate(trial).cost, dtype=float).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one C2 objective value, got {cost}")
        return float(cost[0])

    scenario = Scenario(
        name=POLICY,
        output_directory=output_directory,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=n_trials,
        seed=smac_seed,
    )
    callback = StagedDepthCallback()
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=DEPTH_SCHEDULE[0],
    )
    initial_design = RandomInitialDesign(
        scenario=scenario,
        n_configs=N_INITIAL_CONFIGS,
        seed=smac_seed,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        initial_design=initial_design,
        callbacks=[callback],
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
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    result = {
        "benchmark": "SynthACticBench",
        "problem": "C2-ParameterInteractions",
        "function_name": function_name,
        "dimension": dimension,
        "policy": POLICY,
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": n_instances,
        "instance_map": instance_map,
        "deterministic": True,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "initial_design_seed": smac_seed,
        "initial_max_depth": DEPTH_SCHEDULE[0],
        "final_max_depth": model._rf_opts["max_depth"],
        "min_samples_leaf": model._rf_opts["min_samples_leaf"],
        "min_samples_split": model._rf_opts["min_samples_split"],
        "stage_boundaries": list(STAGE_BOUNDARIES),
        "depth_schedule": list(DEPTH_SCHEDULE),
        "transitions": callback.transitions,
        "n_trials": len(trials),
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "f_min": f_min,
        "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": (
            np.minimum.accumulate(objective_values).astype(float).tolist()
        ),
        "trials_per_config": {
            str(config_id): count
            for config_id, count in sorted(trials_per_config.items())
        },
    }
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"policy={POLICY}, seed={smac_seed}, output={output_path}")
    return result


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / POLICY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=10,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_C2_Ackley_9_15",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for seed in SMAC_SEEDS:
            job = executor.submit(
                run_staged_depth_9_15,
                seed,
                PROBLEM_SEED,
                OUTPUT_DIRECTORY,
                N_TRIALS,
                N_INSTANCES,
                DIMENSION,
                FUNCTION_NAME,
            )
            jobs.append((seed, job))
    for seed, job in jobs:
        print(f"seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

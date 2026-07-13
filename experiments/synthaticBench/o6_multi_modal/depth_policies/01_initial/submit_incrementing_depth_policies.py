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
from smac.callback import Callback
from smac.initial_design import RandomInitialDesign

SMAC_SEEDS = range(5)
DEPTH_INCREMENTS = (3, 2, 1, 4)
INITIAL_DEPTH = 3
BLOCK_SIZE = 100
PROBLEM_SEED = 52
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_INITIAL_CONFIGS = 10
N_TRIALS = 1000
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O6-Multimodal.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs" / "incrementing_depth"


def policy_name(increment: int) -> str:
    return f"increment_depth_by_{increment}_every_100"


def depth_for_completed_trials(
    completed_trials: int,
    increment: int,
) -> int:
    return INITIAL_DEPTH + increment * (completed_trials // BLOCK_SIZE)


def make_instance_map() -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, N_INSTANCES))
    }


class IncrementingDepthCallback(Callback):
    def __init__(self, increment: int) -> None:
        super().__init__()
        self.increment = increment
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = depth_for_completed_trials(
            completed_trials,
            self.increment,
        )
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[IncrementingDepth] increment={self.increment}, "
                f"completed_trials={completed_trials}, max_depth={depth}"
            )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_incrementing_policy(
    increment: int,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
) -> dict[str, Any]:
    if increment not in DEPTH_INCREMENTS:
        raise ValueError(f"Unsupported depth increment: {increment}")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = problem_seed
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        cost = np.asarray(problem.evaluate(trial).cost, dtype=float).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one O6 objective value, got {cost}")
        return float(cost[0])

    policy = policy_name(increment)
    scenario = Scenario(
        name=policy,
        output_directory=output_directory,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=n_trials,
        seed=smac_seed,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=INITIAL_DEPTH,
    )
    initial_design = RandomInitialDesign(
        scenario=scenario,
        n_configs=N_INITIAL_CONFIGS,
        seed=smac_seed,
    )
    callback = IncrementingDepthCallback(increment)
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
        "problem": "O6-Multimodal",
        "policy": policy,
        "depth_policy": "linear_increment",
        "initial_depth": INITIAL_DEPTH,
        "depth_increment": increment,
        "block_size": BLOCK_SIZE,
        "transitions": callback.transitions,
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": N_INSTANCES,
        "instance_map": instance_map,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "initial_design_seed": smac_seed,
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
    print(
        f"policy={policy}, seed={smac_seed}, "
        f"transitions={callback.transitions}, output={output_path}"
    )
    return result


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O6_IncrementDepth",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for increment in DEPTH_INCREMENTS:
            for smac_seed in SMAC_SEEDS:
                jobs.append(
                    (
                        increment,
                        smac_seed,
                        executor.submit(
                            run_incrementing_policy,
                            increment,
                            smac_seed,
                            PROBLEM_SEED,
                            OUTPUT_DIRECTORY,
                            N_TRIALS,
                        ),
                    )
                )
    print(f"Submitted {len(jobs)} incrementing-depth jobs:")
    for increment, smac_seed, job in jobs:
        print(
            f"increment={increment}, seed={smac_seed}: {job.job_id}"
        )


if __name__ == "__main__":
    submit_jobs()

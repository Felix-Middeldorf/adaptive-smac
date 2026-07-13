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

SMAC_SEEDS = range(5)
PROBLEM_SEED = 52
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_TRIALS = 500
MIN_SAMPLES_SPLIT = 3
STAGE_BOUNDARIES = (150, 300)
SLURM_PARTITION = "c23ms"
POLICY = "staged_leaf_3_2_1"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs" / "adaptive"


def make_instance_map() -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, N_INSTANCES))
    }


def leaf_for_completed_trials(completed_trials: int) -> int:
    if completed_trials < STAGE_BOUNDARIES[0]:
        return 3
    if completed_trials < STAGE_BOUNDARIES[1]:
        return 2
    return 1


class StagedLeafCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[tuple[int, int]] = []
        self._last_leaf: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        leaf = leaf_for_completed_trials(completed_trials)
        config_selector._model._rf_opts["min_samples_leaf"] = leaf
        if leaf != self._last_leaf:
            self.transitions.append((completed_trials, leaf))
            self._last_leaf = leaf
            print(
                f"[StagedLeaf] completed_trials={completed_trials}, "
                f"min_samples_leaf={leaf}"
            )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_adaptive_policy(smac_seed: int) -> dict[str, Any]:
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

    scenario = Scenario(
        name=POLICY,
        output_directory=OUTPUT_DIRECTORY / POLICY,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        min_samples_leaf=3,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )
    callback = StagedLeafCallback()
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
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
        "problem": "O1-DeterministicObjective",
        "policy": POLICY,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "smac_seed": smac_seed,
        "problem_seed": PROBLEM_SEED,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": N_INSTANCES,
        "instance_map": instance_map,
        "n_trials": len(trials),
        "stage_boundaries": list(STAGE_BOUNDARIES),
        "transitions": callback.transitions,
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "f_min": f_min,
        "regret": regret,
        "best_regret": (
            np.minimum.accumulate(regret).astype(float).tolist()
        ),
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
        f"policy={POLICY}, seed={smac_seed}, "
        f"transitions={callback.transitions}, "
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
        slurm_array_parallelism=5,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O1_AdaptiveLeaf5Seeds",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for smac_seed in SMAC_SEEDS:
            jobs.append(
                (
                    smac_seed,
                    executor.submit(run_adaptive_policy, smac_seed),
                )
            )

    print(f"Submitted {len(jobs)} adaptive-policy jobs:")
    for smac_seed, job in jobs:
        print(f"seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

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
FIXED_DEPTHS = (3, 6, 9, 15, 20)
PROBLEM_SEED = 52
INSTANCE_SEED = 0
POLICY_SEED = 2026
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_INITIAL_CONFIGS = 10
N_TRIALS = 1000
STAGE_BOUNDARIES = (100, 200, 500)
STAGED_SCHEDULE = (3, 6, 9, 20)
RANDOM_DEPTHS = (3, 9, 20)
RANDOM_BLOCK_SIZE = 50
SLURM_PARTITION = "c23ms"
STAGED_POLICY = "staged_depth_3_6_9_20"
RANDOM_POLICY = "random_depth_3_9_20_every_50"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O6-Multimodal.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs"


def make_instance_map() -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, N_INSTANCES))
    }


def make_random_schedule() -> tuple[int, ...]:
    rng = np.random.default_rng(POLICY_SEED)
    schedule = [int(rng.choice(RANDOM_DEPTHS))]
    for _ in range(1, N_TRIALS // RANDOM_BLOCK_SIZE):
        choices = [depth for depth in RANDOM_DEPTHS if depth != schedule[-1]]
        schedule.append(int(rng.choice(choices)))
    return tuple(schedule)


RANDOM_SCHEDULE = make_random_schedule()


def staged_depth(completed_trials: int) -> int:
    if completed_trials < STAGE_BOUNDARIES[0]:
        return STAGED_SCHEDULE[0]
    if completed_trials < STAGE_BOUNDARIES[1]:
        return STAGED_SCHEDULE[1]
    if completed_trials < STAGE_BOUNDARIES[2]:
        return STAGED_SCHEDULE[2]
    return STAGED_SCHEDULE[3]


def rotating_depth(completed_trials: int) -> int:
    block = min(
        completed_trials // RANDOM_BLOCK_SIZE,
        len(RANDOM_SCHEDULE) - 1,
    )
    return RANDOM_SCHEDULE[block]


class DepthPolicyCallback(Callback):
    def __init__(self, policy: str) -> None:
        super().__init__()
        self.policy = policy
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        if self.policy == STAGED_POLICY:
            depth = staged_depth(completed_trials)
        elif self.policy == RANDOM_POLICY:
            depth = rotating_depth(completed_trials)
        else:
            raise ValueError(f"Unsupported dynamic policy: {self.policy}")
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[DepthPolicy] policy={self.policy}, "
                f"completed_trials={completed_trials}, max_depth={depth}"
            )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_policy(
    policy: str,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
) -> dict[str, Any]:
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

    if policy.startswith("fixed_depth_"):
        initial_depth = int(policy.rsplit("_", 1)[1])
        callback = None
    elif policy == STAGED_POLICY:
        initial_depth = STAGED_SCHEDULE[0]
        callback = DepthPolicyCallback(policy)
    elif policy == RANDOM_POLICY:
        initial_depth = RANDOM_SCHEDULE[0]
        callback = DepthPolicyCallback(policy)
    else:
        raise ValueError(f"Unknown policy: {policy}")

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
        max_depth=initial_depth,
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
        callbacks=[] if callback is None else [callback],
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
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": N_INSTANCES,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "initial_design_seed": smac_seed,
        "instance_map": instance_map,
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
    if policy.startswith("fixed_depth_"):
        result["max_depth"] = initial_depth
    elif policy == STAGED_POLICY:
        result.update(
            stage_boundaries=list(STAGE_BOUNDARIES),
            depth_schedule=list(STAGED_SCHEDULE),
            transitions=callback.transitions,
        )
    else:
        result.update(
            block_size=RANDOM_BLOCK_SIZE,
            depth_choices=list(RANDOM_DEPTHS),
            depth_schedule=list(RANDOM_SCHEDULE),
            policy_seed=POLICY_SEED,
            transitions=callback.transitions,
        )

    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"policy={policy}, seed={smac_seed}, problem=O6-Multimodal, "
        f"output={output_path}"
    )
    return result


def submit_jobs() -> None:
    policies = (
        tuple(f"fixed_depth_{depth}" for depth in FIXED_DEPTHS)
        + (STAGED_POLICY, RANDOM_POLICY)
    )
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=35,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O6_DepthPolicies",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for policy in policies:
            for smac_seed in SMAC_SEEDS:
                jobs.append(
                    (
                        policy,
                        smac_seed,
                        executor.submit(
                            run_policy,
                            policy,
                            smac_seed,
                            PROBLEM_SEED,
                            OUTPUT_DIRECTORY,
                            N_TRIALS,
                        ),
                    )
                )
    print(f"Submitted {len(jobs)} O6 depth-policy jobs:")
    for policy, smac_seed, job in jobs:
        print(f"policy={policy}, seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

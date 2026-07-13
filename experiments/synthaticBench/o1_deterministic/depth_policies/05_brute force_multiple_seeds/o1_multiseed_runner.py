from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback

DEPTHS = (5, 10, 15, 20)
BENCHMARK_SEEDS = (52, 53, 54, 55, 56)
SMAC_SEEDS = tuple(range(5))
STAGE_BOUNDARIES = (250, 600)
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_TRIALS = 1000

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def policy_name(schedule: tuple[int, int, int]) -> str:
    return "depth_policy_" + "_".join(map(str, schedule))


def depth_for_completed_trials(
    completed_trials: int,
    schedule: tuple[int, int, int],
) -> int:
    if completed_trials < STAGE_BOUNDARIES[0]:
        return schedule[0]
    if completed_trials < STAGE_BOUNDARIES[1]:
        return schedule[1]
    return schedule[2]


class ThreeStageDepthCallback(Callback):
    def __init__(self, schedule: tuple[int, int, int]) -> None:
        super().__init__()
        self.schedule = schedule
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = depth_for_completed_trials(completed_trials, self.schedule)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[ThreeStageDepth] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


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


def trajectory_path(
    family: str,
    benchmark_seed: int,
    name: str,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / family
        / f"benchmark_seed_{benchmark_seed}"
        / name
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    family: str,
    benchmark_seed: int,
    name: str,
    smac_seed: int,
    schedule: tuple[int, int, int],
) -> bool:
    path = trajectory_path(family, benchmark_seed, name, smac_seed)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("n_trials") == N_TRIALS
        and data.get("problem_seed") == benchmark_seed
        and data.get("smac_seed") == smac_seed
        and tuple(data.get("depth_schedule", ())) == schedule
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def run(
    benchmark_seed: int,
    smac_seed: int,
    schedule: tuple[int, int, int],
    fixed: bool = False,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Unexpected benchmark seed {benchmark_seed}.")
    if len(schedule) != 3 or any(depth not in DEPTHS for depth in schedule):
        raise ValueError(f"Invalid schedule {schedule!r}.")
    if fixed and len(set(schedule)) != 1:
        raise ValueError(f"Fixed schedule must be constant: {schedule!r}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )

    family = "fixed" if fixed else "policies"
    name = f"fixed_depth_{schedule[0]}" if fixed else policy_name(schedule)
    if trajectory_is_complete(
        family, benchmark_seed, name, smac_seed, schedule
    ):
        print(
            f"Skipping complete {name}, benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(family, benchmark_seed, name, smac_seed).read_text()
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = benchmark_seed
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    output_root = (
        OUTPUT_DIRECTORY / family / f"benchmark_seed_{benchmark_seed}"
    )
    scenario = Scenario(
        name=name,
        output_directory=output_root,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(scenario=scenario, max_depth=schedule[0])
    callback = None if fixed else ThreeStageDepthCallback(schedule)
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
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
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    result = {
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "policy": name,
        "policy_type": "fixed_depth" if fixed else "three_stage_depth",
        "depth_schedule": list(schedule),
        "stage_boundaries": list(STAGE_BOUNDARIES),
        "benchmark_seed": benchmark_seed,
        "problem_seed": benchmark_seed,
        "smac_seed": smac_seed,
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
        "f_min": f_min,
        "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": np.minimum.accumulate(objective_values).astype(float).tolist(),
        "trials_per_config": {
            str(config_id): count
            for config_id, count in sorted(trials_per_config.items())
        },
    }
    if fixed:
        result["max_depth"] = schedule[0]
    else:
        result["transitions"] = callback.transitions
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"policy={name}, benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}, output={output_path}"
    )
    return result


def run_policy_batch(
    schedule: tuple[int, int, int],
    benchmark_seeds: Iterable[int] = BENCHMARK_SEEDS,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
) -> list[dict[str, Any]]:
    return [
        run(benchmark_seed, smac_seed, schedule)
        for benchmark_seed in benchmark_seeds
        for smac_seed in smac_seeds
    ]


def run_fixed_batch(
    depth: int,
    benchmark_seeds: Iterable[int] = BENCHMARK_SEEDS,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
) -> list[dict[str, Any]]:
    schedule = (depth, depth, depth)
    return [
        run(benchmark_seed, smac_seed, schedule, fixed=True)
        for benchmark_seed in benchmark_seeds
        for smac_seed in smac_seeds
    ]

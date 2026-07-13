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

DEPTHS = (5, 8, 12, 15, 20)
SMAC_SEEDS = tuple(range(5))
PROBLEM_SEED = 52
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INSTANCES = 10
N_TRIALS = 1000
NO_RANDOM_DESIGN_PROBABILITY = 0.0
DEFAULT_RANDOM_DESIGN_PROBABILITY = 0.5

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def random_design_label(probability: float) -> str:
    percentage = int(round(probability * 100))
    if not np.isclose(probability * 100, percentage):
        raise ValueError(f"Probability {probability!r} is not a whole percent.")
    return f"random_design_{percentage}"


def fixed_depth_name(depth: int) -> str:
    return f"fixed_depth_{depth}"


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
    random_design_probability: float,
    depth: int,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / random_design_label(random_design_probability)
        / fixed_depth_name(depth)
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    random_design_probability: float,
    depth: int,
    smac_seed: int,
) -> bool:
    path = trajectory_path(random_design_probability, depth, smac_seed)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    return (
        data.get("n_trials") == N_TRIALS
        and data.get("problem_seed") == PROBLEM_SEED
        and data.get("benchmark_seed") == PROBLEM_SEED
        and data.get("smac_seed") == smac_seed
        and data.get("max_depth") == depth
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            random_design_probability,
        )
        and len(data.get("best_so_far", ())) == N_TRIALS
    )


def run_fixed_depth(
    smac_seed: int,
    depth: int,
    random_design_probability: float,
) -> dict[str, Any]:
    if depth not in DEPTHS:
        raise ValueError(f"Expected depth from {DEPTHS}, got {depth!r}.")
    if not 0.0 <= random_design_probability <= 1.0:
        raise ValueError(
            "random_design_probability must be in [0, 1], got "
            f"{random_design_probability!r}."
        )
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )

    if trajectory_is_complete(random_design_probability, depth, smac_seed):
        print(
            f"Skipping complete {fixed_depth_name(depth)}, "
            f"random_design_probability={random_design_probability}, "
            f"smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(
                random_design_probability,
                depth,
                smac_seed,
            ).read_text()
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = PROBLEM_SEED
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    name = fixed_depth_name(depth)
    scenario = Scenario(
        name=name,
        output_directory=OUTPUT_DIRECTORY
        / random_design_label(random_design_probability),
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(scenario=scenario, max_depth=depth)
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=random_design_probability,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
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
        "policy_type": "fixed_depth",
        "depth_schedule": [depth, depth, depth],
        "max_depth": depth,
        "smac_seed": smac_seed,
        "benchmark_seed": PROBLEM_SEED,
        "problem_seed": PROBLEM_SEED,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": N_INSTANCES,
        "n_trials": len(trials),
        "random_design_probability": random_design_probability,
        "random_design_label": random_design_label(random_design_probability),
        "dimension": int(problem_cfg.task.dimensions),
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "min_samples_split": int(model._rf_opts["min_samples_split"]),
        "instance_map": instance_map,
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
        f"policy={name}, random_design_probability="
        f"{random_design_probability}, seed={smac_seed}, "
        f"output={output_path}"
    )
    return result


def run_fixed_batch(
    depth: int,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
    random_design_probability: float = DEFAULT_RANDOM_DESIGN_PROBABILITY,
) -> list[dict[str, Any]]:
    return [
        run_fixed_depth(smac_seed, depth, random_design_probability)
        for smac_seed in smac_seeds
    ]


def run_no_random_fixed_batch(
    depth: int,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
) -> list[dict[str, Any]]:
    return run_fixed_batch(
        depth,
        smac_seeds,
        NO_RANDOM_DESIGN_PROBABILITY,
    )


def run_default_random_fixed_batch(
    depth: int,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
) -> list[dict[str, Any]]:
    return run_fixed_batch(
        depth,
        smac_seeds,
        DEFAULT_RANDOM_DESIGN_PROBABILITY,
    )

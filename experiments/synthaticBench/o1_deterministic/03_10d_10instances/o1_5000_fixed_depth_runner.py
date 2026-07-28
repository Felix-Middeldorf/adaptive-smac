from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

DEPTHS = (5, 10, 15, 20, 30, 40, 50)
BENCHMARK_SEEDS = (40,)
SMAC_SEEDS = tuple(range(5))
INSTANCE_SEED = 0
INSTANCE_STD = 2.0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
N_TRIALS = 5_000
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 1
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def fixed_depth_name(depth: int) -> str:
    return f"fixed_depth_{depth}"


def make_instance_map() -> dict[str, float]:
    """Create one deterministic map reused by all depths and SMAC seeds."""
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{index}": float(offset)
        for index, offset in enumerate(rng.normal(0, INSTANCE_STD, N_INSTANCES))
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def output_directory(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"benchmark_seed_{benchmark_seed}"
        / fixed_depth_name(depth)
        / str(smac_seed)
    )


def trajectory_path(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return output_directory(benchmark_seed, smac_seed, depth) / "trajectory.json"


def trajectory_is_complete(benchmark_seed: int, smac_seed: int, depth: int) -> bool:
    directory = output_directory(benchmark_seed, smac_seed, depth)
    required = (
        directory / "trajectory.json",
        directory / "runhistory.json",
        directory / "incumbent.json",
        directory / "runtime.json",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        data = json.loads(required[0].read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("policy") == fixed_depth_name(depth)
        and data.get("benchmark_seed") == benchmark_seed
        and data.get("smac_seed") == smac_seed
        and data.get("max_depth") == depth
        and data.get("dimension") == DIMENSION
        and data.get("n_instances") == N_INSTANCES
        and data.get("n_trials") == N_TRIALS
        and data.get("min_samples_leaf") == MIN_SAMPLES_LEAF
        and data.get("min_samples_split") == MIN_SAMPLES_SPLIT
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(data.get("instance_map", ())) == N_INSTANCES
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def run_fixed_depth(
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Benchmark seed must be one of {BENCHMARK_SEEDS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"SMAC seed must be one of {SMAC_SEEDS}.")
    if depth not in DEPTHS:
        raise ValueError(f"Depth must be one of {DEPTHS}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(benchmark_seed, smac_seed, depth):
        print(
            f"Skipping complete benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}, depth={depth}."
        )
        return json.loads(
            trajectory_path(benchmark_seed, smac_seed, depth).read_text()
        )

    total_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = benchmark_seed
    problem_cfg.problem.function.wrapped_bench.dim = DIMENSION
    problem_cfg.task.dimensions = DIMENSION
    problem_cfg.task.search_space_n_floats = DIMENSION
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    name = fixed_depth_name(depth)
    scenario = Scenario(
        name=name,
        output_directory=OUTPUT_DIRECTORY / f"benchmark_seed_{benchmark_seed}",
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        overwrite=True,
    )
    optimization_started = time.perf_counter()
    incumbent = smac.optimize()
    optimization_seconds = time.perf_counter() - optimization_started

    trials = ordered_trials(smac.runhistory)
    if len(trials) != N_TRIALS:
        raise RuntimeError(f"Expected {N_TRIALS} trials, got {len(trials)}.")
    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    incumbent_data = {
        "configuration": dict(incumbent),
        "cost": float(smac.runhistory.get_cost(incumbent)),
    }
    runtime_data = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_seconds": optimization_seconds,
        "total_seconds": time.perf_counter() - total_started,
    }
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "facade": "AlgorithmConfigurationFacade",
        "policy": name,
        "policy_type": "fixed_depth",
        "max_depth": int(model._rf_opts["max_depth"]),
        "benchmark_seed": benchmark_seed,
        "problem_seed": benchmark_seed,
        "smac_seed": smac_seed,
        "instance_seed": INSTANCE_SEED,
        "instance_distribution": "normal",
        "instance_mean": 0.0,
        "instance_standard_deviation": INSTANCE_STD,
        "instance_map": instance_map,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "dimension": DIMENSION,
        "n_instances": N_INSTANCES,
        "n_trials": len(trials),
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "min_samples_split": int(model._rf_opts["min_samples_split"]),
        "config_selector_retrain_after": 8,
        "incumbent": incumbent_data["configuration"],
        "incumbent_cost": incumbent_data["cost"],
        "runtime": runtime_data,
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
    directory = scenario.output_directory
    smac.runhistory.save(directory / "runhistory.json")
    atomic_write_json(directory / "incumbent.json", incumbent_data)
    atomic_write_json(directory / "runtime.json", runtime_data)
    atomic_write_json(directory / "trajectory.json", result)
    print(
        f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}, "
        f"depth={depth}, output={directory}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-seed", type=int, default=40)
    parser.add_argument("--smac-seed", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    run_fixed_depth(args.benchmark_seed, args.smac_seed, args.depth)


if __name__ == "__main__":
    main()

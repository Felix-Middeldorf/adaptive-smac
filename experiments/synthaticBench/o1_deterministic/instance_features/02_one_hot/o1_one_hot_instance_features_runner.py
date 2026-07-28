#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Run O1 with one-hot instance features and SMAC's default random forest."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade
from smac import Scenario


HERE = Path(__file__).resolve().parent
ENUMERATED_EXPERIMENT = HERE.parent / "01_initial"
if str(ENUMERATED_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(ENUMERATED_EXPERIMENT))

# Reuse the complete benchmark setup from the enumerated-feature countercheck.
from o1_enumerated_instance_features_runner import (  # noqa: E402
    BENCHMARK_SEED,
    DIMENSION,
    EXPERIMENT_VERSION,
    INSTANCE_SEED,
    INSTANCE_STD,
    N_INSTANCES,
    N_TRIALS,
    PROBLEM_CONFIG,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    atomic_write_json,
    make_instance_map,
    ordered_trials,
)


OUTPUT_DIRECTORY = HERE / "smac_output"
SCENARIO_NAME = "default_rf_one_hot_instance_features"


def make_one_hot_instance_features(
    instances: dict[str, float],
) -> dict[str, list[float]]:
    """Return one length-N_INSTANCES indicator vector per instance."""
    names = list(instances)
    return {
        name: [float(column == row) for column in range(len(names))]
        for row, name in enumerate(names)
    }


def run_directory(smac_seed: int, output_root: Path = OUTPUT_DIRECTORY) -> Path:
    return output_root / SCENARIO_NAME / str(smac_seed)


def trajectory_is_complete(smac_seed: int) -> bool:
    path = run_directory(smac_seed) / "trajectory.json"
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("benchmark_seed") == BENCHMARK_SEED
        and data.get("smac_seed") == smac_seed
        and data.get("dimension") == DIMENSION
        and data.get("n_instances") == N_INSTANCES
        and data.get("n_trials") == N_TRIALS
        and data.get("instance_feature_encoding") == "one_hot"
        and data.get("instance_feature_dimension") == N_INSTANCES
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def run_experiment(
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_DIRECTORY,
    require_hash_seed: bool = True,
) -> dict[str, Any]:
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"SMAC seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if require_hash_seed and os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if n_trials == N_TRIALS and output_root == OUTPUT_DIRECTORY:
        if trajectory_is_complete(smac_seed):
            print(f"Skipping complete smac_seed={smac_seed}.")
            return json.loads(
                (run_directory(smac_seed) / "trajectory.json").read_text()
            )

    total_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = BENCHMARK_SEED
    problem_cfg.problem.function.wrapped_bench.dim = DIMENSION
    problem_cfg.task.dimensions = DIMENSION
    problem_cfg.task.search_space_n_floats = DIMENSION
    problem = make_problem(problem_cfg)

    instance_map = make_instance_map()
    instance_features = make_one_hot_instance_features(instance_map)
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    scenario = Scenario(
        name=SCENARIO_NAME,
        output_directory=output_root,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        instance_features=instance_features,
        n_trials=n_trials,
        seed=smac_seed,
    )

    # No custom model or random design: exercise the untouched AC facade path.
    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=target_function,
        overwrite=True,
    )
    model = smac._model
    default_model_options = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in model._rf_opts.items()
    }

    optimization_started = time.perf_counter()
    try:
        incumbent = smac.optimize()
    except Exception as error:
        forest = getattr(model, "_rf", None)
        failure = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "benchmark_seed": BENCHMARK_SEED,
            "smac_seed": smac_seed,
            "completed_trials": len(smac.runhistory),
            "configuration_dimensions": DIMENSION,
            "raw_instance_feature_dimensions": N_INSTANCES,
            "pca_components": int(model._pca_components),
            "pca_applied": bool(model._apply_pca),
            "forest_input_dimensions": getattr(forest, "n_features_in_", None),
        }
        atomic_write_json(scenario.output_directory / "failure.json", failure)
        raise
    optimization_seconds = time.perf_counter() - optimization_started

    trials = ordered_trials(smac.runhistory)
    if len(trials) != n_trials:
        raise RuntimeError(f"Expected {n_trials} trials, got {len(trials)}.")
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
        "surrogate": "SMAC default random forest",
        "surrogate_overrides": {},
        "benchmark_seed": BENCHMARK_SEED,
        "problem_seed": BENCHMARK_SEED,
        "smac_seed": smac_seed,
        "dimension": DIMENSION,
        "n_instances": N_INSTANCES,
        "n_trials": len(trials),
        "instance_seed": INSTANCE_SEED,
        "instance_distribution": "normal",
        "instance_mean": 0.0,
        "instance_standard_deviation": INSTANCE_STD,
        "instance_map": instance_map,
        "instance_feature_encoding": "one_hot",
        "instance_feature_dimension": N_INSTANCES,
        "instance_features": instance_features,
        "default_model_options": default_model_options,
        "default_pca_components": int(model._pca_components),
        "pca_applied_after_optimization": bool(model._apply_pca),
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
    print(f"smac_seed={smac_seed}, n_trials={n_trials}, output={directory}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smac-seed", type=int, required=True, choices=SMAC_SEEDS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    run_experiment(
        args.smac_seed,
        n_trials=args.n_trials,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()

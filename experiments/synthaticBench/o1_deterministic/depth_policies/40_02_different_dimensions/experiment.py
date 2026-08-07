"""Dimension study for the compact LLM RF policy and fixed-depth controls."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
SHARED_EXPERIMENT = HERE.parent / "40_llm_chooses"
if str(SHARED_EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(SHARED_EXPERIMENT))

import o1_llm_runner as base
from o1_compact_llm_runner import run_compact_llm_policy


BENCHMARK_SEED = 40
DIMENSIONS = (2, 5, 25, 50, 100)
SMAC_SEEDS = tuple(range(5))
FIXED_DEPTHS = (5, 10, 15, 20, 30)
N_INSTANCES = 10
N_TRIALS = 1_000
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
PCA_COMPONENTS = 4
RANDOM_DESIGN_PROBABILITY = 0.0
FIXED_EXPERIMENT_VERSION = 1
FIXED_N_TREES = 10
COMPARISON_FIXED_N_TREES = 100
COMPACT_POLICY_NAME = "openai_compact_llm_rf_policy"
OUTPUT_ROOT = HERE / "results"


def dimension_root(dimension: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / f"dimension_{dimension}"


def fixed_output_directory(
    dimension: int,
    depth: int,
    smac_seed: int,
    output_root: Path = OUTPUT_ROOT,
    *,
    n_trees: int = FIXED_N_TREES,
) -> Path:
    return (
        dimension_root(dimension, output_root)
        / f"benchmark_seed_{BENCHMARK_SEED}"
        / fixed_policy_name(depth, n_trees)
        / str(smac_seed)
    )


def fixed_policy_name(depth: int, n_trees: int = FIXED_N_TREES) -> str:
    """Keep original 10-tree paths while isolating reruns by tree count."""
    if n_trees == FIXED_N_TREES:
        return f"fixed_depth_{depth}"
    return f"fixed_depth_{depth}_{n_trees}_trees"


def fixed_identity(
    dimension: int,
    depth: int,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    n_trees: int = FIXED_N_TREES,
) -> dict[str, Any]:
    return {
        "experiment_version": FIXED_EXPERIMENT_VERSION,
        "experiment": "different_dimensions_fixed_depth",
        "problem": "O1-DeterministicObjective",
        "benchmark_seed": BENCHMARK_SEED,
        "smac_seed": smac_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": PYTHONHASHSEED,
        "dimension": dimension,
        "n_instances": N_INSTANCES,
        "n_trials": n_trials,
        "deterministic": True,
        "fixed_depth": depth,
        "n_trees": n_trees,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "feature_ratio": 5.0 / 6.0,
        "pca_components": PCA_COMPONENTS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
    }


def run_fixed_depth(
    dimension: int,
    depth: int,
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
    n_trees: int = FIXED_N_TREES,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}.")
    if depth not in FIXED_DEPTHS:
        raise ValueError(f"depth must be one of {FIXED_DEPTHS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trees < 1:
        raise ValueError("n_trees must be positive.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )

    identity = fixed_identity(dimension, depth, smac_seed, n_trials, n_trees)
    output_path = fixed_output_directory(
        dimension, depth, smac_seed, output_root, n_trees=n_trees
    )
    completion_path = output_path / "completed.json"
    trajectory_path = output_path / "trajectory.json"
    if completion_path.exists() and trajectory_path.exists():
        completion = base._read_json(completion_path)
        if (
            completion.get("state") == "complete"
            and completion.get("identity") == identity
        ):
            print(f"Skipping complete run {output_path}.")
            return base._read_json(trajectory_path)

    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and base._read_json(identity_path) != identity:
        raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(identity_path, identity)
    resume = base._resume_state_is_valid(output_path)

    problem_cfg = OmegaConf.load(base.PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = BENCHMARK_SEED
    problem_cfg.problem.function.wrapped_bench.dim = dimension
    problem_cfg.task.dimensions = dimension
    problem_cfg.task.search_space_n_floats = dimension
    problem = make_problem(problem_cfg)
    instance_map = base.make_instance_map(N_INSTANCES, INSTANCE_SEED)
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        return float(
            problem.evaluate(
                TrialInfo(config=config, instance=instance, seed=seed)
            ).cost
        )

    scenario_root = dimension_root(dimension, output_root) / f"benchmark_seed_{BENCHMARK_SEED}"
    policy_name = fixed_policy_name(depth, n_trees)
    scenario = base.Scenario(
        name=policy_name,
        output_directory=scenario_root,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=n_trials,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != output_path:
        raise RuntimeError(
            f"Unexpected output {scenario.output_directory}; expected {output_path}."
        )
    model = base.ACFacade.get_model(
        scenario=scenario,
        n_trees=n_trees,
        ratio_features=5.0 / 6.0,
        min_samples_split=3,
        min_samples_leaf=3,
        max_depth=depth,
        pca_components=PCA_COMPONENTS,
    )
    random_design = base.ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    smac = base.ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        overwrite=not resume,
    )
    base.atomic_write_json(
        output_path / "run_metadata.json",
        {
            **identity,
            "output_directory": str(output_path),
            "instance_map": instance_map,
            "local_smac_root": str(base.LOCAL_SMAC_ROOT),
        },
    )
    base.atomic_write_json(completion_path, {"state": "running", "identity": identity})
    started = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - started

    trials = base.ordered_trials(smac.runhistory)
    costs = [float(np.asarray(value.cost).reshape(-1)[0]) for _, value in trials]
    objective_values = [
        float(np.asarray(value.cost).reshape(-1)[0]) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(int(key.config_id) for key, _ in trials)
    result = {
        **identity,
        "benchmark": "SynthACticBench",
        "policy": policy_name,
        "instance_map": instance_map,
        "finished_trials": len(trials),
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
            str(key): value for key, value in sorted(trials_per_config.items())
        },
        "walltime_seconds_this_process": walltime,
    }
    base.atomic_write_json(trajectory_path, result)
    base.atomic_write_json(completion_path, {"state": "complete", "identity": identity})
    print(
        f"Completed dimension={dimension}, fixed_depth={depth}, n_trees={n_trees}, "
        f"smac_seed={smac_seed}, output={output_path}."
    )
    return result


def run_compact(
    dimension: int,
    smac_seed: int,
    *,
    decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    return run_compact_llm_policy(
        BENCHMARK_SEED,
        smac_seed,
        n_trials=n_trials,
        output_root=dimension_root(dimension, output_root),
        dimension=dimension,
        n_instances=N_INSTANCES,
        instance_seed=INSTANCE_SEED,
        policy_name=COMPACT_POLICY_NAME,
        decision_provider=decision_provider,
    )


def fixed_jobs(
    dimensions: tuple[int, ...] = DIMENSIONS,
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (dimension, depth, smac_seed)
        for dimension in dimensions
        for depth in FIXED_DEPTHS
        for smac_seed in SMAC_SEEDS
    )


def compact_jobs(
    dimensions: tuple[int, ...] = DIMENSIONS,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (dimension, smac_seed)
        for dimension in dimensions
        for smac_seed in SMAC_SEEDS
    )

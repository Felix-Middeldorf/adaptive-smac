"""Evaluate frozen LLM-proposed RF configurations across all dimensions."""

from __future__ import annotations

import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf

import experiment
from o1_compact_llm_runner import CompactRFSettings


EXPERIMENT_VERSION = 1
SOURCE_SMAC_SEED = 0
SOURCE_CHECKPOINTS = (100, 500)
DEFAULT_EVALUATION_SMAC_SEEDS = (0,)
POLICY_PREFIX = "fixed_llm_config"


@dataclass(frozen=True)
class SelectedConfiguration:
    identifier: str
    source_dimension: int
    source_smac_seed: int
    source_checkpoint: int
    settings: CompactRFSettings

    @property
    def policy_name(self) -> str:
        return f"{POLICY_PREFIX}_source_d{self.source_dimension}_cp{self.source_checkpoint}"

    def provenance(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "source_dimension": self.source_dimension,
            "source_smac_seed": self.source_smac_seed,
            "source_checkpoint": self.source_checkpoint,
        }


def _selected(
    source_dimension: int,
    checkpoint: int,
    *,
    n_trees: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    feature_ratio: float,
) -> SelectedConfiguration:
    return SelectedConfiguration(
        identifier=f"source_d{source_dimension}_cp{checkpoint}",
        source_dimension=source_dimension,
        source_smac_seed=SOURCE_SMAC_SEED,
        source_checkpoint=checkpoint,
        settings=CompactRFSettings(
            n_trees=n_trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            feature_ratio=feature_ratio,
        ),
    )


# Reproducible selection rule: seed 0 proposals at checkpoints 100 and 500.
SELECTED_CONFIGURATIONS = (
    _selected(2, 100, n_trees=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, feature_ratio=1.0),
    _selected(2, 500, n_trees=100, max_depth=30, min_samples_split=2, min_samples_leaf=1, feature_ratio=1.0),
    _selected(5, 100, n_trees=100, max_depth=15, min_samples_split=2, min_samples_leaf=1, feature_ratio=0.8),
    _selected(5, 500, n_trees=100, max_depth=25, min_samples_split=2, min_samples_leaf=1, feature_ratio=1.0),
    _selected(25, 100, n_trees=100, max_depth=10, min_samples_split=2, min_samples_leaf=2, feature_ratio=0.9),
    _selected(25, 500, n_trees=100, max_depth=20, min_samples_split=2, min_samples_leaf=2, feature_ratio=0.9),
    _selected(50, 100, n_trees=100, max_depth=12, min_samples_split=2, min_samples_leaf=1, feature_ratio=5.0 / 6.0),
    _selected(50, 500, n_trees=100, max_depth=20, min_samples_split=2, min_samples_leaf=1, feature_ratio=5.0 / 6.0),
    _selected(100, 100, n_trees=50, max_depth=12, min_samples_split=2, min_samples_leaf=1, feature_ratio=0.5),
    _selected(100, 500, n_trees=100, max_depth=20, min_samples_split=2, min_samples_leaf=1, feature_ratio=0.5),
)
CONFIGURATIONS_BY_ID = {item.identifier: item for item in SELECTED_CONFIGURATIONS}


def source_trajectory(configuration: SelectedConfiguration) -> Path:
    return (
        experiment.dimension_root(configuration.source_dimension)
        / f"benchmark_seed_{experiment.BENCHMARK_SEED}"
        / experiment.COMPACT_POLICY_NAME
        / str(configuration.source_smac_seed)
        / "trajectory.json"
    )


def validate_source(configuration: SelectedConfiguration) -> None:
    path = source_trajectory(configuration)
    if not path.is_file():
        raise RuntimeError(f"Missing source trajectory for {configuration.identifier}: {path}")
    payload = experiment.base._read_json(path)
    proposed = payload["llm_policy"]["decisions"][str(configuration.source_checkpoint)]["settings"]
    if CompactRFSettings.from_mapping(proposed) != configuration.settings:
        raise RuntimeError(f"Frozen settings differ from their source in {path}.")


def output_directory(
    target_dimension: int,
    configuration: SelectedConfiguration,
    smac_seed: int,
    output_root: Path = experiment.OUTPUT_ROOT,
) -> Path:
    return (
        experiment.dimension_root(target_dimension, output_root)
        / f"benchmark_seed_{experiment.BENCHMARK_SEED}"
        / configuration.policy_name
        / str(smac_seed)
    )


def run_configuration(
    target_dimension: int,
    configuration_id: str,
    smac_seed: int,
    *,
    n_trials: int = experiment.N_TRIALS,
    output_root: Path = experiment.OUTPUT_ROOT,
) -> dict[str, Any]:
    if target_dimension not in experiment.DIMENSIONS:
        raise ValueError(f"target_dimension must be one of {experiment.DIMENSIONS}.")
    if configuration_id not in CONFIGURATIONS_BY_ID:
        raise ValueError(f"Unknown configuration: {configuration_id}.")
    if smac_seed not in experiment.SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {experiment.SMAC_SEEDS}.")
    if os.environ.get("PYTHONHASHSEED") != experiment.PYTHONHASHSEED:
        raise RuntimeError(f"Expected PYTHONHASHSEED={experiment.PYTHONHASHSEED}.")

    configuration = CONFIGURATIONS_BY_ID[configuration_id]
    validate_source(configuration)
    settings = configuration.settings
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "cross_dimension_fixed_llm_configuration",
        "problem": "O1-DeterministicObjective",
        "benchmark_seed": experiment.BENCHMARK_SEED,
        "smac_seed": smac_seed,
        "instance_seed": experiment.INSTANCE_SEED,
        "pythonhashseed": experiment.PYTHONHASHSEED,
        "dimension": target_dimension,
        "n_instances": experiment.N_INSTANCES,
        "n_trials": n_trials,
        "deterministic": True,
        "selected_configuration": configuration.provenance(),
        **settings.to_dict(),
        "pca_components": experiment.PCA_COMPONENTS,
        "random_design_probability": experiment.RANDOM_DESIGN_PROBABILITY,
    }
    output_path = output_directory(
        target_dimension, configuration, smac_seed, output_root
    )
    completion_path = output_path / "completed.json"
    trajectory_path = output_path / "trajectory.json"
    if completion_path.exists() and trajectory_path.exists():
        completion = experiment.base._read_json(completion_path)
        if completion.get("state") == "complete" and completion.get("identity") == identity:
            print(f"Skipping complete run {output_path}.")
            return experiment.base._read_json(trajectory_path)

    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and experiment.base._read_json(identity_path) != identity:
        raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    experiment.base.atomic_write_json(identity_path, identity)
    resume = experiment.base._resume_state_is_valid(output_path)

    problem_cfg = OmegaConf.load(experiment.base.PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = experiment.BENCHMARK_SEED
    problem_cfg.problem.function.wrapped_bench.dim = target_dimension
    problem_cfg.task.dimensions = target_dimension
    problem_cfg.task.search_space_n_floats = target_dimension
    problem = make_problem(problem_cfg)
    instance_map = experiment.base.make_instance_map(
        experiment.N_INSTANCES, experiment.INSTANCE_SEED
    )
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        return float(
            problem.evaluate(
                TrialInfo(config=config, instance=instance, seed=seed)
            ).cost
        )

    scenario_root = (
        experiment.dimension_root(target_dimension, output_root)
        / f"benchmark_seed_{experiment.BENCHMARK_SEED}"
    )
    scenario = experiment.base.Scenario(
        name=configuration.policy_name,
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
    model = experiment.base.ACFacade.get_model(
        scenario=scenario,
        n_trees=settings.n_trees,
        ratio_features=settings.feature_ratio,
        min_samples_split=settings.min_samples_split,
        min_samples_leaf=settings.min_samples_leaf,
        max_depth=settings.max_depth,
        pca_components=experiment.PCA_COMPONENTS,
    )
    random_design = experiment.base.ACFacade.get_random_design(
        scenario=scenario,
        probability=experiment.RANDOM_DESIGN_PROBABILITY,
    )
    smac = experiment.base.ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        overwrite=not resume,
    )
    experiment.base.atomic_write_json(
        output_path / "run_metadata.json",
        {
            **identity,
            "output_directory": str(output_path),
            "source_trajectory": str(source_trajectory(configuration)),
            "instance_map": instance_map,
            "local_smac_root": str(experiment.base.LOCAL_SMAC_ROOT),
        },
    )
    experiment.base.atomic_write_json(
        completion_path, {"state": "running", "identity": identity}
    )
    started = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - started

    trials = experiment.base.ordered_trials(smac.runhistory)
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
        "policy": configuration.policy_name,
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
    experiment.base.atomic_write_json(trajectory_path, result)
    experiment.base.atomic_write_json(
        completion_path, {"state": "complete", "identity": identity}
    )
    print(
        f"Completed target_dimension={target_dimension}, "
        f"configuration={configuration_id}, smac_seed={smac_seed}."
    )
    return result


def job_arguments(
    target_dimensions: tuple[int, ...] = experiment.DIMENSIONS,
    smac_seeds: tuple[int, ...] = DEFAULT_EVALUATION_SMAC_SEEDS,
) -> tuple[tuple[int, str, int], ...]:
    return tuple(
        (target_dimension, configuration.identifier, smac_seed)
        for target_dimension in target_dimensions
        for configuration in SELECTED_CONFIGURATIONS
        for smac_seed in smac_seeds
    )

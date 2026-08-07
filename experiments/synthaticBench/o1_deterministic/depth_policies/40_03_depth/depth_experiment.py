"""Depth-only compact-LLM ablation with all other RF settings fixed."""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf


HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / "40_llm_chooses"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

import o1_compact_llm_runner as compact
import o1_llm_runner as base


BENCHMARK_SEED = 40
DIMENSIONS = (25, 50, 100)
SMAC_SEEDS = tuple(range(5))
FIXED_DEPTHS = (5, 10, 15, 20, 30)
N_INSTANCES = 10
N_TRIALS = 1_000
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_TREES = 100
MIN_SAMPLES_SPLIT = 2  # Smallest value accepted by sklearn trees.
MIN_SAMPLES_LEAF = 1
FEATURE_RATIO = 5.0 / 6.0
PCA_COMPONENTS = 4
RANDOM_DESIGN_PROBABILITY = 0.0
CHECKPOINTS = (100, 250, 500)
POLICY_NAME = "openai_compact_depth_only_policy"
EXPERIMENT_VERSION = 1
POLICY_VERSION = 1
OUTPUT_ROOT = HERE / "results"


@dataclass(frozen=True)
class DepthSettings:
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    feature_ratio: float

    def __post_init__(self) -> None:
        if self.n_trees != N_TREES:
            raise ValueError(f"n_trees must remain fixed at {N_TREES}.")
        if not 1 <= self.max_depth <= 30:
            raise ValueError("max_depth must be an integer in [1, 30].")
        if self.min_samples_split != MIN_SAMPLES_SPLIT:
            raise ValueError(f"min_samples_split must be {MIN_SAMPLES_SPLIT}.")
        if self.min_samples_leaf != MIN_SAMPLES_LEAF:
            raise ValueError(f"min_samples_leaf must be {MIN_SAMPLES_LEAF}.")
        if not math.isclose(self.feature_ratio, FEATURE_RATIO):
            raise ValueError(f"feature_ratio must remain fixed at {FEATURE_RATIO}.")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_ratio": self.feature_ratio,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DepthSettings":
        return cls(
            n_trees=int(value["n_trees"]),
            max_depth=int(value["max_depth"]),
            min_samples_split=int(value["min_samples_split"]),
            min_samples_leaf=int(value["min_samples_leaf"]),
            feature_ratio=float(value["feature_ratio"]),
        )


def settings(depth: int) -> DepthSettings:
    return DepthSettings(
        n_trees=N_TREES,
        max_depth=depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        feature_ratio=FEATURE_RATIO,
    )


INITIAL_SETTINGS = settings(20)
DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "max_depth": {"type": "integer", "minimum": 1, "maximum": 30},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": ["max_depth", "confidence", "reason"],
    "additionalProperties": False,
}


def validate_depth_decision(
    payload: dict[str, Any],
) -> tuple[DepthSettings, dict[str, Any]]:
    if set(payload) != set(DECISION_SCHEMA["required"]):
        raise ValueError("Depth decision has missing or additional fields.")
    depth = payload["max_depth"]
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= 30:
        raise ValueError("max_depth must be an integer in [1, 30].")
    confidence = float(payload["confidence"])
    reason = str(payload["reason"]).strip()
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if not reason or len(reason) > 500:
        raise ValueError("reason must contain 1 to 500 characters.")
    chosen = settings(depth)
    return chosen, {
        "max_depth": depth,
        "confidence": confidence,
        "reason": reason,
    }


def depth_prompt(summary: dict[str, Any]) -> str:
    prompt = f"""You choose only the maximum tree depth for the random-forest
surrogate used in the next phase of a SMAC algorithm-configuration run. Lower
objective values are better. All other forest settings are immutable: 100
trees, minimum split size 2, minimum leaf size 1, feature ratio 5/6, and 4 PCA
components. Choose an integer max_depth from 1 through 30 that you expect to
give the best SMAC optimization performance during the remaining trials.

The compact data describe the objective dimensionality and bounds, optimization
progress, expected improvement, marginalized predictions, prediction variance,
proxy errors, evaluation allocation, and actual fitted-tree depths. Use those
signals and the previous depth decisions. Return only the structured object
required by the response schema.

Important definitions:
- absolute_proxy_error compares a proposal-time marginalized prediction with
  that configuration's first observed instance cost; it adds no evaluations.
- relative_proxy_error divides this error by max(abs(proxy), 1e-12).
- best_config_mean_cost is the best running mean observed cost using only
  evaluations available by that trial.
- depth_utilization is mean actual tree depth divided by the depth limit.

COMPACT DATA
{json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)}
"""
    forbidden = ("synthactic", "o1-deterministic", "benchmark_seed")
    if any(value in prompt.lower() for value in forbidden):
        raise RuntimeError("The prompt leaks workload identity.")
    return prompt


class CompactDepthPolicyCallback(base.LLMRFPolicyCallback):
    def __init__(self, *, objective_dimension: int, **kwargs: Any) -> None:
        self.objective_dimension = objective_dimension
        super().__init__(
            **kwargs,
            settings_class=DepthSettings,
            decision_validator=validate_depth_decision,
            prompt_builder=depth_prompt,
            policy_version=POLICY_VERSION,
        )

    def _summary(self, checkpoint: int, trigger_trial: int, runhistory: Any) -> dict[str, Any]:
        summary = compact.compact_summary(
            checkpoint=checkpoint,
            trigger_trial=trigger_trial,
            runhistory=runhistory,
            telemetry_path=self.telemetry_path,
            current_settings=self.next_settings,
            decisions=self.state["decisions"],
            fit_observations=self.state["fit_observations"],
            objective_dimension=self.objective_dimension,
        )
        summary["allowed_next_settings"] = {"max_depth": "integer in [1,30]"}
        summary["immutable_rf_settings"] = {
            "n_trees": N_TREES,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "feature_ratio": FEATURE_RATIO,
            "pca_components": PCA_COMPONENTS,
        }
        return summary


def dimension_root(dimension: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return Path(output_root) / f"dimension_{dimension}"


def run_depth_policy(
    dimension: int,
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
    decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS or smac_seed not in SMAC_SEEDS:
        raise ValueError("Unsupported dimension or SMAC seed.")
    provider = decision_provider or base.OpenAIResponsesClient(
        decision_schema=DECISION_SCHEMA,
        schema_name="smac_depth_only_choice",
    ).invoke
    return base.run_llm_policy(
        BENCHMARK_SEED,
        smac_seed,
        n_trials=n_trials,
        output_root=dimension_root(dimension, output_root),
        decision_provider=provider,
        policy_name=POLICY_NAME,
        experiment_version=EXPERIMENT_VERSION,
        callback_factory=functools.partial(
            CompactDepthPolicyCallback,
            objective_dimension=dimension,
        ),
        identity_extra={
            "policy_scope": "max_depth_only",
            "allowed_max_depth": [1, 30],
            "summary_mode": "ten_window_compact_aggregates",
            "summary_windows": compact.SUMMARY_WINDOWS,
        },
        dimension=dimension,
        n_instances=N_INSTANCES,
        instance_seed=INSTANCE_SEED,
        initial_settings=INITIAL_SETTINGS,
    )


def fixed_output_directory(dimension: int, depth: int, smac_seed: int, output_root: Path) -> Path:
    return dimension_root(dimension, output_root) / f"benchmark_seed_{BENCHMARK_SEED}" / f"fixed_depth_{depth}" / str(smac_seed)


def run_fixed_depth(
    dimension: int,
    depth: int,
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS or depth not in FIXED_DEPTHS or smac_seed not in SMAC_SEEDS:
        raise ValueError("Unsupported fixed-run parameters.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(f"Expected PYTHONHASHSEED={PYTHONHASHSEED}.")
    chosen = settings(depth)
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "depth_only_fixed_control",
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
        **chosen.to_dict(),
        "pca_components": PCA_COMPONENTS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
    }
    output_path = fixed_output_directory(dimension, depth, smac_seed, output_root)
    completion_path, trajectory_path = output_path / "completed.json", output_path / "trajectory.json"
    if completion_path.exists() and trajectory_path.exists():
        completion = base._read_json(completion_path)
        if completion.get("state") == "complete" and completion.get("identity") == identity:
            return base._read_json(trajectory_path)
    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and base._read_json(identity_path) != identity:
        raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(identity_path, identity)
    resume = base._resume_state_is_valid(output_path)

    cfg = OmegaConf.load(base.PROBLEM_CONFIG)
    cfg.problem.function.wrapped_bench.seed = BENCHMARK_SEED
    cfg.problem.function.wrapped_bench.dim = dimension
    cfg.task.dimensions = dimension
    cfg.task.search_space_n_floats = dimension
    problem = make_problem(cfg)
    instance_map = base.make_instance_map(N_INSTANCES, INSTANCE_SEED)
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        return float(problem.evaluate(TrialInfo(config=config, instance=instance, seed=seed)).cost)

    scenario_root = dimension_root(dimension, output_root) / f"benchmark_seed_{BENCHMARK_SEED}"
    scenario = base.Scenario(
        name=f"fixed_depth_{depth}", output_directory=scenario_root,
        configspace=problem.configspace, deterministic=True,
        instances=list(instance_map), n_trials=n_trials, seed=smac_seed, n_workers=1,
    )
    if scenario.output_directory != output_path:
        raise RuntimeError(f"Unexpected output path {scenario.output_directory}.")
    model = base.ACFacade.get_model(
        scenario=scenario, n_trees=N_TREES, ratio_features=FEATURE_RATIO,
        min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=depth, pca_components=PCA_COMPONENTS,
    )
    random_design = base.ACFacade.get_random_design(scenario=scenario, probability=0.0)
    smac = base.ACFacade(
        scenario=scenario, target_function=target_function, model=model,
        random_design=random_design, overwrite=not resume,
    )
    base.atomic_write_json(output_path / "run_metadata.json", {**identity, "instance_map": instance_map, "local_smac_root": str(base.LOCAL_SMAC_ROOT)})
    base.atomic_write_json(completion_path, {"state": "running", "identity": identity})
    started = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - started
    trials = base.ordered_trials(smac.runhistory)
    costs = [float(np.asarray(v.cost).reshape(-1)[0]) for _, v in trials]
    objective = [float(np.asarray(v.cost).reshape(-1)[0]) - instance_map[k.instance] for k, v in trials]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective]
    counts = Counter(int(k.config_id) for k, _ in trials)
    result = {
        **identity, "benchmark": "SynthACticBench", "policy": f"fixed_depth_{depth}",
        "instance_map": instance_map, "finished_trials": len(trials),
        "incumbent": dict(incumbent), "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)), "cost": costs,
        "objective_value": objective, "f_min": f_min, "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": np.minimum.accumulate(objective).astype(float).tolist(),
        "trials_per_config": {str(k): v for k, v in sorted(counts.items())},
        "walltime_seconds_this_process": walltime,
    }
    base.atomic_write_json(trajectory_path, result)
    base.atomic_write_json(completion_path, {"state": "complete", "identity": identity})
    return result


def fixed_jobs() -> tuple[tuple[int, int, int], ...]:
    return tuple((d, depth, seed) for d in DIMENSIONS for depth in FIXED_DEPTHS for seed in SMAC_SEEDS)


def llm_jobs() -> tuple[tuple[int, int], ...]:
    return tuple((d, seed) for d in DIMENSIONS for seed in SMAC_SEEDS)

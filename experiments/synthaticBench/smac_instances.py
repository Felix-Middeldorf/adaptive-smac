"""Instance-aware SynthACticBench integration for SMAC."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.trials import TrialInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf
from smac.callback import Callback


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SYNTHACTIC_ROOT = REPOSITORY_ROOT / "external" / "SynthACticBench"
PROBLEM_CONFIG_ROOT = (
    SYNTHACTIC_ROOT / "synthacticbench" / "configs" / "problem" / "SynthACticBench"
)


def load_problem(config_name: str = "O1-DeterministicObjective.yaml") -> Any:
    """Instantiate a SynthACticBench problem from one of its packaged YAML files."""
    config_path = PROBLEM_CONFIG_ROOT / config_name
    if not config_path.is_file():
        available = ", ".join(sorted(path.name for path in PROBLEM_CONFIG_ROOT.glob("*.yaml")))
        raise FileNotFoundError(
            f"Unknown SynthACticBench problem {config_name!r}. Available problems: {available}"
        )

    cfg = OmegaConf.load(config_path)
    return instantiate(cfg.problem)


def generate_instance_map(
    n_instances: int,
    *,
    seed: int,
    mean: float = 0.0,
    std: float = 2.0,
) -> dict[str, float]:
    """Generate the instance offsets used by SynthACticBench."""
    if n_instances < 1:
        raise ValueError("n_instances must be at least one")
    if std < 0:
        raise ValueError("std must be non-negative")

    rng = np.random.default_rng(seed)
    offsets = rng.normal(loc=mean, scale=std, size=n_instances)
    return {f"i{index}": float(offset) for index, offset in enumerate(offsets)}


def make_instance_features(instances: Mapping[str, float]) -> dict[str, list[float]]:
    """Represent each SynthACticBench offset as a one-dimensional SMAC feature."""
    return {name: [float(offset)] for name, offset in instances.items()}


def make_target_function(problem: Any, instances: Mapping[str, float]) -> Any:
    """Create a SMAC-compatible target that preserves benchmark instances."""
    if getattr(problem.function, "benchmark_name", None) == "o3":
        raise TypeError(
            "This target supports scalar objectives only; use a multi-objective "
            "adapter for O3-MultipleObjectives.yaml"
        )
    instance_map = dict(instances)
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        # SynthACticBench's current evaluation API does not consume SMAC's seed.
        del seed
        if instance not in instance_map:
            raise KeyError(f"Unknown instance {instance!r}")
        value = problem.evaluate(TrialInfo(config=config, instance=instance))
        cost = np.asarray(value.cost)
        if cost.ndim != 0:
            raise TypeError(
                "This target supports scalar objectives only; use a multi-objective "
                "adapter for O3-MultipleObjectives.yaml"
            )
        return float(cost)

    return target_function


def depth_for_trial(policy: str, completed_trials: int) -> int:
    """Return the random-forest depth for an adaptive-depth policy."""
    if policy == "rotate_10":
        return (4, 8, 20)[(completed_trials // 10) % 3]
    if policy == "staged_80_50_rest":
        if completed_trials < 80:
            return 4
        if completed_trials < 130:
            return 8
        return 20
    raise ValueError(f"Unknown policy: {policy}")


class DepthPolicyCallback(Callback):
    """Apply the existing adaptive-depth policies to SMAC's random forest."""

    def __init__(self, policy: str) -> None:
        super().__init__()
        self.policy = policy
        self.last_depth: int | None = None
        self.depth_changes: list[tuple[int, int]] = []

    def on_next_configurations_start(self, config_selector: Any) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = depth_for_trial(self.policy, completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self.last_depth:
            self.depth_changes.append((completed_trials, depth))
            self.last_depth = depth


class FixedMinSamplesLeafCallback(Callback):
    """Set SMAC's random-forest minimum leaf size before every surrogate fit."""

    def __init__(self, min_samples_leaf: int) -> None:
        super().__init__()
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least one")
        self.min_samples_leaf = min_samples_leaf
        self.observed_values: list[int] = []

    def on_next_configurations_start(self, config_selector: Any) -> None:
        config_selector._model._rf_opts["min_samples_leaf"] = self.min_samples_leaf
        self.observed_values.append(
            int(config_selector._model._rf_opts["min_samples_leaf"])
        )

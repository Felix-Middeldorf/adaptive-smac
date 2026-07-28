"""Adaptive RF-depth policies for deterministic ACLib surrogate experiments."""

from __future__ import annotations

import csv
import fcntl
import functools
import hashlib
import inspect
import json
import math
import os
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ConfigSpace import Configuration


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"
local_smac = str(LOCAL_SMAC_ROOT)
if local_smac in sys.path:
    sys.path.remove(local_smac)
sys.path.insert(0, local_smac)

from smac import AlgorithmConfigurationFacade, Scenario
from smac.callback import Callback
from smac.initial_design.random_design import RandomInitialDesign
from smac.model.random_forest.random_forest import RandomForest
from smac.utils.configspace import convert_configurations_to_array

from fixed_depth_experiment import (
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_TREES,
    PCA_COMPONENTS,
    RANDOM_DESIGN_PROBABILITY,
    load_initial_choice,
    resolve_initial_configuration,
)
from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)
from surrogate_telemetry import (
    TELEMETRY_FILENAME,
    TELEMETRY_SCHEMA_VERSION,
    TELEMETRY_SUMMARY_FILENAME,
    SurrogateTelemetryCallback,
    TelemetryEI,
)


N_TRIALS = 5_000
SMAC_SEEDS = (0, 1, 2)
INITIAL_DEPTH = 5
DEPTH_INCREMENT = 5
SATURATION_FRACTION = 0.9
ERROR_TREND_WINDOW = 25
ERROR_TREND_MIN_RANK_CORRELATION = 0.5
ERROR_TREND_MIN_FITTED_GROWTH = 1.25
EXPERIMENT_VERSION = 1

POLICIES = (
    "saturation_k50",
    "saturation_k100",
    "saturation_k250",
    "rotating_saturation_k50",
    "error_variance_trend_25",
    "oracle_incumbent_depth",
)


@dataclass(frozen=True)
class PolicyExperimentDefinition:
    benchmark_key: str
    initials: str
    directory: Path
    initial_directory: Path

    @property
    def output_root(self) -> Path:
        return self.directory / "results"

    @property
    def initial_choice_file(self) -> Path:
        return self.initial_directory / "initial_config.json"

    @property
    def incumbent_events_file(self) -> Path:
        return (
            self.initial_directory
            / "analytics_cache"
            / "incumbent_trajectory_events.csv"
        )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _acquire_lock(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"Policy run is already active: {path}") from error
    return handle


def run_directory(
    output_root: Path,
    policy: str,
    smac_seed: int,
) -> Path:
    return output_root / policy / str(smac_seed)


def _fingerprint(config: Configuration | Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_oracle_schedule(
    definition: PolicyExperimentDefinition,
    smac_seed: int,
) -> dict[str, Any]:
    """Choose the best validated fixed-depth incumbent at every trial."""
    path = definition.incumbent_events_file
    if not path.is_file():
        raise FileNotFoundError(
            f"Oracle policy requires the analytics cache {path}."
        )
    source_bytes = path.read_bytes()
    rows = list(csv.DictReader(source_bytes.decode("utf-8").splitlines()))
    depths = (5, 10, 15, 20, 30)
    by_depth: dict[int, list[tuple[int, float]]] = {depth: [] for depth in depths}
    for row in rows:
        if int(row["smac_seed"]) != smac_seed:
            continue
        depth = int(row["depth"])
        if depth in by_depth:
            by_depth[depth].append(
                (int(row["trial"]), float(row["full_training_par10"]))
            )
    missing = [depth for depth, events in by_depth.items() if not events]
    if missing:
        raise RuntimeError(
            f"Oracle source has no seed-{smac_seed} events for depths {missing}."
        )
    for events in by_depth.values():
        events.sort()

    depth_coverage = {
        str(depth): max(trial for trial, _ in events)
        for depth, events in by_depth.items()
    }
    common_trial_limit = min(depth_coverage.values())
    indices = {depth: 0 for depth in depths}
    current = {depth: math.inf for depth in depths}
    segments: list[dict[str, Any]] = []
    last_depth: int | None = None
    for trial in range(1, common_trial_limit + 1):
        for depth in depths:
            events = by_depth[depth]
            while (
                indices[depth] < len(events)
                and events[indices[depth]][0] <= trial
            ):
                current[depth] = events[indices[depth]][1]
                indices[depth] += 1
        best_depth = min(depths, key=lambda depth: (current[depth], depth))
        if best_depth != last_depth:
            segments.append(
                {
                    "trial": trial,
                    "depth": best_depth,
                    "validated_full_training_par10": current[best_depth],
                }
            )
            last_depth = best_depth

    return {
        "kind": "offline_oracle_from_fixed_depth_incumbents",
        "smac_seed": smac_seed,
        "source": str(path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "comparison_quantile_seed": 0,
        "depths": list(depths),
        "depth_coverage": depth_coverage,
        "common_trial_limit": common_trial_limit,
        "after_common_limit": "hold the final selected depth",
        "tie_break": "smaller depth",
        "segments": segments,
    }


def _rank_correlation_with_time(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    ranks[order] = np.arange(len(array), dtype=float)
    correlation = np.corrcoef(np.arange(len(array), dtype=float), ranks)[0, 1]
    return float(correlation) if np.isfinite(correlation) else 0.0


class AdaptiveDepthPolicyCallback(Callback):
    """Update the next RF fit's depth and durably record every decision."""

    def __init__(
        self,
        *,
        policy: str,
        output_directory: Path,
        model: RandomForest,
        benchmark: ACLibSurrogateBenchmark,
        training_instances: tuple[str, ...],
        oracle_schedule: dict[str, Any] | None,
        overwrite: bool,
    ) -> None:
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy!r}.")
        self.policy = policy
        self.output_directory = output_directory
        self.model = model
        self.benchmark = benchmark
        self.training_instances = training_instances
        self.oracle_schedule = oracle_schedule
        self.events_path = output_directory / "policy_events.jsonl"
        self.state_path = output_directory / "policy_state.json"
        self.policy_validation_evaluations = 0

        if overwrite:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.events_path.write_text("", encoding="utf-8")
            self.state = self._initial_state()
            self._save_state()
        elif self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("policy") != policy:
                raise RuntimeError("Persisted policy state has a different policy.")
            self.state.setdefault(
                "next_depth", int(self.state.get("current_depth", INITIAL_DEPTH))
            )
            self.state.setdefault("last_fitted_depth", None)
        else:
            self.state = self._initial_state()
            self._save_state()
        self._next_event_index = (
            sum(1 for _ in self.events_path.open("r", encoding="utf-8"))
            if self.events_path.exists()
            else 0
        )
        self._install_controlled_train()

    def _initial_state(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "current_depth": INITIAL_DEPTH,
            "next_depth": INITIAL_DEPTH,
            "last_fitted_depth": None,
            "last_observed_training_rows": None,
            "saturation_streak": 0,
            "rotation_low": 5,
            "rotation_high": 10,
            "calibration_window": [],
            "calibration_seen_fingerprints": [],
            "transitions": [
                {
                    "completed_trials": 0,
                    "depth": INITIAL_DEPTH,
                    "reason": "initial",
                }
            ],
        }

    def _install_controlled_train(self) -> None:
        original_train = self.model.train

        @functools.wraps(original_train)
        def controlled_train(*args: Any, **kwargs: Any) -> Any:
            depth = int(self.state["next_depth"])
            self.model._rf_opts["max_depth"] = depth
            result = original_train(*args, **kwargs)
            self.state["current_depth"] = depth
            self.state["last_fitted_depth"] = depth
            self._save_state()
            return result

        self.model.train = controlled_train

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = {
            "event_index": self._next_event_index,
            "policy": self.policy,
            **event,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_event_index += 1

    def _set_depth(
        self,
        config_selector: Any,
        depth: int,
        *,
        completed_trials: int,
        reason: str,
    ) -> None:
        depth = int(depth)
        previous = int(self.state["next_depth"])
        self.state["next_depth"] = depth
        if depth != previous:
            transition = {
                "completed_trials": completed_trials,
                "depth": depth,
                "previous_depth": previous,
                "reason": reason,
            }
            self.state["transitions"].append(transition)
            self._append_event({"event_type": "depth_change", **transition})
            print(
                f"[DepthPolicy] policy={self.policy} "
                f"trials={completed_trials} {previous}->{depth}: {reason}"
            )
        self._save_state()

    def _observe_new_fit(
        self,
        config_selector: Any,
        completed_trials: int,
    ) -> None:
        forest = getattr(self.model, "_rf", None)
        estimators = getattr(forest, "estimators_", None)
        training_rows = int(getattr(config_selector, "_previous_entries", -1))
        if (
            not estimators
            or training_rows < 0
            or training_rows == self.state["last_observed_training_rows"]
        ):
            return

        fitted_depth_limit = int(
            self.state.get("last_fitted_depth")
            or self.model._rf_opts["max_depth"]
        )
        depths = [int(tree.get_depth()) for tree in estimators]
        mean_depth = float(np.mean(depths))
        saturated = mean_depth >= SATURATION_FRACTION * fitted_depth_limit
        self.state["last_observed_training_rows"] = training_rows

        event = {
            "event_type": "fit_observation",
            "completed_trials": completed_trials,
            "training_rows": training_rows,
            "fitted_depth_limit": fitted_depth_limit,
            "actual_tree_depth_mean": mean_depth,
            "actual_tree_depth_min": min(depths),
            "actual_tree_depth_max": max(depths),
            "tree_count": len(depths),
            "saturation_threshold": SATURATION_FRACTION
            * fitted_depth_limit,
            "saturated": saturated,
        }

        if self.policy.startswith("saturation_k"):
            required = int(self.policy.rsplit("k", 1)[1])
            self.state["saturation_streak"] = (
                int(self.state["saturation_streak"]) + 1
                if saturated
                else 0
            )
            event["saturation_streak"] = self.state["saturation_streak"]
            event["required_streak"] = required
            self._append_event(event)
            if self.state["saturation_streak"] >= required:
                self.state["saturation_streak"] = 0
                self._set_depth(
                    config_selector,
                    fitted_depth_limit + DEPTH_INCREMENT,
                    completed_trials=completed_trials,
                    reason=(
                        f"mean actual depth reached {SATURATION_FRACTION:.1f} "
                        f"of the limit for {required} consecutive RF fits"
                    ),
                )
            else:
                self._save_state()
            return

        if self.policy == "rotating_saturation_k50":
            low = int(self.state["rotation_low"])
            high = int(self.state["rotation_high"])
            if fitted_depth_limit == high:
                self.state["saturation_streak"] = (
                    int(self.state["saturation_streak"]) + 1
                    if saturated
                    else 0
                )
            event.update(
                {
                    "rotation_low": low,
                    "rotation_high": high,
                    "high_depth_saturation_streak": self.state[
                        "saturation_streak"
                    ],
                    "required_streak": 50,
                }
            )
            self._append_event(event)
            if (
                fitted_depth_limit == high
                and self.state["saturation_streak"] >= 50
            ):
                self.state["rotation_low"] = high
                self.state["rotation_high"] = high + DEPTH_INCREMENT
                self.state["saturation_streak"] = 0
                next_depth = high + DEPTH_INCREMENT
                reason = (
                    f"higher rotating depth {high} was saturated for "
                    "50 consecutive high-depth RF fits"
                )
            else:
                next_depth = high if fitted_depth_limit == low else low
                reason = f"alternate rotating depths {low} and {high}"
            self._set_depth(
                config_selector,
                next_depth,
                completed_trials=completed_trials,
                reason=reason,
            )
            return

        self._append_event(event)
        self._save_state()

    def _oracle_depth(self, completed_trials: int) -> int:
        if self.oracle_schedule is None:
            raise RuntimeError("Oracle policy has no schedule.")
        trial = max(1, completed_trials)
        selected = int(self.oracle_schedule["segments"][0]["depth"])
        for segment in self.oracle_schedule["segments"]:
            if int(segment["trial"]) > trial:
                break
            selected = int(segment["depth"])
        return selected

    def on_next_configurations_start(self, config_selector: Any) -> None:
        completed_trials = len(config_selector._runhistory)
        self._observe_new_fit(config_selector, completed_trials)
        if self.policy == "oracle_incumbent_depth":
            desired = self._oracle_depth(completed_trials)
            self._set_depth(
                config_selector,
                desired,
                completed_trials=completed_trials,
                reason="seed-specific fixed-depth incumbent oracle schedule",
            )

    def _calibration_trend(self) -> dict[str, Any]:
        window = self.state["calibration_window"]
        ratios = [float(item["standardized_absolute_error"]) for item in window]
        logged = np.log1p(np.asarray(ratios, dtype=float))
        x = np.arange(len(logged), dtype=float)
        slope = float(np.polyfit(x, logged, 1)[0])
        fitted_growth = float(math.exp(slope * (len(logged) - 1)))
        rank_correlation = _rank_correlation_with_time(list(logged))
        midpoint = len(ratios) // 2
        first_median = float(np.median(ratios[:midpoint]))
        second_median = float(np.median(ratios[midpoint:]))
        trigger = (
            rank_correlation >= ERROR_TREND_MIN_RANK_CORRELATION
            and fitted_growth >= ERROR_TREND_MIN_FITTED_GROWTH
            and second_median > first_median
        )
        return {
            "window": len(ratios),
            "log_ratio_slope": slope,
            "fitted_growth_factor": fitted_growth,
            "rank_correlation_with_time": rank_correlation,
            "first_half_median": first_median,
            "second_half_median": second_median,
            "trigger": trigger,
        }

    def on_next_configurations_end(
        self,
        config_selector: Any,
        config: Configuration,
    ) -> None:
        if self.policy != "error_variance_trend_25":
            return
        forest = getattr(self.model, "_rf", None)
        if not getattr(forest, "estimators_", None):
            return
        fingerprint = _fingerprint(config)
        seen = set(self.state["calibration_seen_fingerprints"])
        if fingerprint in seen:
            return

        config_array = convert_configurations_to_array([config])
        means, variances = self.model.predict_marginalized(config_array)
        predicted = float(np.asarray(means).reshape(-1)[0])
        variance = max(0.0, float(np.asarray(variances).reshape(-1)[0]))
        costs = [
            self.benchmark.evaluate(config, instance, seed=0)[0]
            for instance in self.training_instances
        ]
        self.policy_validation_evaluations += len(costs)
        actual = float(np.mean(costs))
        absolute_error = abs(actual - predicted)
        standard_deviation_floor = 1e-8 * max(1.0, abs(predicted))
        predicted_standard_deviation = math.sqrt(variance)
        denominator = max(predicted_standard_deviation, standard_deviation_floor)
        ratio = absolute_error / denominator
        observation = {
            "configuration_fingerprint": fingerprint,
            "completed_trials": len(config_selector._runhistory),
            "depth": int(
                self.state.get("last_fitted_depth")
                or self.state["current_depth"]
            ),
            "predicted_full_training_par10": predicted,
            "prediction_variance": variance,
            "prediction_standard_deviation": predicted_standard_deviation,
            "actual_full_training_par10_qseed_0": actual,
            "absolute_error": absolute_error,
            "standardized_absolute_error": ratio,
        }
        self.state["calibration_seen_fingerprints"].append(fingerprint)
        self.state["calibration_window"].append(observation)
        self.state["calibration_window"] = self.state["calibration_window"][
            -ERROR_TREND_WINDOW:
        ]

        event: dict[str, Any] = {
            "event_type": "calibration_observation",
            **observation,
        }
        if len(self.state["calibration_window"]) == ERROR_TREND_WINDOW:
            trend = self._calibration_trend()
            event["trend"] = trend
            self._append_event(event)
            if trend["trigger"]:
                new_depth = int(self.state["next_depth"]) + DEPTH_INCREMENT
                self.state["calibration_window"] = []
                self._set_depth(
                    config_selector,
                    new_depth,
                    completed_trials=len(config_selector._runhistory),
                    reason=(
                        "25-proposal standardized-error trend: "
                        f"rank correlation={trend['rank_correlation_with_time']:.3f}, "
                        f"fitted growth={trend['fitted_growth_factor']:.3f}"
                    ),
                )
                return
        else:
            self._append_event(event)
        self._save_state()


def _serialize_trajectory(
    facade: AlgorithmConfigurationFacade,
) -> list[dict[str, Any]]:
    return [
        {
            "config_ids": [int(config_id) for config_id in item.config_ids],
            "costs": [float(cost) for cost in item.costs],
            "trial": int(item.trial),
            "walltime": float(item.walltime),
        }
        for item in facade.intensifier.trajectory
    ]


def _has_valid_resume_checkpoint(
    output_path: Path,
    *,
    policy: str,
) -> bool:
    """Validate every state component before allowing SMAC to continue."""
    required = (
        "scenario.json",
        "optimization.json",
        "runhistory.json",
        "intensifier.json",
        "policy_state.json",
    )
    present = {
        filename: (output_path / filename).exists()
        for filename in required
    }
    if not any(present.values()):
        return False
    if not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        raise RuntimeError(
            f"Partial resume checkpoint in {output_path}; missing {missing}. "
            "Archive the interrupted directory before restarting it."
        )

    try:
        payload = {
            filename: json.loads(
                (output_path / filename).read_text(encoding="utf-8")
            )
            for filename in required
        }
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Corrupted resume checkpoint in {output_path}. Archive the "
            "interrupted directory before restarting it."
        ) from error

    runhistory = payload["runhistory.json"]
    intensifier = payload["intensifier.json"]
    policy_state = payload["policy_state.json"]
    finished = int(runhistory.get("stats", {}).get("finished", -1))
    if finished < 1 or finished != len(runhistory.get("data", [])):
        raise RuntimeError(
            f"Inconsistent runhistory checkpoint in {output_path}."
        )
    config_ids = {int(value) for value in runhistory.get("configs", {})}
    referenced_ids = {
        int(value)
        for value in (
            list(intensifier.get("incumbent_ids", []))
            + list(intensifier.get("rejected_config_ids", []))
        )
    }
    for event in intensifier.get("trajectory", []):
        referenced_ids.update(
            int(value) for value in event["config_ids"]
        )
        if int(event["trial"]) > finished:
            raise RuntimeError(
                f"Intensifier trajectory is ahead of runhistory in "
                f"{output_path}."
            )
    if referenced_ids - config_ids:
        raise RuntimeError(
            f"Intensifier references absent configurations in {output_path}."
        )
    if policy_state.get("policy") != policy:
        raise RuntimeError(
            f"Persisted policy identity mismatch in {output_path}."
        )
    return True


def run_policy(
    *,
    definition: PolicyExperimentDefinition,
    policy: str,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if output_root is None:
        output_root = definition.output_root
    output_root = Path(output_root).resolve()
    output_path = run_directory(output_root, policy, smac_seed)

    with closing(_acquire_lock(output_path)):
        summary_path = output_path / "summary.json"
        completion_path = output_path / "completed.json"
        if completion_path.exists() and not overwrite:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            if (
                completion.get("state") == "complete"
                and completion.get("n_trials") == n_trials
                and summary_path.exists()
            ):
                print(f"Complete policy run found; skipping {output_path}")
                return json.loads(summary_path.read_text(encoding="utf-8"))

        spec = get_benchmark_spec(definition.benchmark_key)
        data = load_benchmark_data(spec)
        choice, choice_payload = load_initial_choice(
            definition.initial_choice_file
        )
        initial_config = resolve_initial_configuration(data, choice)
        training_instances = data.training_instances
        training_features = {
            instance: data.features[instance]
            for instance in training_instances
        }
        resume = (
            not overwrite
            and _has_valid_resume_checkpoint(
                output_path,
                policy=policy,
            )
        )
        oracle_schedule_path = output_path / "oracle_schedule.json"
        if policy == "oracle_incumbent_depth":
            if resume and oracle_schedule_path.exists():
                oracle_schedule = json.loads(
                    oracle_schedule_path.read_text(encoding="utf-8")
                )
            else:
                oracle_schedule = derive_oracle_schedule(
                    definition, smac_seed
                )
                _atomic_json(oracle_schedule_path, oracle_schedule)
        else:
            oracle_schedule = None
        initial_depth = (
            int(oracle_schedule["segments"][0]["depth"])
            if oracle_schedule is not None
            else INITIAL_DEPTH
        )
        scenario = Scenario(
            configspace=data.configspace,
            name=policy,
            output_directory=output_root,
            deterministic=True,
            objectives="PAR10",
            crash_cost=spec.timeout_cost,
            n_trials=n_trials,
            use_default_config=choice.kind == "default",
            instances=list(training_instances),
            instance_features=training_features,
            seed=smac_seed,
            n_workers=1,
        )
        if scenario.output_directory != output_path:
            raise RuntimeError(
                f"Unexpected output path {scenario.output_directory}; "
                f"expected {output_path}."
            )
        if choice.kind == "default":
            initial_design = AlgorithmConfigurationFacade.get_initial_design(
                scenario
            )
        else:
            initial_design = RandomInitialDesign(
                scenario,
                n_configs=0,
                additional_configs=[initial_config],
            )

        model = AlgorithmConfigurationFacade.get_model(
            scenario=scenario,
            n_trees=N_TREES,
            max_depth=initial_depth,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            pca_components=PCA_COMPONENTS,
        )
        random_design = AlgorithmConfigurationFacade.get_random_design(
            scenario,
            probability=RANDOM_DESIGN_PROBABILITY,
        )
        acquisition_function = TelemetryEI.from_smac_ei(
            AlgorithmConfigurationFacade.get_acquisition_function(scenario)
        )
        benchmark = ACLibSurrogateBenchmark(spec)

        def deterministic_target(
            config: Configuration,
            instance: str,
            seed: int = 0,
        ) -> tuple[float, dict[str, Any]]:
            del seed
            cost, info = benchmark.evaluate(config, instance, seed=0)
            info["fixed_target_quantile_seed"] = 0
            return cost, info

        policy_callback = AdaptiveDepthPolicyCallback(
            policy=policy,
            output_directory=output_path,
            model=model,
            benchmark=benchmark,
            training_instances=training_instances,
            oracle_schedule=oracle_schedule,
            overwrite=not resume,
        )
        telemetry = SurrogateTelemetryCallback(
            output_directory=output_path,
            model=model,
            acquisition_function=acquisition_function,
            overwrite=not resume,
        )
        facade = AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=deterministic_target,
            model=model,
            acquisition_function=acquisition_function,
            initial_design=initial_design,
            random_design=random_design,
            callbacks=[policy_callback, telemetry],
            overwrite=not resume,
        )

        rf_source = Path(inspect.getfile(RandomForest)).resolve()
        if LOCAL_SMAC_ROOT.resolve() not in rf_source.parents:
            raise RuntimeError(f"Expected local SMAC RF, found {rf_source}.")
        identity = {
            "experiment_version": EXPERIMENT_VERSION,
            "benchmark": spec.key,
            "display_name": spec.display_name,
            "initials": definition.initials,
            "policy": policy,
            "smac_seed": smac_seed,
            "n_trials": n_trials,
            "deterministic": True,
            "target_quantile_seed": 0,
            "n_training_instances": len(training_instances),
            "test_instances_used": 0,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "initial_configuration_choice": choice_payload,
            "smac_model": {
                "n_trees": N_TREES,
                "initial_max_depth": initial_depth,
                "min_samples_split": MIN_SAMPLES_SPLIT,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "pca_components": PCA_COMPONENTS,
            },
            "random_design_probability": RANDOM_DESIGN_PROBABILITY,
            "policy_parameters": {
                "depth_increment": DEPTH_INCREMENT,
                "saturation_fraction": SATURATION_FRACTION,
                "error_trend_window": ERROR_TREND_WINDOW,
                "error_trend_min_rank_correlation": (
                    ERROR_TREND_MIN_RANK_CORRELATION
                ),
                "error_trend_min_fitted_growth": (
                    ERROR_TREND_MIN_FITTED_GROWTH
                ),
            },
            "oracle_schedule": oracle_schedule,
            "local_smac_root": str(LOCAL_SMAC_ROOT),
            "local_random_forest_source": str(rf_source),
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "assets": asset_metadata(spec),
        }
        _atomic_json(output_path / "run_metadata.json", identity)
        _atomic_json(
            completion_path,
            {"state": "running", **identity},
        )

        started = time.time()
        incumbent = facade.optimize()
        walltime = time.time() - started
        telemetry_summary = telemetry.audit(facade.runhistory)
        _atomic_json(
            output_path / TELEMETRY_SUMMARY_FILENAME,
            telemetry_summary,
        )
        _atomic_json(
            output_path / "trajectory.json",
            _serialize_trajectory(facade),
        )
        incumbent_cost = float(
            facade.runhistory.average_cost(incumbent, normalize=False)
        )
        result = {
            **identity,
            "output_directory": str(output_path),
            "finished_trials": int(facade.runhistory.finished),
            "submitted_trials": int(facade.runhistory.submitted),
            "configurations": len(facade.runhistory._config_ids),
            "incumbent": dict(incumbent),
            "incumbent_cost_on_evaluated_instance_keys": incumbent_cost,
            "final_depth": int(policy_callback.state["next_depth"]),
            "final_fitted_depth": policy_callback.state[
                "last_fitted_depth"
            ],
            "next_requested_depth": int(
                policy_callback.state["next_depth"]
            ),
            "depth_transitions": policy_callback.state["transitions"],
            "policy_validation_evaluations_this_process": (
                policy_callback.policy_validation_evaluations
            ),
            "benchmark_evaluations_this_process": benchmark.evaluation_count,
            "walltime_seconds_this_process": walltime,
            "configuration_telemetry": telemetry_summary,
        }
        _atomic_json(summary_path, result)
        _atomic_json(
            output_path / "incumbent.json",
            {
                "configuration": dict(incumbent),
                "cost_on_evaluated_instance_keys": incumbent_cost,
            },
        )
        _atomic_json(
            completion_path,
            {"state": "complete", **identity},
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

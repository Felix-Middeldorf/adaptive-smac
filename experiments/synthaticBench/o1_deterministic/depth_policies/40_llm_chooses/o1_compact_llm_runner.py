"""Token-efficient direct-API RF policy for deterministic O1."""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import o1_llm_runner as base


POLICY_NAME = "openai_compact_llm_rf_policy"
POLICY_VERSION = 2
EXPERIMENT_VERSION = 4
SUMMARY_WINDOWS = 10
RECENT_RF_FITS = 50
OBJECTIVE_LOWER_BOUND = -100.0
OBJECTIVE_UPPER_BOUND = 100.0

RANGES: dict[str, tuple[float, float]] = {
    "n_trees": (1, 100),
    "max_depth": (1, 30),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10),
    "feature_ratio": (0.0, 1.0),
}


@dataclass(frozen=True)
class CompactRFSettings:
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    feature_ratio: float

    def __post_init__(self) -> None:
        integer_values = {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
        }
        for name, value in integer_values.items():
            lower, upper = RANGES[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer.")
            if not lower <= value <= upper:
                raise ValueError(
                    f"{name}={value} is outside [{int(lower)}, {int(upper)}]."
                )
        if not math.isfinite(self.feature_ratio):
            raise ValueError("feature_ratio must be finite.")
        if not 0.0 < self.feature_ratio <= 1.0:
            raise ValueError("feature_ratio must be in (0, 1].")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_ratio": self.feature_ratio,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "CompactRFSettings":
        return cls(
            n_trees=int(value["n_trees"]),
            max_depth=int(value["max_depth"]),
            min_samples_split=int(value["min_samples_split"]),
            min_samples_leaf=int(value["min_samples_leaf"]),
            feature_ratio=float(value["feature_ratio"]),
        )


DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "n_trees": {"type": "integer", "minimum": 1, "maximum": 100},
        "max_depth": {"type": "integer", "minimum": 1, "maximum": 30},
        "min_samples_split": {
            "type": "integer",
            "minimum": 2,
            "maximum": 10,
        },
        "min_samples_leaf": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
        },
        "feature_ratio": {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "maximum": 1.0,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": [
        "n_trees",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "feature_ratio",
        "confidence",
        "reason",
    ],
    "additionalProperties": False,
}


def validate_decision(
    payload: dict[str, Any],
) -> tuple[CompactRFSettings, dict[str, Any]]:
    if set(payload) != set(DECISION_SCHEMA["required"]):
        raise ValueError("LLM decision has missing or additional fields.")
    settings = CompactRFSettings.from_mapping(payload)
    confidence = float(payload["confidence"])
    reason = str(payload["reason"]).strip()
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if not reason or len(reason) > 500:
        raise ValueError("reason must contain 1 to 500 characters.")
    return settings, {
        **settings.to_dict(),
        "confidence": confidence,
        "reason": reason,
    }


def _short(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return float(f"{value:.6g}")


def _distribution(values: list[float | None]) -> dict[str, Any]:
    array = np.asarray(
        [value for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "q25": _short(np.quantile(array, 0.25)),
        "median": _short(np.median(array)),
        "mean": _short(np.mean(array)),
        "q75": _short(np.quantile(array, 0.75)),
    }


def _error_variance_correlation(
    errors: list[float | None], variances: list[float | None]
) -> float | None:
    pairs = [
        (float(error), float(variance))
        for error, variance in zip(errors, variances)
        if error is not None
        and variance is not None
        and math.isfinite(float(error))
        and math.isfinite(float(variance))
        and error >= 0
        and variance >= 0
    ]
    if len(pairs) < 3:
        return None
    x, y = np.asarray(pairs, dtype=float).T
    x = np.log1p(x)
    y = np.log1p(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return _short(np.corrcoef(x, y)[0, 1])


def _trial_diagnostics(
    runhistory: Any,
    telemetry_path: Path,
    checkpoint: int,
) -> list[dict[str, Any]]:
    trials = base.ordered_trials(runhistory)[:checkpoint]
    if len(trials) != checkpoint:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has only {len(trials)} completed trials."
        )
    telemetry = [
        record
        for record in base._load_jsonl(telemetry_path)
        if record.get("event_type") == "proposal"
    ]
    first_cost_by_config: dict[int, float] = {}
    costs_by_config: dict[int, list[float]] = defaultdict(list)
    rows = []
    for trial_number, (key, value) in enumerate(trials, 1):
        config_id = int(key.config_id)
        observed = float(np.asarray(value.cost).reshape(-1)[0])
        first_cost_by_config.setdefault(config_id, observed)
        config = runhistory.get_config(config_id)
        proposal = base._proposal_for_trial(
            telemetry,
            base.configuration_fingerprint(config),
            trial_number,
        )
        predicted = None if proposal is None else proposal.get(
            "predicted_marginal_mean"
        )
        variance = None if proposal is None else proposal.get(
            "predicted_marginal_variance"
        )
        ei = None if proposal is None else proposal.get("expected_improvement")
        proxy = first_cost_by_config[config_id]
        absolute_error = (
            None if predicted is None else abs(float(predicted) - proxy)
        )
        relative_error = (
            None
            if absolute_error is None
            else absolute_error / max(abs(proxy), base.RELATIVE_ERROR_FLOOR)
        )
        costs_by_config[config_id].append(observed)
        best_config_mean = min(
            float(np.mean(costs)) for costs in costs_by_config.values()
        )
        rows.append(
            {
                "trial": trial_number,
                "config_id": config_id,
                "observed_cost": observed,
                "expected_improvement": None if ei is None else float(ei),
                "predicted_mean": (
                    None if predicted is None else float(predicted)
                ),
                "prediction_variance": (
                    None if variance is None else float(variance)
                ),
                "absolute_proxy_error": absolute_error,
                "relative_proxy_error": relative_error,
                "best_config_mean_cost": best_config_mean,
            }
        )
    return rows


def _window_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries = np.linspace(0, len(rows), SUMMARY_WINDOWS + 1, dtype=int)
    seen_before: set[int] = set()
    output = []
    for index, (start, stop) in enumerate(
        zip(boundaries[:-1], boundaries[1:]), 1
    ):
        chunk = rows[start:stop]
        config_ids = {int(row["config_id"]) for row in chunk}
        best_before = (
            rows[start - 1]["best_config_mean_cost"] if start else None
        )
        best_after = chunk[-1]["best_config_mean_cost"]
        errors = [row["absolute_proxy_error"] for row in chunk]
        variances = [row["prediction_variance"] for row in chunk]
        output.append(
            {
                "window": index,
                "trials": [chunk[0]["trial"], chunk[-1]["trial"]],
                "trial_count": len(chunk),
                "unique_configurations": len(config_ids),
                "new_configurations": len(config_ids - seen_before),
                "acquisition_diagnostic_rows": sum(
                    row["predicted_mean"] is not None for row in chunk
                ),
                "best_config_mean_cost_at_end": _short(best_after),
                "best_cost_improvement_in_window": (
                    None
                    if best_before is None
                    else _short(best_before - best_after)
                ),
                "observed_cost": _distribution(
                    [row["observed_cost"] for row in chunk]
                ),
                "expected_improvement": _distribution(
                    [row["expected_improvement"] for row in chunk]
                ),
                "predicted_mean": _distribution(
                    [row["predicted_mean"] for row in chunk]
                ),
                "prediction_variance": _distribution(variances),
                "absolute_proxy_error": _distribution(errors),
                "relative_proxy_error": _distribution(
                    [row["relative_proxy_error"] for row in chunk]
                ),
                "log_error_variance_correlation": (
                    _error_variance_correlation(errors, variances)
                ),
            }
        )
        seen_before.update(config_ids)
    return output


def _rf_fit_summaries(
    fit_observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recent = fit_observations[-RECENT_RF_FITS:]
    if not recent:
        return []
    groups = np.array_split(np.arange(len(recent)), min(5, len(recent)))
    output = []
    for group in groups:
        chunk = [recent[int(index)] for index in group]
        output.append(
            {
                "fit_indices": [chunk[0]["fit_index"], chunk[-1]["fit_index"]],
                "fit_count": len(chunk),
                "training_rows": [
                    chunk[0]["training_rows"],
                    chunk[-1]["training_rows"],
                ],
                "settings": chunk[-1]["settings"],
                "actual_tree_depth_mean": _distribution(
                    [item["actual_tree_depth_mean"] for item in chunk]
                ),
                "depth_utilization": _distribution(
                    [item["depth_utilization"] for item in chunk]
                ),
            }
        )
    return output


def compact_summary(
    *,
    checkpoint: int,
    trigger_trial: int,
    runhistory: Any,
    telemetry_path: Path,
    current_settings: CompactRFSettings,
    decisions: dict[str, Any],
    fit_observations: list[dict[str, Any]],
    objective_dimension: int = base.DIMENSION,
) -> dict[str, Any]:
    rows = _trial_diagnostics(runhistory, telemetry_path, checkpoint)
    evaluation_counts = Counter(row["config_id"] for row in rows)
    return {
        "summary_version": 1,
        "checkpoint": checkpoint,
        "actual_completed_trials_at_call": trigger_trial,
        "objective_direction": "minimize",
        "optimization_search_space": {
            "total_dimensions": objective_dimension,
            "parameter_type": "continuous real-valued",
            "compact_constraint": (
                f"{OBJECTIVE_LOWER_BOUND:g} <= x_i <= "
                f"{OBJECTIVE_UPPER_BOUND:g} for i=0,...,{objective_dimension - 1}"
            ),
            "parameters": [
                {
                    "name": f"x_{index}",
                    "lower_bound": OBJECTIVE_LOWER_BOUND,
                    "upper_bound": OBJECTIVE_UPPER_BOUND,
                }
                for index in range(objective_dimension)
            ],
            "conditionals": False,
            "forbidden_combinations": False,
        },
        "summary_rule": (
            f"All trials 1-{checkpoint} aggregated into "
            f"{SUMMARY_WINDOWS} consecutive equal-width windows; no raw "
            "configuration coordinates are included."
        ),
        "current_rf_settings": current_settings.to_dict(),
        "allowed_next_settings": {
            "n_trees": "integer in [1,100]",
            "max_depth": "integer in [1,30]",
            "min_samples_split": "integer in [2,10]",
            "min_samples_leaf": "integer in [1,10]",
            "feature_ratio": "number in (0,1]",
        },
        "previous_decisions": [
            {
                "checkpoint": int(key),
                "settings": value["settings"],
                "confidence": value["confidence"],
            }
            for key, value in sorted(
                decisions.items(), key=lambda item: int(item[0])
            )
        ],
        "evaluation_allocation": {
            "completed_trials": len(rows),
            "unique_configurations": len(evaluation_counts),
            "evaluations_per_configuration": _distribution(
                list(evaluation_counts.values())
            ),
            "final_best_config_mean_cost": _short(
                rows[-1]["best_config_mean_cost"]
            ),
        },
        "trial_windows": _window_summaries(rows),
        "recent_rf_fit_windows": _rf_fit_summaries(fit_observations),
    }


def decision_prompt(summary: dict[str, Any]) -> str:
    prompt = f"""You choose random-forest surrogate hyperparameters for the
next phase of a SMAC algorithm-configuration run. Lower objective values are
better. The compact data explicitly describe the total dimensionality, names,
types, and bounds of the underlying objective-function parameters. They also
summarize optimization progress,
expected improvement, marginalized RF predictions, prediction variance,
first-instance proxy errors, evaluation allocation, and actual fitted-tree
depth. Select the allowed values you expect to give the best SMAC optimization
performance in the remaining trials. Base the choice on the observed trends
and current settings. Return only the structured object required by the schema.

Important definitions:
- absolute_proxy_error compares the proposal-time marginalized prediction with
  the configuration's first observed instance cost; it adds no evaluations.
- relative_proxy_error divides that error by max(abs(proxy), 1e-12).
- log_error_variance_correlation is the Pearson correlation between
  log1p(absolute error) and log1p(prediction variance).
- best_config_mean_cost is the lowest running mean observed cost among all
  configurations using only their evaluations available by that trial.
- depth_utilization is average actual tree depth divided by its depth limit.

COMPACT DATA
{json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)}
"""
    forbidden = ("synthactic", "o1-deterministic", "benchmark_seed")
    if any(value in prompt.lower() for value in forbidden):
        raise RuntimeError("The compact LLM prompt leaks workload identity.")
    return prompt


class CompactLLMRFPolicyCallback(base.LLMRFPolicyCallback):
    def __init__(
        self,
        *,
        objective_dimension: int = base.DIMENSION,
        **kwargs: Any,
    ) -> None:
        self.objective_dimension = objective_dimension
        super().__init__(
            **kwargs,
            settings_class=CompactRFSettings,
            decision_validator=validate_decision,
            prompt_builder=decision_prompt,
            policy_version=POLICY_VERSION,
        )

    def _summary(
        self,
        checkpoint: int,
        trigger_trial: int,
        runhistory: Any,
    ) -> dict[str, Any]:
        return compact_summary(
            checkpoint=checkpoint,
            trigger_trial=trigger_trial,
            runhistory=runhistory,
            telemetry_path=self.telemetry_path,
            current_settings=self.next_settings,
            decisions=self.state["decisions"],
            fit_observations=self.state["fit_observations"],
            objective_dimension=self.objective_dimension,
        )


def run_compact_llm_policy(
    benchmark_seed: int,
    smac_seed: int,
    *,
    n_trials: int = base.N_TRIALS,
    output_root: Path = base.OUTPUT_ROOT,
    overwrite: bool = False,
    decision_provider: Callable[
        [str], tuple[dict[str, Any], dict[str, Any]]
    ]
    | None = None,
    dimension: int = base.DIMENSION,
    n_instances: int = base.N_INSTANCES,
    instance_seed: int = base.INSTANCE_SEED,
    policy_name: str = POLICY_NAME,
) -> dict[str, Any]:
    provider = decision_provider
    if provider is None:
        provider = base.OpenAIResponsesClient(
            decision_schema=DECISION_SCHEMA,
            schema_name="smac_compact_rf_hyperparameters",
        ).invoke
    return base.run_llm_policy(
        benchmark_seed,
        smac_seed,
        n_trials=n_trials,
        output_root=output_root,
        overwrite=overwrite,
        decision_provider=provider,
        policy_name=policy_name,
        experiment_version=EXPERIMENT_VERSION,
        callback_factory=functools.partial(
            CompactLLMRFPolicyCallback,
            objective_dimension=dimension,
        ),
        identity_extra={
            "summary_mode": "ten_window_compact_aggregates",
            "summary_windows": SUMMARY_WINDOWS,
            "recent_rf_fits_summarized": RECENT_RF_FITS,
            "diagnostic_sample_size": None,
            "allowed_rf_ranges": {
                key: list(value) for key, value in RANGES.items()
            },
        },
        dimension=dimension,
        n_instances=n_instances,
        instance_seed=instance_seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-seed",
        type=int,
        choices=base.BENCHMARK_SEEDS,
        required=True,
    )
    parser.add_argument(
        "--smac-seed", type=int, choices=base.SMAC_SEEDS, required=True
    )
    parser.add_argument("--n-trials", type=int, default=base.N_TRIALS)
    parser.add_argument("--output-root", type=Path, default=base.OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_compact_llm_policy(
        args.benchmark_seed,
        args.smac_seed,
        n_trials=args.n_trials,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

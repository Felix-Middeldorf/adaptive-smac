"""Mistral RF-policy reruns for the deterministic O1 dimension study."""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai


HERE = Path(__file__).resolve().parent
SHARED_POLICY_CODE = HERE.parent / "40_llm_chooses"
if str(SHARED_POLICY_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_POLICY_CODE))

import o1_compact_llm_runner as compact
import o1_llm_runner as base


BENCHMARK_SEED = 40
DIMENSIONS = (10, 25, 50, 100)
SMAC_SEEDS = tuple(range(5))
N_INSTANCES = 10
N_TRIALS = 1_000
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
FIXED_N_TREES = 100
MISTRAL_MODEL = "mistralai-mistral-small-4-119b"
RWTHGPT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
RWTHGPT_API_KEY_FILE = Path.home() / ".config/kiconnect/rwthgpt_api_key"
MAX_INPUT_TOKENS = 200_000
RATE_LIMIT_WAIT_SECONDS = 15

COMPACT_POLICY_NAME = "rwthgpt_mistral_compact_fixed_100_trees"
EVERY_SECOND_POLICY_NAME = "rwthgpt_mistral_every_second_fixed_100_trees"
EXPERIMENT_VERSION = 1
POLICY_VERSION = 1
OUTPUT_ROOT = HERE / "results"

INITIAL_SETTINGS = compact.CompactRFSettings(
    n_trees=FIXED_N_TREES,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=3,
    feature_ratio=5.0 / 6.0,
)
RAW_DECISION_FIELDS = {
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "feature_ratio",
    "confidence",
    "reason",
}
CACHED_DECISION_FIELDS = RAW_DECISION_FIELDS | {"n_trees"}


def dimension_root(dimension: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return Path(output_root) / f"dimension_{dimension}"


def _model_dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _json_from_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("RWTHGPT response must be a JSON object.")
    return payload


class RWTHGPTMistralClient:
    """OpenAI-compatible RWTHGPT client with deliberate 429 retry behaviour."""

    def invoke(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        token_estimate = math.ceil(len(prompt) / 4)
        if token_estimate > MAX_INPUT_TOKENS:
            raise RuntimeError(
                f"Prompt estimate {token_estimate} exceeds {MAX_INPUT_TOKENS}."
            )
        try:
            api_key = RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(
                f"Could not read RWTHGPT key from {RWTHGPT_API_KEY_FILE}."
            ) from error
        if not api_key:
            raise RuntimeError("RWTHGPT API-key file is empty.")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=RWTHGPT_BASE_URL,
            timeout=300.0,
            max_retries=0,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You select random-forest surrogate hyperparameters for SMAC. "
                    "Return only the requested JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        started = time.perf_counter()
        attempts = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=MISTRAL_MODEL,
                    messages=messages,
                )
            except openai.RateLimitError:
                attempts += 1
                print(
                    f"RWTHGPT request denied for {MISTRAL_MODEL}; waiting "
                    f"{RATE_LIMIT_WAIT_SECONDS} seconds before retry {attempts}."
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            content = response.choices[0].message.content
            if not content:
                print("RWTHGPT returned an empty completion; retrying in 15 seconds.")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                continue
            try:
                payload = _json_from_content(content)
                # Validate here as well as in the callback, so a malformed model
                # completion is retried instead of terminating a SMAC run.
                validate_fixed_tree_decision(payload)
                break
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                print(
                    f"RWTHGPT returned an invalid decision ({error}); retrying "
                    f"in {RATE_LIMIT_WAIT_SECONDS} seconds."
                )
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
        return payload, {
            "response_id": response.id,
            "model_requested": MISTRAL_MODEL,
            "model_returned": response.model,
            "elapsed_seconds": time.perf_counter() - started,
            "usage": _model_dump(response.usage),
            "input_token_estimate": token_estimate,
            "rate_limit_retries": attempts,
            "api_base_url": RWTHGPT_BASE_URL,
        }


def validate_fixed_tree_decision(
    payload: dict[str, Any],
) -> tuple[compact.CompactRFSettings, dict[str, Any]]:
    fields = set(payload)
    if fields == CACHED_DECISION_FIELDS:
        if int(payload["n_trees"]) != FIXED_N_TREES:
            raise ValueError(f"Cached n_trees must remain {FIXED_N_TREES}.")
    elif fields != RAW_DECISION_FIELDS:
        raise ValueError("The Mistral decision has missing or additional fields.")
    settings = compact.CompactRFSettings(
        n_trees=FIXED_N_TREES,
        max_depth=int(payload["max_depth"]),
        min_samples_split=int(payload["min_samples_split"]),
        min_samples_leaf=int(payload["min_samples_leaf"]),
        feature_ratio=float(payload["feature_ratio"]),
    )
    confidence = float(payload["confidence"])
    reason = str(payload["reason"]).strip()
    if not 0.0 <= confidence <= 1.0 or not reason or len(reason) > 500:
        raise ValueError("Invalid confidence or reason in the Mistral decision.")
    return settings, {**settings.to_dict(), "confidence": confidence, "reason": reason}


def _prompt(summary: dict[str, Any], mode: str) -> str:
    description = (
        "The data are ten aggregate windows of the completed runhistory."
        if mode == "compact"
        else "The data include every second completed trial with its configuration, "
        "observed cost, and proposal-time SMAC diagnostics."
    )
    return f"""Choose random-forest surrogate hyperparameters for the next phase of a
SMAC algorithm-configuration run. Lower objective values are better. The
number of trees is fixed to {FIXED_N_TREES} and must not be proposed or changed.
{description}

Choose only from:
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: number in (0, 1]

Return exactly one JSON object with these keys and no others:
{{"max_depth": integer, "min_samples_split": integer,
  "min_samples_leaf": integer, "feature_ratio": number,
  "confidence": number from 0 to 1, "reason": concise string}}

Definitions: expected_improvement, predicted_mean, and prediction_variance are
recorded at proposal time. absolute_proxy_error compares the proposal-time
prediction with the first observed instance cost for that configuration.
best_config_mean_cost is the best running mean over configurations.

DATA
{json.dumps(summary, sort_keys=True, separators=(',', ':'), allow_nan=False)}
"""


def _fixed_tree_compact_summary(
    checkpoint: int,
    trigger_trial: int,
    runhistory: Any,
    callback: "MistralFixedTreeCallback",
) -> dict[str, Any]:
    summary = compact.compact_summary(
        checkpoint=checkpoint,
        trigger_trial=trigger_trial,
        runhistory=runhistory,
        telemetry_path=callback.telemetry_path,
        current_settings=callback.next_settings,
        decisions=callback.state["decisions"],
        fit_observations=callback.state["fit_observations"],
        objective_dimension=callback.dimension,
    )
    summary["fixed_rf_hyperparameters"] = {"n_trees": FIXED_N_TREES}
    summary["allowed_next_settings"].pop("n_trees")
    return summary


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else float(f"{value:.8g}")


def _every_second_summary(
    checkpoint: int,
    trigger_trial: int,
    runhistory: Any,
    callback: "MistralFixedTreeCallback",
) -> dict[str, Any]:
    diagnostics = compact._trial_diagnostics(
        runhistory, callback.telemetry_path, checkpoint
    )
    trials = base.ordered_trials(runhistory)[:checkpoint]
    records = []
    for number, ((key, _), diagnostic) in enumerate(zip(trials, diagnostics), 1):
        if number % 2:
            continue
        config = runhistory.get_config(int(key.config_id))
        records.append(
            {
                "trial": number,
                "config_id": int(key.config_id),
                "instance": str(key.instance),
                "configuration": {
                    str(name): _number(value) for name, value in dict(config).items()
                },
                **{
                    name: _number(diagnostic[name])
                    for name in (
                        "observed_cost",
                        "expected_improvement",
                        "predicted_mean",
                        "prediction_variance",
                        "absolute_proxy_error",
                        "relative_proxy_error",
                        "best_config_mean_cost",
                    )
                },
            }
        )
    return {
        "summary_version": 1,
        "summary_mode": "every_second_completed_trial_with_configuration_and_smac_diagnostics",
        "checkpoint": checkpoint,
        "actual_completed_trials_at_call": trigger_trial,
        "total_trial_budget": N_TRIALS,
        "remaining_trials": N_TRIALS - checkpoint,
        "objective_direction": "minimize",
        "configuration_space": {
            "total_dimensions": callback.dimension,
            "parameter_type": "continuous real-valued",
            "bounds": [-100.0, 100.0],
            "hierarchy": "none",
        },
        "fixed_rf_hyperparameters": {"n_trees": FIXED_N_TREES},
        "current_rf_settings": callback.next_settings.to_dict(),
        "allowed_next_settings": {
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
            for key, value in sorted(callback.state["decisions"].items())
        ],
        "every_second_trial_records": records,
    }


class MistralFixedTreeCallback(base.LLMRFPolicyCallback):
    def __init__(self, *, summary_mode: str, dimension: int, **kwargs: Any) -> None:
        self.summary_mode = summary_mode
        self.dimension = dimension
        super().__init__(
            settings_class=compact.CompactRFSettings,
            decision_validator=validate_fixed_tree_decision,
            prompt_builder=lambda summary: _prompt(summary, summary_mode),
            policy_version=POLICY_VERSION,
            **kwargs,
        )

    def _summary(
        self, checkpoint: int, trigger_trial: int, runhistory: Any
    ) -> dict[str, Any]:
        if self.summary_mode == "compact":
            return _fixed_tree_compact_summary(checkpoint, trigger_trial, runhistory, self)
        return _every_second_summary(checkpoint, trigger_trial, runhistory, self)

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        audit = super().audit(n_trials=n_trials)
        audit.update(
            {
                "model": MISTRAL_MODEL,
                "api": "RWTHGPT OpenAI-compatible chat completions",
                "api_key_source": str(RWTHGPT_API_KEY_FILE),
                "fixed_n_trees": FIXED_N_TREES,
                "summary_mode": self.summary_mode,
            }
        )
        return audit


def _run(
    dimension: int,
    smac_seed: int,
    *,
    policy_name: str,
    summary_mode: str,
    decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    provider = decision_provider or RWTHGPTMistralClient().invoke
    return base.run_llm_policy(
        BENCHMARK_SEED,
        smac_seed,
        n_trials=n_trials,
        output_root=dimension_root(dimension, output_root),
        decision_provider=provider,
        policy_name=policy_name,
        experiment_version=EXPERIMENT_VERSION,
        callback_factory=functools.partial(
            MistralFixedTreeCallback, summary_mode=summary_mode, dimension=dimension
        ),
        identity_extra={
            "llm_provider": "rwthgpt",
            "rwthgpt_model": MISTRAL_MODEL,
            "openai_model": MISTRAL_MODEL,
            "openai_reasoning_effort": None,
            "summary_mode": summary_mode,
            "fixed_n_trees": FIXED_N_TREES,
            "model_selectable_hyperparameters": [
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "feature_ratio",
            ],
        },
        dimension=dimension,
        n_instances=N_INSTANCES,
        instance_seed=INSTANCE_SEED,
        initial_settings=INITIAL_SETTINGS,
    )


def run_mistral_compact(dimension: int, smac_seed: int) -> dict[str, Any]:
    return _run(
        dimension,
        smac_seed,
        policy_name=COMPACT_POLICY_NAME,
        summary_mode="ten_window_compact_aggregates",
    )


def run_mistral_every_second(dimension: int, smac_seed: int) -> dict[str, Any]:
    return _run(
        dimension,
        smac_seed,
        policy_name=EVERY_SECOND_POLICY_NAME,
        summary_mode="every_second_completed_trial_with_configuration_and_smac_diagnostics",
    )


def jobs() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (kind, dimension, seed)
        for kind in ("compact", "every_second")
        for dimension in DIMENSIONS
        for seed in SMAC_SEEDS
    )

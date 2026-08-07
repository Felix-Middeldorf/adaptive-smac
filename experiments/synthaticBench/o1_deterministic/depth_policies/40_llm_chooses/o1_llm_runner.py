"""Direct-API LLM policy for SMAC's RF on deterministic O1."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback
from smac.runhistory.enumerations import StatusType
from smac.utils.configspace import convert_configurations_to_array


CHECKPOINTS = (100, 250, 500)
N_TRIALS = 1_000
BENCHMARK_SEEDS = (40, 42)
SMAC_SEEDS = tuple(range(5))
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
INSTANCE_STD = 2.0
RANDOM_DESIGN_PROBABILITY = 0.0
PCA_COMPONENTS = 4
EXPERIMENT_VERSION = 2
POLICY_VERSION = 2
OPENAI_MODEL = "gpt-5.6-terra"
OPENAI_REASONING_EFFORT = "medium"
OPENAI_TIMEOUT_SECONDS = 120
OPENAI_MAX_ATTEMPTS = 3
DIAGNOSTIC_SAMPLE_SIZE = 100
RELATIVE_ERROR_FLOOR = 1e-12

TREE_CHOICES = (10, 50, 100)
DEPTH_CHOICES = (5, 10, 15, 20, 30)
SPLIT_CHOICES = (2, 3, 4, 8)
LEAF_CHOICES = (1, 3, 5)
FEATURE_RATIO_CHOICES = (0.3, 0.5, 5.0 / 6.0, 1.0)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_ROOT = HERE / "smac_output"
POLICY_NAME = "openai_llm_rf_policy"
POLICY_STATE_FILENAME = "llm_policy_state.json"
POLICY_EVENTS_FILENAME = "llm_policy_events.jsonl"
TELEMETRY_FILENAME = "proposal_telemetry.jsonl"
REQUEST_DIRECTORY_NAME = "llm_requests"


@dataclass(frozen=True)
class RFSettings:
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    feature_ratio: float

    def __post_init__(self) -> None:
        allowed = {
            "n_trees": (self.n_trees, TREE_CHOICES),
            "max_depth": (self.max_depth, DEPTH_CHOICES),
            "min_samples_split": (
                self.min_samples_split,
                SPLIT_CHOICES,
            ),
            "min_samples_leaf": (self.min_samples_leaf, LEAF_CHOICES),
            "feature_ratio": (self.feature_ratio, FEATURE_RATIO_CHOICES),
        }
        for name, (value, choices) in allowed.items():
            if value not in choices:
                raise ValueError(f"Unsupported {name}={value}; choose from {choices}.")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_ratio": self.feature_ratio,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RFSettings":
        return cls(
            n_trees=int(value["n_trees"]),
            max_depth=int(value["max_depth"]),
            min_samples_split=int(value["min_samples_split"]),
            min_samples_leaf=int(value["min_samples_leaf"]),
            feature_ratio=float(value["feature_ratio"]),
        )


# Exact defaults of AlgorithmConfigurationFacade.get_model in local SMAC.
DEFAULT_SETTINGS = RFSettings(
    n_trees=10,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=3,
    feature_ratio=5.0 / 6.0,
)

DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "n_trees": {"type": "integer", "enum": list(TREE_CHOICES)},
        "max_depth": {"type": "integer", "enum": list(DEPTH_CHOICES)},
        "min_samples_split": {
            "type": "integer",
            "enum": list(SPLIT_CHOICES),
        },
        "min_samples_leaf": {
            "type": "integer",
            "enum": list(LEAF_CHOICES),
        },
        "feature_ratio": {
            "type": "number",
            "enum": list(FEATURE_RATIO_CHOICES),
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string", "minLength": 1},
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


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, default=_json_default))
        handle.write("\n")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number != len(lines):
                raise
    return records


def configuration_fingerprint(config: Any) -> str:
    serialized = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def make_instance_map(
    n_instances: int = N_INSTANCES,
    instance_seed: int = INSTANCE_SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(instance_seed)
    return {
        f"i{index}": float(value)
        for index, value in enumerate(
            rng.normal(0, INSTANCE_STD, n_instances)
        )
    }


def output_directory(
    output_root: Path,
    benchmark_seed: int,
    smac_seed: int,
    policy_name: str = POLICY_NAME,
) -> Path:
    return (
        output_root
        / f"benchmark_seed_{benchmark_seed}"
        / policy_name
        / str(smac_seed)
    )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        (
            (key, value)
            for key, value in runhistory.items()
            if value.status != StatusType.RUNNING
        ),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def evenly_spaced_trial_numbers(
    completed_trials: int,
    sample_size: int = DIAGNOSTIC_SAMPLE_SIZE,
) -> list[int]:
    if completed_trials < sample_size:
        raise ValueError(
            f"Need at least {sample_size} trials, got {completed_trials}."
        )
    # ceil(i*n/k), i=1..k: exactly k unique positions including n. At n=500,
    # this is precisely 5, 10, ..., 500.
    return [
        (index * completed_trials + sample_size - 1) // sample_size
        for index in range(1, sample_size + 1)
    ]


def _proposal_for_trial(
    records: list[dict[str, Any]],
    fingerprint: str,
    trial_number: int,
) -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record.get("configuration_fingerprint") == fingerprint
        and int(record["completed_trials_before_proposal"])
        <= trial_number - 1
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda record: (
            int(record["completed_trials_before_proposal"]),
            int(record["proposal_index"]),
        ),
    )


def sampled_runhistory_diagnostics(
    runhistory: Any,
    telemetry_path: Path,
    checkpoint: int,
) -> list[dict[str, Any]]:
    trials = ordered_trials(runhistory)
    if len(trials) < checkpoint:
        raise RuntimeError(
            f"Checkpoint {checkpoint} requested with only {len(trials)} trials."
        )
    trials = trials[:checkpoint]
    sampled_numbers = evenly_spaced_trial_numbers(checkpoint)
    telemetry = [
        record
        for record in _load_jsonl(telemetry_path)
        if record.get("event_type") == "proposal"
    ]
    first_cost_by_config: dict[int, float] = {}
    for key, value in trials:
        first_cost_by_config.setdefault(
            int(key.config_id),
            float(np.asarray(value.cost).reshape(-1)[0]),
        )

    output = []
    for trial_number in sampled_numbers:
        key, value = trials[trial_number - 1]
        config_id = int(key.config_id)
        config = runhistory.get_config(config_id)
        fingerprint = configuration_fingerprint(config)
        proposal = _proposal_for_trial(
            telemetry, fingerprint, trial_number
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
            else absolute_error / max(abs(proxy), RELATIVE_ERROR_FLOOR)
        )
        output.append(
            {
                "trial_number": trial_number,
                "configuration_values": {
                    str(name): value for name, value in dict(config).items()
                },
                "instance": str(key.instance),
                "status": value.status.name,
                "observed_trial_cost": float(
                    np.asarray(value.cost).reshape(-1)[0]
                ),
                "proposal_completed_trials": (
                    None
                    if proposal is None
                    else int(proposal["completed_trials_before_proposal"])
                ),
                "expected_improvement": (
                    None if ei is None else float(ei)
                ),
                "predicted_mean_over_instances": (
                    None if predicted is None else float(predicted)
                ),
                "rf_prediction_variance": (
                    None if variance is None else float(variance)
                ),
                "first_evaluated_instance_cost_proxy": proxy,
                "absolute_proxy_error": absolute_error,
                "relative_proxy_error": relative_error,
            }
        )
    if len(output) != DIAGNOSTIC_SAMPLE_SIZE:
        raise RuntimeError("Did not construct exactly 100 diagnostic trials.")
    return output


def validate_decision(
    payload: dict[str, Any],
) -> tuple[RFSettings, dict[str, Any]]:
    if set(payload) != set(DECISION_SCHEMA["required"]):
        raise ValueError("LLM decision has missing or additional fields.")
    settings = RFSettings.from_mapping(payload)
    confidence = float(payload["confidence"])
    reason = str(payload["reason"]).strip()
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")
    if not reason:
        raise ValueError("reason must not be empty.")
    return settings, {
        **settings.to_dict(),
        "confidence": confidence,
        "reason": reason,
    }


def decision_prompt(summary: dict[str, Any]) -> str:
    prompt = f"""You choose random-forest surrogate hyperparameters for the
next phase of a SMAC algorithm-configuration run. Lower objective values are
better. Analyze the 100 evenly spaced completed trials below, emphasizing how
prediction errors, RF variances, expected improvement, evaluation allocation,
and optimization progress change with trial number and the current RF settings.
Choose one allowed setting for every hyperparameter that you expect to produce
the best SMAC optimization performance in the remaining trials. Return only
the structured object required by the response schema.

Metric definitions:
- predicted_mean_over_instances is the marginalized RF prediction stored when
  the configuration was proposed.
- first_evaluated_instance_cost_proxy is that configuration's first observed
  target cost and serves only as a cheap real-performance proxy.
- absolute_proxy_error = abs(predicted_mean_over_instances - proxy).
- relative_proxy_error = absolute_proxy_error / max(abs(proxy), 1e-12).
- rf_prediction_variance is the RF variance for the marginalized prediction.
- expected_improvement is the acquisition value at proposal time; higher is
  preferred by the acquisition maximizer.
- null proposal diagnostics mean that configuration was not proposed by a
  trained acquisition model, for example during initial design.

DATA
{json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)}
"""
    forbidden = ("synthactic", "o1-deterministic", "benchmark_seed")
    if any(value in prompt.lower() for value in forbidden):
        raise RuntimeError("The LLM prompt leaks workload identity.")
    return prompt


class OpenAIResponsesClient:
    """Minimal direct Responses API client with strict structured output."""

    endpoint = "https://api.openai.com/v1/responses"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        decision_schema: dict[str, Any] | None = None,
        schema_name: str = "smac_rf_hyperparameters",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.decision_schema = decision_schema or DECISION_SCHEMA
        self.schema_name = schema_name

    @staticmethod
    def _extract_output_text(response: dict[str, Any]) -> str:
        texts = [
            content["text"]
            for item in response.get("output", [])
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        ]
        if not texts:
            refusals = [
                content.get("refusal")
                for item in response.get("output", [])
                for content in item.get("content", [])
                if content.get("type") == "refusal"
            ]
            raise RuntimeError(
                f"OpenAI response contained no output text; refusals={refusals}."
            )
        return "".join(texts)

    def invoke(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        body = {
            "model": OPENAI_MODEL,
            "input": prompt,
            "reasoning": {"effort": OPENAI_REASONING_EFFORT},
            "max_output_tokens": 1_000,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": self.schema_name,
                    "strict": True,
                    "schema": self.decision_schema,
                }
            },
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "adaptive-smac-llm-policy/1",
            },
            method="POST",
        )
        started = time.perf_counter()
        response = None
        for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=OPENAI_TIMEOUT_SECONDS
                ) as handle:
                    response = json.load(handle)
                break
            except urllib.error.HTTPError as error:
                body_text = error.read().decode("utf-8", errors="replace")
                retryable = error.code == 429 or 500 <= error.code < 600
                if not retryable or attempt == OPENAI_MAX_ATTEMPTS:
                    raise RuntimeError(
                        f"OpenAI API HTTP {error.code}: {body_text[-2_000:]}"
                    ) from error
                time.sleep(2**attempt)
            except urllib.error.URLError as error:
                if attempt == OPENAI_MAX_ATTEMPTS:
                    raise RuntimeError(f"OpenAI API network error: {error}") from error
                time.sleep(2**attempt)
        if response is None:
            raise RuntimeError("OpenAI API returned no response.")
        elapsed = time.perf_counter() - started
        if response.get("status") != "completed":
            raise RuntimeError(
                f"OpenAI response status is {response.get('status')!r}: "
                f"{response.get('incomplete_details')!r}"
            )
        payload = json.loads(self._extract_output_text(response))
        metadata = {
            "response_id": response.get("id"),
            "model_requested": OPENAI_MODEL,
            "model_returned": response.get("model"),
            "status": response.get("status"),
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "elapsed_seconds": elapsed,
            "usage": response.get("usage"),
            "store": False,
        }
        return payload, metadata


class LLMRFPolicyCallback(Callback):
    """Record proposal-time diagnostics and apply API-selected RF settings."""

    def __init__(
        self,
        *,
        output_path: Path,
        model: Any,
        decision_provider: Callable[
            [str], tuple[dict[str, Any], dict[str, Any]]
        ],
        overwrite: bool,
        settings_class: type[RFSettings] = RFSettings,
        decision_validator: Callable[
            [dict[str, Any]], tuple[Any, dict[str, Any]]
        ] = validate_decision,
        prompt_builder: Callable[[dict[str, Any]], str] = decision_prompt,
        policy_version: int = POLICY_VERSION,
        initial_settings: Any = DEFAULT_SETTINGS,
    ) -> None:
        super().__init__()
        self.output_path = output_path
        self.model = model
        self.decision_provider = decision_provider
        self.settings_class = settings_class
        self.decision_validator = decision_validator
        self.prompt_builder = prompt_builder
        self.policy_version = policy_version
        self.initial_settings = initial_settings
        self.state_path = output_path / POLICY_STATE_FILENAME
        self.events_path = output_path / POLICY_EVENTS_FILENAME
        self.telemetry_path = output_path / TELEMETRY_FILENAME
        self.requests_path = output_path / REQUEST_DIRECTORY_NAME
        if overwrite:
            output_path.mkdir(parents=True, exist_ok=True)
            self.events_path.write_text("", encoding="utf-8")
            self.telemetry_path.write_text("", encoding="utf-8")
            self.state = {
                "policy_version": self.policy_version,
                "current_settings": self.initial_settings.to_dict(),
                "decisions": {},
                "fit_observations": [],
                "transitions": [
                    {
                        "completed_trials": 0,
                        "source": "smac_defaults",
                        "settings": self.initial_settings.to_dict(),
                    }
                ],
            }
            self._save_state()
        elif self.state_path.exists():
            self.state = _read_json(self.state_path)
            if self.state.get("policy_version") != self.policy_version:
                raise RuntimeError("Persisted policy version differs.")
        else:
            raise RuntimeError("Resume requested without LLM policy state.")
        self._proposal_count = sum(
            record.get("event_type") == "proposal"
            for record in _load_jsonl(self.telemetry_path)
        )
        self.next_settings = self.settings_class.from_mapping(
            self.state["current_settings"]
        )
        self._install_controlled_train()

    def _save_state(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _apply_settings(self, settings: RFSettings) -> None:
        self.model._rf_opts["n_estimators"] = settings.n_trees
        self.model._rf_opts["max_depth"] = settings.max_depth
        self.model._rf_opts["min_samples_split"] = settings.min_samples_split
        self.model._rf_opts["min_samples_leaf"] = settings.min_samples_leaf
        self.model._ratio_features = settings.feature_ratio

    def _install_controlled_train(self) -> None:
        original_train = self.model.train

        @functools.wraps(original_train)
        def controlled_train(*args: Any, **kwargs: Any) -> Any:
            settings = self.next_settings
            self._apply_settings(settings)
            training_data = args[0] if args else kwargs.get("X")
            started = time.perf_counter()
            result = original_train(*args, **kwargs)
            depths = [
                int(tree.get_depth()) for tree in self.model._rf.estimators_
            ]
            observation = {
                "fit_index": len(self.state["fit_observations"]),
                "training_rows": (
                    -1 if training_data is None else int(len(training_data))
                ),
                "fit_duration_seconds": time.perf_counter() - started,
                "settings": settings.to_dict(),
                "actual_tree_depth_mean": float(np.mean(depths)),
                "actual_tree_depth_min": min(depths),
                "actual_tree_depth_max": max(depths),
                "depth_utilization": float(np.mean(depths))
                / settings.max_depth,
            }
            self.state["fit_observations"].append(observation)
            self._save_state()
            return result

        self.model.train = controlled_train

    def _summary(
        self,
        checkpoint: int,
        trigger_trial: int,
        runhistory: Any,
    ) -> dict[str, Any]:
        diagnostics = sampled_runhistory_diagnostics(
            runhistory,
            self.telemetry_path,
            checkpoint,
        )
        return {
            "checkpoint": checkpoint,
            "actual_completed_trials_at_call": trigger_trial,
            "sample_rule": (
                f"Exactly {DIAGNOSTIC_SAMPLE_SIZE} evenly spaced completed "
                f"trials from trial 1 through {checkpoint}, including trial "
                f"{checkpoint}."
            ),
            "objective_direction": "minimize",
            "current_rf_settings": self.next_settings.to_dict(),
            "allowed_next_settings": {
                "n_trees": list(TREE_CHOICES),
                "max_depth": list(DEPTH_CHOICES),
                "min_samples_split": list(SPLIT_CHOICES),
                "min_samples_leaf": list(LEAF_CHOICES),
                "feature_ratio": list(FEATURE_RATIO_CHOICES),
            },
            "previous_decisions": [
                {
                    "checkpoint": int(key),
                    "settings": value["settings"],
                    "confidence": value["confidence"],
                }
                for key, value in sorted(
                    self.state["decisions"].items(),
                    key=lambda item: int(item[0]),
                )
            ],
            "recent_rf_fit_information": self.state["fit_observations"][-20:],
            "sampled_runhistory": diagnostics,
        }

    def _call_checkpoint(
        self,
        checkpoint: int,
        trigger_trial: int,
        runhistory: Any,
    ) -> None:
        key = str(checkpoint)
        if key in self.state["decisions"]:
            self.next_settings = self.settings_class.from_mapping(
                self.state["decisions"][key]["settings"]
            )
            return
        directory = self.requests_path / f"checkpoint_{checkpoint:04d}"
        summary = self._summary(checkpoint, trigger_trial, runhistory)
        prompt = self.prompt_builder(summary)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "llm_input.json", summary)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        validated_path = directory / "validated_decision.json"
        invocation_path = directory / "openai_response_metadata.json"
        if validated_path.exists():
            settings, normalized = self.decision_validator(
                _read_json(validated_path)
            )
            metadata = (
                _read_json(invocation_path)
                if invocation_path.exists()
                else {"source": "cached_validated_decision"}
            )
        else:
            raw, metadata = self.decision_provider(prompt)
            atomic_write_json(directory / "structured_output.json", raw)
            settings, normalized = self.decision_validator(raw)
            atomic_write_json(validated_path, normalized)
            atomic_write_json(invocation_path, metadata)
        decision = {
            "checkpoint": checkpoint,
            "actual_completed_trials_at_call": trigger_trial,
            "request_sha256": request_hash,
            "settings": settings.to_dict(),
            "confidence": normalized["confidence"],
            "reason": normalized["reason"],
            "response_id": metadata.get("response_id"),
            "model": metadata.get("model_returned", OPENAI_MODEL),
            "elapsed_seconds": metadata.get("elapsed_seconds"),
            "usage": metadata.get("usage"),
        }
        self.state["decisions"][key] = decision
        self.state["current_settings"] = settings.to_dict()
        self.state["transitions"].append(
            {
                "completed_trials": trigger_trial,
                "source": "openai_responses_api",
                "checkpoint": checkpoint,
                "settings": settings.to_dict(),
            }
        )
        self.next_settings = settings
        append_jsonl(
            self.events_path,
            {"event_type": "llm_decision", **decision},
        )
        self._save_state()
        print(
            f"[LLMRFPolicy] checkpoint={checkpoint} trigger={trigger_trial} "
            f"settings={settings.to_dict()}"
        )

    def on_next_configurations_start(self, config_selector: Any) -> None:
        runhistory = config_selector._runhistory
        completed = int(getattr(runhistory, "finished", len(runhistory)))
        for checkpoint in CHECKPOINTS:
            if checkpoint <= completed:
                self._call_checkpoint(checkpoint, completed, runhistory)

    def on_next_configurations_end(
        self,
        config_selector: Any,
        config: Any,
    ) -> None:
        record: dict[str, Any] = {
            "event_type": "proposal",
            "proposal_index": self._proposal_count + 1,
            "completed_trials_before_proposal": len(
                config_selector._runhistory
            ),
            "configuration_fingerprint": configuration_fingerprint(config),
            "origin": config.origin,
            "model_is_trained": False,
            "expected_improvement": None,
            "predicted_marginal_mean": None,
            "predicted_marginal_variance": None,
            "rf_settings": self.next_settings.to_dict(),
            "telemetry_error": None,
        }
        origin = "" if config.origin is None else str(config.origin)
        selected = origin.startswith("Acquisition Function")
        if selected and config_selector._model is not None:
            try:
                array = convert_configurations_to_array([config])
                mean, variance = self.model.predict_marginalized(array)
                acquisition = config_selector._acquisition_function([config])
                record.update(
                    {
                        "model_is_trained": True,
                        "expected_improvement": float(
                            np.asarray(acquisition).reshape(-1)[0]
                        ),
                        "predicted_marginal_mean": float(
                            np.asarray(mean).reshape(-1)[0]
                        ),
                        "predicted_marginal_variance": float(
                            np.asarray(variance).reshape(-1)[0]
                        ),
                    }
                )
            except Exception as error:
                record["telemetry_error"] = (
                    f"{type(error).__name__}: {error}"
                )
        append_jsonl(self.telemetry_path, record)
        self._proposal_count += 1

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        expected = [checkpoint for checkpoint in CHECKPOINTS if checkpoint < n_trials]
        missing = [
            checkpoint
            for checkpoint in expected
            if str(checkpoint) not in self.state["decisions"]
        ]
        if missing:
            raise RuntimeError(f"Completed run lacks LLM decisions {missing}.")
        return {
            "policy_version": self.policy_version,
            "checkpoints": list(CHECKPOINTS),
            "model": OPENAI_MODEL,
            "reasoning_effort": OPENAI_REASONING_EFFORT,
            "default_settings_initial_phase": self.initial_settings.to_dict(),
            "decisions": self.state["decisions"],
            "transitions": self.state["transitions"],
            "rf_fit_count": len(self.state["fit_observations"]),
        }


def _resume_state_is_valid(path: Path) -> bool:
    required = tuple(
        path / filename
        for filename in (
            "scenario.json",
            "configspace.json",
            "runhistory.json",
            "intensifier.json",
            "optimization.json",
        )
    )
    existing = [item for item in required if item.exists()]
    if not existing:
        return False
    if len(existing) != len(required):
        missing = [item.name for item in required if not item.exists()]
        raise RuntimeError(f"Partial SMAC checkpoint; missing {missing}.")
    for item in required:
        _read_json(item)
    return True


def run_llm_policy(
    benchmark_seed: int,
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
    decision_provider: Callable[
        [str], tuple[dict[str, Any], dict[str, Any]]
    ]
    | None = None,
    policy_name: str = POLICY_NAME,
    experiment_version: int = EXPERIMENT_VERSION,
    callback_factory: Callable[..., LLMRFPolicyCallback] | None = None,
    identity_extra: dict[str, Any] | None = None,
    dimension: int = DIMENSION,
    n_instances: int = N_INSTANCES,
    instance_seed: int = INSTANCE_SEED,
    initial_settings: Any = DEFAULT_SETTINGS,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"benchmark_seed must be one of {BENCHMARK_SEEDS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if n_instances < 1:
        raise ValueError("n_instances must be positive.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    output_root = Path(output_root).resolve()
    output_path = output_directory(
        output_root, benchmark_seed, smac_seed, policy_name
    )
    identity = {
        "experiment_version": experiment_version,
        "problem": "O1-DeterministicObjective",
        "benchmark_seed": benchmark_seed,
        "smac_seed": smac_seed,
        "instance_seed": instance_seed,
        "pythonhashseed": PYTHONHASHSEED,
        "dimension": dimension,
        "n_instances": n_instances,
        "n_trials": n_trials,
        "deterministic": True,
        "checkpoints": list(CHECKPOINTS),
        "diagnostic_sample_size": DIAGNOSTIC_SAMPLE_SIZE,
        "default_rf_settings": initial_settings.to_dict(),
        "pca_components": PCA_COMPONENTS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "openai_model": OPENAI_MODEL,
        "openai_reasoning_effort": OPENAI_REASONING_EFFORT,
    }
    if identity_extra:
        identity.update(identity_extra)
    completion_path = output_path / "completed.json"
    trajectory_path = output_path / "trajectory.json"
    if completion_path.exists() and not overwrite:
        completion = _read_json(completion_path)
        if (
            completion.get("state") == "complete"
            and completion.get("identity") == identity
            and trajectory_path.exists()
        ):
            print(f"Skipping complete run {output_path}.")
            return _read_json(trajectory_path)
    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and not overwrite:
        if _read_json(identity_path) != identity:
            raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(identity_path, identity)
    resume = False if overwrite else _resume_state_is_valid(output_path)

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = benchmark_seed
    problem_cfg.problem.function.wrapped_bench.dim = dimension
    problem_cfg.task.dimensions = dimension
    problem_cfg.task.search_space_n_floats = dimension
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances, instance_seed)
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        return float(
            problem.evaluate(
                TrialInfo(config=config, instance=instance, seed=seed)
            ).cost
        )

    scenario_root = output_root / f"benchmark_seed_{benchmark_seed}"
    scenario = Scenario(
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
    model = ACFacade.get_model(
        scenario=scenario,
        n_trees=initial_settings.n_trees,
        ratio_features=initial_settings.feature_ratio,
        min_samples_split=initial_settings.min_samples_split,
        min_samples_leaf=initial_settings.min_samples_leaf,
        max_depth=initial_settings.max_depth,
        pca_components=PCA_COMPONENTS,
    )
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    provider = decision_provider or OpenAIResponsesClient().invoke
    callback_type = callback_factory or LLMRFPolicyCallback
    callback = callback_type(
        output_path=output_path,
        model=model,
        decision_provider=provider,
        overwrite=not resume,
        initial_settings=initial_settings,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        callbacks=[callback],
        overwrite=not resume,
    )
    model_source = Path(inspect.getfile(type(model))).resolve()
    if LOCAL_SMAC_ROOT.resolve() not in model_source.parents:
        raise RuntimeError(f"Expected local SMAC model, found {model_source}.")
    atomic_write_json(
        output_path / "run_metadata.json",
        {
            **identity,
            "output_directory": str(output_path),
            "local_smac_model_source": str(model_source),
            "instance_map": instance_map,
            "api_key_source": "OPENAI_API_KEY environment variable",
            "api_key_persisted_in_output": False,
        },
    )
    atomic_write_json(
        completion_path,
        {"state": "running", "identity": identity},
    )
    started = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - started
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
        "best_so_far": np.minimum.accumulate(objective_values)
        .astype(float)
        .tolist(),
        "trials_per_config": {
            str(key): value for key, value in sorted(trials_per_config.items())
        },
        "walltime_seconds_this_process": walltime,
        "llm_policy": callback.audit(n_trials=n_trials),
    }
    atomic_write_json(trajectory_path, result)
    atomic_write_json(
        completion_path,
        {"state": "complete", "identity": identity},
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def local_smac_metadata() -> dict[str, str]:
    import smac

    return {
        "module": str(Path(smac.__file__).resolve()),
        "model": str(Path(inspect.getfile(ACFacade.get_model)).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-seed", type=int, choices=BENCHMARK_SEEDS, required=True
    )
    parser.add_argument(
        "--smac-seed", type=int, choices=SMAC_SEEDS, required=True
    )
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_llm_policy(
        args.benchmark_seed,
        args.smac_seed,
        n_trials=args.n_trials,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

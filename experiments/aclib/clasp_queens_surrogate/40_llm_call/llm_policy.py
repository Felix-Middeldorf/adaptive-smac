"""Codex-controlled SMAC random-forest policy with anonymized telemetry."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ConfigSpace import Configuration

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from fixed_depth_experiment import (
    LOCAL_SMAC_ROOT,
    AlgorithmConfigurationFacade,
    RandomForest,
    RandomInitialDesign,
    Scenario,
    _acquire_run_lock,
    _atomic_json,
    _package_version,
    _serialize_trajectory,
    load_initial_choice,
    resolve_initial_configuration,
)
from smac.callback import Callback
from smac.runhistory import RunHistory
from smac.runhistory.enumerations import StatusType
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


CHECKPOINTS = (500, 1_000, 1_500)
N_TRIALS = 2_500
SMAC_SEEDS = (0, 1, 2)
PCA_COMPONENTS = 4
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1
POLICY_VERSION = 1
CODEX_MODEL = "gpt-5.6-terra"
CODEX_TIMEOUT_SECONDS = 300

TREE_CHOICES = (10, 50, 100)
DEPTH_CHOICES = (5, 10, 15, 20, 30)
SPLIT_CHOICES = (2, 3, 4, 8)
LEAF_CHOICES = (1, 3, 5)
FEATURE_RATIO_CHOICES = (0.3, 0.5, 5.0 / 6.0, 1.0)

POLICY_STATE_FILENAME = "llm_policy_state.json"
POLICY_EVENTS_FILENAME = "llm_policy_events.jsonl"
REQUEST_DIRECTORY_NAME = "llm_requests"


@dataclass(frozen=True)
class ExperimentDefinition:
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


@dataclass(frozen=True)
class RFSettings:
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    feature_ratio: float

    def __post_init__(self) -> None:
        if self.n_trees not in TREE_CHOICES:
            raise ValueError(f"Unsupported tree count {self.n_trees}.")
        if self.max_depth not in DEPTH_CHOICES:
            raise ValueError(f"Unsupported depth {self.max_depth}.")
        if self.min_samples_split not in SPLIT_CHOICES:
            raise ValueError(
                f"Unsupported split size {self.min_samples_split}."
            )
        if self.min_samples_leaf not in LEAF_CHOICES:
            raise ValueError(f"Unsupported leaf size {self.min_samples_leaf}.")
        if self.feature_ratio not in FEATURE_RATIO_CHOICES:
            raise ValueError(f"Unsupported feature ratio {self.feature_ratio}.")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_ratio": self.feature_ratio,
        }

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "RFSettings":
        return cls(
            n_trees=int(payload["n_trees"]),
            max_depth=int(payload["max_depth"]),
            min_samples_split=int(payload["min_samples_split"]),
            min_samples_leaf=int(payload["min_samples_leaf"]),
            feature_ratio=float(payload["feature_ratio"]),
        )


# AlgorithmConfigurationFacade.get_model defaults. PCA remains fixed because
# switching the preprocessing representation inside a run is unsafe.
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
        "reason": {"type": "string"},
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


def run_directory(output_root: Path, smac_seed: int) -> Path:
    return output_root / "llm_policy" / str(smac_seed)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise RuntimeError(f"Partial SMAC checkpoint in {path}; missing {missing}.")
    for item in required:
        _read_json(item)
    return True


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return {
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _load_telemetry(path: Path) -> list[dict[str, Any]]:
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
            # A concurrently interrupted final JSONL record is ignored. Earlier
            # corruption remains an error.
            if line_number != len(lines):
                raise
    return records


def _runhistory_summary(runhistory: RunHistory) -> dict[str, Any]:
    completed = []
    config_costs: dict[int, list[float]] = defaultdict(list)
    config_counts: Counter[int] = Counter()
    for key in runhistory:
        value = runhistory[key]
        if value.status == StatusType.RUNNING:
            continue
        cost = float(np.asarray(value.cost, dtype=float).reshape(-1)[0])
        completed.append((key, value, cost))
        config_costs[int(key.config_id)].append(cost)
        config_counts[int(key.config_id)] += 1

    costs = [item[2] for item in completed]
    recent_costs = costs[-100:]
    target_times = [float(item[1].time) for item in completed]
    starts = [float(item[1].starttime) for item in completed]
    ends = [float(item[1].endtime) for item in completed]
    elapsed = max(ends) - min(starts) if starts and ends else 0.0
    best_id = None
    best_mean = None
    if config_costs:
        best_id = min(config_costs, key=lambda key: np.mean(config_costs[key]))
        best_mean = float(np.mean(config_costs[best_id]))
    return {
        "completed_target_trials": len(completed),
        "evaluated_configurations": len(config_counts),
        "target_cost_distribution_all": _quantiles(costs),
        "target_cost_distribution_last_100": _quantiles(recent_costs),
        "target_walltime_distribution_seconds": _quantiles(target_times),
        "evaluations_per_configuration": _quantiles(
            [float(value) for value in config_counts.values()]
        ),
        "best_observed_configuration": {
            "anonymous_configuration_id": best_id,
            "mean_on_its_evaluated_instances": best_mean,
            "evaluated_instance_count": (
                config_counts.get(best_id, 0) if best_id is not None else 0
            ),
        },
        "observed_runhistory_elapsed_seconds": elapsed,
        "completed_trials_per_hour": (
            3600.0 * len(completed) / elapsed if elapsed > 0 else None
        ),
        "target_walltime_sum_seconds": float(sum(target_times)),
        "non_target_elapsed_approximation_seconds": (
            max(0.0, elapsed - sum(target_times)) if elapsed > 0 else None
        ),
    }


def _telemetry_summary(
    telemetry_path: Path,
    fit_observations: list[dict[str, Any]],
) -> dict[str, Any]:
    records = _load_telemetry(telemetry_path)
    proposals = [
        record
        for record in records
        if record.get("event_type") == "proposal"
        and record.get("model_is_trained") is True
        and record.get("telemetry_error") is None
    ]
    completions = {
        str(record["configuration_fingerprint"]): record
        for record in records
        if record.get("event_type") == "first_completed_evaluation"
    }
    recent = proposals[-200:]

    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for record in recent:
            value: Any = record
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            if value is not None:
                output.append(float(value))
        return output

    first_costs = []
    for proposal in recent:
        completion = completions.get(str(proposal["configuration_fingerprint"]))
        if completion is not None:
            first_costs.append(float(completion["cost"]))

    recent_fits = fit_observations[-50:]
    return {
        "trained_proposals_total": len(proposals),
        "window": "most recent 200 trained proposals and 50 RF fits",
        "trained_proposals_in_window": len(recent),
        "expected_improvement": _quantiles(
            values(("acquisition", "value"))
        ),
        "predicted_marginal_mean": _quantiles(
            values(("prediction", "mean_par10"))
        ),
        "predicted_marginal_variance": _quantiles(
            values(("prediction", "variance"))
        ),
        "first_observed_target_cost_for_proposals": _quantiles(first_costs),
        "rf_fit_count_total": len(fit_observations),
        "rf_fit_duration_seconds": _quantiles(
            [float(item["fit_duration_seconds"]) for item in recent_fits]
        ),
        "rf_training_rows": _quantiles(
            [float(item["training_rows"]) for item in recent_fits]
        ),
        "actual_tree_depth_mean": _quantiles(
            [float(item["actual_tree_depth_mean"]) for item in recent_fits]
        ),
        "tree_depth_utilization": _quantiles(
            [float(item["depth_utilization"]) for item in recent_fits]
        ),
    }


def _decision_prompt(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    prompt = f"""You control the random-forest surrogate used by a sequential
model-based optimizer. Lower target cost is better. Choose the RF settings for
the next optimization phase using only the anonymized aggregate telemetry
below. Balance predictive usefulness, uncertainty quality, exploration, and
runtime. Do not identify, infer, or mention the workload. Do not request more
data. Return only the JSON object required by the supplied schema.

ANONYMIZED TELEMETRY
{payload}
"""
    lowered = prompt.lower()
    for forbidden in ("clasp", "queens", "clasp_queens"):
        if forbidden in lowered:
            raise RuntimeError("The Codex request contains a forbidden identity.")
    return prompt


def _validate_decision(payload: dict[str, Any]) -> tuple[RFSettings, dict[str, Any]]:
    if set(payload) != set(DECISION_SCHEMA["required"]):
        raise ValueError("Codex decision has missing or additional fields.")
    settings = RFSettings.from_mapping(payload)
    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Codex confidence must lie in [0, 1].")
    reason = str(payload["reason"]).strip()
    if not reason:
        raise ValueError("Codex reason must not be empty.")
    normalized = {
        **settings.to_dict(),
        "confidence": confidence,
        "reason": reason,
    }
    return settings, normalized


class CodexCLIClient:
    """One isolated, structured, ChatGPT-authenticated Codex invocation."""

    def __init__(self, binary: Path | None = None) -> None:
        configured = os.environ.get("CODEX_BINARY")
        found = configured or shutil.which("codex")
        if binary is None and not found:
            raise RuntimeError("Codex CLI was not found; set CODEX_BINARY.")
        self.binary = Path(binary or str(found)).resolve()

    def invoke(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.binary.is_file():
            raise RuntimeError(f"Codex CLI does not exist: {self.binary}")
        with tempfile.TemporaryDirectory(prefix="smac-llm-policy-") as temp:
            temporary = Path(temp)
            schema_path = temporary / "decision_schema.json"
            _atomic_json(schema_path, DECISION_SCHEMA)
            command = [
                str(self.binary),
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--color",
                "never",
                "--model",
                CODEX_MODEL,
                "--output-schema",
                str(schema_path),
                "-",
            ]
            allowed_environment = {
                key: os.environ[key]
                for key in (
                    "HOME",
                    "PATH",
                    "CODEX_HOME",
                    "CODEX_CA_CERTIFICATE",
                    "SSL_CERT_FILE",
                    "HTTPS_PROXY",
                    "HTTP_PROXY",
                    "NO_PROXY",
                    "LANG",
                    "LC_ALL",
                    "TERM",
                )
                if key in os.environ
            }
            started = time.perf_counter()
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                cwd=temporary,
                env=allowed_environment,
                timeout=CODEX_TIMEOUT_SECONDS,
                check=False,
            )
            elapsed = time.perf_counter() - started
        invocation = {
            "codex_binary": str(self.binary),
            "codex_model": CODEX_MODEL,
            "returncode": process.returncode,
            "elapsed_seconds": elapsed,
            "stderr": process.stderr,
        }
        if process.returncode != 0:
            raise RuntimeError(
                "Codex invocation failed: " + process.stderr[-4_000:]
            )
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Codex returned non-JSON stdout: {process.stdout[-2_000:]}"
            ) from error
        invocation["stdout"] = process.stdout
        return payload, invocation


class LLMRFPolicyCallback(Callback):
    """Call Codex at fixed checkpoints and control each subsequent RF fit."""

    def __init__(
        self,
        *,
        output_directory: Path,
        model: RandomForest,
        decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]],
        overwrite: bool,
    ) -> None:
        super().__init__()
        self.output_directory = Path(output_directory)
        self.model = model
        self.decision_provider = decision_provider
        self.state_path = self.output_directory / POLICY_STATE_FILENAME
        self.events_path = self.output_directory / POLICY_EVENTS_FILENAME
        self.requests_path = self.output_directory / REQUEST_DIRECTORY_NAME
        if overwrite:
            self.output_directory.mkdir(parents=True, exist_ok=True)
            self.events_path.write_text("", encoding="utf-8")
            self.state = {
                "policy_version": POLICY_VERSION,
                "current_settings": DEFAULT_SETTINGS.to_dict(),
                "last_fitted_settings": None,
                "decisions": {},
                "fit_observations": [],
                "transitions": [
                    {
                        "completed_trials": 0,
                        "source": "smac_algorithm_configuration_facade_defaults",
                        "settings": DEFAULT_SETTINGS.to_dict(),
                    }
                ],
            }
            self._save_state()
        elif self.state_path.exists():
            self.state = _read_json(self.state_path)
            if self.state.get("policy_version") != POLICY_VERSION:
                raise RuntimeError("Persisted LLM policy version differs.")
        else:
            raise RuntimeError("Resume requested without LLM policy state.")
        self._next_event_index = (
            sum(1 for _ in self.events_path.open("r", encoding="utf-8"))
            if self.events_path.exists()
            else 0
        )
        self.next_settings = RFSettings.from_mapping(
            self.state["current_settings"]
        )
        self._install_controlled_train()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _append_event(self, event: dict[str, Any]) -> None:
        payload = {"event_index": self._next_event_index, **event}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        self._next_event_index += 1

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
            training_rows = (
                int(len(training_data)) if training_data is not None else -1
            )
            started = time.perf_counter()
            result = original_train(*args, **kwargs)
            duration = time.perf_counter() - started
            estimators = self.model._rf.estimators_
            depths = [int(tree.get_depth()) for tree in estimators]
            observation = {
                "fit_index": len(self.state["fit_observations"]),
                "training_rows": training_rows,
                "fit_duration_seconds": duration,
                "settings": settings.to_dict(),
                "actual_tree_depth_mean": float(np.mean(depths)),
                "actual_tree_depth_min": min(depths),
                "actual_tree_depth_max": max(depths),
                "depth_utilization": float(np.mean(depths)) / settings.max_depth,
                "actual_max_features": int(self.model._rf.max_features),
                "transformed_feature_count": int(self.model._rf.n_features_in_),
            }
            self.state["fit_observations"].append(observation)
            self.state["last_fitted_settings"] = observation
            self._save_state()
            return result

        self.model.train = controlled_train

    def _compressed_summary(
        self,
        checkpoint: int,
        trigger_trial: int,
        runhistory: RunHistory,
    ) -> dict[str, Any]:
        summary = {
            "policy_checkpoint": checkpoint,
            "actual_completed_trials_at_call": trigger_trial,
            "objective_direction": "minimize",
            "current_rf_settings": self.next_settings.to_dict(),
            "fixed_preprocessing": {"pca_components": PCA_COMPONENTS},
            "allowed_next_settings": {
                "n_trees": list(TREE_CHOICES),
                "max_depth": list(DEPTH_CHOICES),
                "min_samples_split": list(SPLIT_CHOICES),
                "min_samples_leaf": list(LEAF_CHOICES),
                "feature_ratio": list(FEATURE_RATIO_CHOICES),
            },
            "previous_anonymous_decisions": [
                {
                    "checkpoint": int(key),
                    "settings": value["settings"],
                    "confidence": value["confidence"],
                    "reason": value["reason"],
                }
                for key, value in sorted(
                    self.state["decisions"].items(),
                    key=lambda item: int(item[0]),
                )
            ],
            "runhistory_aggregates": _runhistory_summary(runhistory),
            "surrogate_and_acquisition_aggregates": _telemetry_summary(
                self.output_directory / TELEMETRY_FILENAME,
                self.state["fit_observations"],
            ),
        }
        serialized = json.dumps(summary, sort_keys=True).lower()
        for forbidden in ("clasp", "queens", "clasp_queens"):
            if forbidden in serialized:
                raise RuntimeError("Compressed summary leaks workload identity.")
        return summary

    def _call_checkpoint(
        self,
        checkpoint: int,
        trigger_trial: int,
        runhistory: RunHistory,
    ) -> None:
        key = str(checkpoint)
        if key in self.state["decisions"]:
            self.next_settings = RFSettings.from_mapping(
                self.state["decisions"][key]["settings"]
            )
            return

        directory = self.requests_path / f"checkpoint_{checkpoint:04d}"
        request_path = directory / "anonymized_summary.json"
        prompt_path = directory / "prompt.txt"
        raw_path = directory / "codex_raw_response.json"
        parsed_path = directory / "validated_decision.json"
        invocation_path = directory / "codex_invocation.json"
        summary = self._compressed_summary(checkpoint, trigger_trial, runhistory)
        prompt = _decision_prompt(summary)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(request_path, summary)
        prompt_path.write_text(prompt, encoding="utf-8")
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if parsed_path.exists():
            payload = _read_json(parsed_path)
            settings, normalized = _validate_decision(payload)
            invocation = (
                _read_json(invocation_path)
                if invocation_path.exists()
                else {"source": "cached_validated_decision"}
            )
        else:
            raw, invocation = self.decision_provider(prompt)
            _atomic_json(raw_path, raw)
            settings, normalized = _validate_decision(raw)
            _atomic_json(parsed_path, normalized)
            _atomic_json(invocation_path, invocation)

        decision = {
            "checkpoint": checkpoint,
            "actual_completed_trials_at_call": trigger_trial,
            "request_sha256": request_hash,
            "settings": settings.to_dict(),
            "confidence": normalized["confidence"],
            "reason": normalized["reason"],
            "codex_model": invocation.get("codex_model", CODEX_MODEL),
            "codex_elapsed_seconds": invocation.get("elapsed_seconds"),
        }
        self.state["decisions"][key] = decision
        self.state["current_settings"] = settings.to_dict()
        self.state["transitions"].append(
            {
                "completed_trials": trigger_trial,
                "source": "codex",
                "checkpoint": checkpoint,
                "settings": settings.to_dict(),
            }
        )
        self.next_settings = settings
        self._append_event({"event_type": "codex_decision", **decision})
        self._save_state()
        print(
            f"[LLMRFPolicy] checkpoint={checkpoint} "
            f"trigger={trigger_trial} settings={settings.to_dict()}"
        )

    def on_next_configurations_start(self, config_selector: Any) -> None:
        runhistory = config_selector._runhistory
        completed_trials = int(
            getattr(runhistory, "finished", len(runhistory))
        )
        for checkpoint in CHECKPOINTS:
            if checkpoint <= completed_trials:
                self._call_checkpoint(
                    checkpoint,
                    completed_trials,
                    runhistory,
                )

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        decisions = self.state["decisions"]
        expected = [checkpoint for checkpoint in CHECKPOINTS if checkpoint < n_trials]
        missing = [checkpoint for checkpoint in expected if str(checkpoint) not in decisions]
        if missing:
            raise RuntimeError(f"Completed run is missing Codex decisions {missing}.")
        return {
            "policy_version": POLICY_VERSION,
            "checkpoints": list(CHECKPOINTS),
            "default_settings_first_500_trials": DEFAULT_SETTINGS.to_dict(),
            "decisions": decisions,
            "transitions": self.state["transitions"],
            "rf_fit_count": len(self.state["fit_observations"]),
        }


def run_llm_policy(
    *,
    definition: ExperimentDefinition,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    output_root: Path | None = None,
    overwrite: bool = False,
    decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    choice, choice_payload = load_initial_choice(definition.initial_choice_file)
    initial_config = resolve_initial_configuration(data, choice)
    output_root = Path(output_root or definition.output_root).resolve()
    output_path = run_directory(output_root, smac_seed)
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": spec.key,
        "smac_seed": smac_seed,
        "n_trials": n_trials,
        "checkpoints": list(CHECKPOINTS),
        "deterministic": True,
        "target_quantile_seed": 0,
        "training_instances": len(data.training_instances),
        "test_instances_used": 0,
        "initial_configuration_choice": choice_payload,
        "default_rf_settings": DEFAULT_SETTINGS.to_dict(),
        "pca_components": PCA_COMPONENTS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "codex_model": CODEX_MODEL,
        "codex_request_identity_anonymized": True,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "assets": asset_metadata(spec),
    }

    with closing(_acquire_run_lock(output_path)):
        identity_path = output_path / "run_identity.json"
        completion_path = output_path / "completed.json"
        summary_path = output_path / "summary.json"
        if completion_path.exists() and not overwrite:
            completion = _read_json(completion_path)
            if (
                completion.get("state") == "complete"
                and completion.get("identity") == identity
                and summary_path.exists()
            ):
                print(f"Complete LLM policy run found; skipping {output_path}")
                return _read_json(summary_path)
        if identity_path.exists() and not overwrite:
            if _read_json(identity_path) != identity:
                raise RuntimeError(f"Existing run identity differs in {output_path}.")
        _atomic_json(identity_path, identity)
        resume = False if overwrite else _resume_state_is_valid(output_path)

        scenario = Scenario(
            configspace=data.configspace,
            name="llm_policy",
            output_directory=output_root,
            deterministic=True,
            objectives="PAR10",
            crash_cost=spec.timeout_cost,
            n_trials=n_trials,
            use_default_config=choice.kind == "default",
            instances=list(data.training_instances),
            instance_features={
                instance: data.features[instance]
                for instance in data.training_instances
            },
            seed=smac_seed,
            n_workers=1,
        )
        if scenario.output_directory != output_path:
            raise RuntimeError(
                f"Unexpected output path {scenario.output_directory}; expected {output_path}."
            )
        initial_design = (
            AlgorithmConfigurationFacade.get_initial_design(scenario)
            if choice.kind == "default"
            else RandomInitialDesign(
                scenario,
                n_configs=0,
                additional_configs=[initial_config],
            )
        )
        model = AlgorithmConfigurationFacade.get_model(scenario=scenario)
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

        if decision_provider is None:
            client = CodexCLIClient()
            decision_provider = client.invoke
        policy = LLMRFPolicyCallback(
            output_directory=output_path,
            model=model,
            decision_provider=decision_provider,
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
            callbacks=[policy, telemetry],
            overwrite=not resume,
        )
        rf_source = Path(inspect.getfile(RandomForest)).resolve()
        if LOCAL_SMAC_ROOT.resolve() not in rf_source.parents:
            raise RuntimeError(f"Expected local SMAC RF, found {rf_source}.")
        _atomic_json(
            output_path / "run_metadata.json",
            {
                **identity,
                "output_directory": str(output_path),
                "training_only": True,
                "local_random_forest_source": str(rf_source),
                "llm_privacy": {
                    "benchmark_identity_in_request": False,
                    "instance_paths_in_request": False,
                    "target_configuration_values_in_request": False,
                    "codex_working_directory": "isolated temporary directory",
                    "codex_sandbox": "read-only",
                    "user_config_and_project_rules": "ignored",
                },
                "versions": {
                    "python": sys.version,
                    "smac_distribution": _package_version("smac"),
                    "ConfigSpace": _package_version("ConfigSpace"),
                    "epm": _package_version("epm"),
                    "pyrfr": _package_version("pyrfr"),
                    "numpy": _package_version("numpy"),
                },
            },
        )
        _atomic_json(completion_path, {"state": "running", "identity": identity})
        started = time.time()
        incumbent = facade.optimize()
        walltime = time.time() - started
        telemetry_summary = telemetry.audit(facade.runhistory)
        policy_summary = policy.audit(n_trials=n_trials)
        _atomic_json(output_path / TELEMETRY_SUMMARY_FILENAME, telemetry_summary)
        _atomic_json(output_path / "trajectory.json", _serialize_trajectory(facade))
        summary = {
            **identity,
            "output_directory": str(output_path),
            "finished_trials": int(facade.runhistory.finished),
            "configurations": len(facade.runhistory._config_ids),
            "incumbent": dict(incumbent),
            "incumbent_cost_on_evaluated_instance_keys": float(
                facade.runhistory.average_cost(incumbent, normalize=False)
            ),
            "walltime_seconds_this_process": walltime,
            "llm_policy": policy_summary,
            "configuration_telemetry": telemetry_summary,
        }
        _atomic_json(summary_path, summary)
        _atomic_json(completion_path, {"state": "complete", "identity": identity})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary


def local_smac_metadata() -> dict[str, str]:
    import smac

    return {
        "module": str(Path(smac.__file__).resolve()),
        "random_forest": str(Path(inspect.getfile(RandomForest)).resolve()),
    }

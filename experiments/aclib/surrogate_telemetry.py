"""Durable per-configuration telemetry for ACLib SMAC runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from ConfigSpace import Configuration
from scipy.stats import norm
from smac.acquisition.function.expected_improvement import EI
from smac.callback import Callback
from smac.utils.configspace import convert_configurations_to_array


TELEMETRY_SCHEMA_VERSION = 3
TELEMETRY_FILENAME = "configuration_telemetry.jsonl"
TELEMETRY_SUMMARY_FILENAME = "configuration_telemetry_summary.json"
_ACQUISITION_CACHE_ATTRIBUTE = "_aclib_surrogate_telemetry_ei"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_configuration(config: Configuration) -> dict[str, Any]:
    """Return active configuration values in a stable JSON-compatible mapping."""
    return {
        str(name): _json_safe(value)
        for name, value in sorted(dict(config).items())
    }


def configuration_fingerprint(config: Configuration) -> str:
    """Hash configuration values independently of origin or object identity."""
    serialized = json.dumps(
        canonical_configuration(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _one_float(value: Any) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1:
        raise ValueError(
            f"Expected one scalar, received shape {np.asarray(value).shape}."
        )
    result = float(array[0])
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite scalar, received {result!r}.")
    return result


def _same_json_value(left: Any, right: Any) -> bool:
    """Compare values after applying the representation used in telemetry."""
    return _json_safe(left) == _json_safe(right)


class TelemetryEI(EI):
    """SMAC's ordinary EI with proposal-local prediction caching.

    SMAC already computes the RF prediction while optimizing EI. Retaining the
    moments on each candidate lets the proposal callback record the selected
    candidate without repeating an expensive marginalization over every
    training instance. The acquisition values and ordering are unchanged.
    """

    def __init__(self, xi: float = 0.0, log: bool = False) -> None:
        super().__init__(xi=xi, log=log)
        self._telemetry_update_serial = -1

    @classmethod
    def from_smac_ei(cls, acquisition_function: EI) -> "TelemetryEI":
        if type(acquisition_function) is not EI:
            raise TypeError(
                "Telemetry requires AlgorithmConfigurationFacade's plain EI; "
                f"received {type(acquisition_function).__name__}."
            )
        if bool(getattr(acquisition_function, "_log", False)):
            raise ValueError("TelemetryEI currently requires EI in raw cost space.")
        return cls(
            xi=float(acquisition_function._xi),
            log=False,
        )

    @property
    def meta(self) -> dict[str, Any]:
        # Preserve AlgorithmConfigurationFacade's normal component identity.
        metadata = super().meta
        metadata["name"] = "EI"
        return metadata

    def _update(self, **kwargs: Any) -> None:
        super()._update(**kwargs)
        self._telemetry_update_serial += 1

    def _ei_from_moments(
        self,
        means: np.ndarray,
        variances: np.ndarray,
    ) -> np.ndarray:
        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update(eta=<int>) before EI."
            )
        if self._log:
            raise ValueError("TelemetryEI only supports EI in raw cost space.")

        means = np.asarray(means, dtype=float)
        variances = np.asarray(variances, dtype=float)
        standard_deviations = np.sqrt(variances)
        improvement = self._eta - means - self._xi
        safe_standard_deviations = standard_deviations.copy()
        zero_uncertainty = safe_standard_deviations == 0.0
        safe_standard_deviations[zero_uncertainty] = 1.0
        z = improvement / safe_standard_deviations
        values = (
            improvement * norm.cdf(z)
            + safe_standard_deviations * norm.pdf(z)
        )
        values[zero_uncertainty] = 0.0
        if (values < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample."
            )
        return values

    def value_from_moments(self, mean: float, variance: float) -> float:
        """Return exact current EI without another model prediction."""
        return _one_float(
            self._ei_from_moments(
                np.asarray([[mean]], dtype=float),
                np.asarray([[variance]], dtype=float),
            )
        )

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("EI has no model.")

        configuration_array = convert_configurations_to_array(configurations)
        if len(configuration_array.shape) == 1:
            configuration_array = configuration_array[np.newaxis, :]
        means, variances = self.model.predict_marginalized(
            configuration_array
        )
        values = self._ei_from_moments(means, variances)
        if np.any(np.isnan(values)):
            indices = np.where(np.isnan(values))[0]
            values[indices, :] = -np.finfo(float).max

        fitted_forest = getattr(self.model, "_rf", None)
        for config, mean, variance, value in zip(
            configurations,
            means.reshape(-1),
            variances.reshape(-1),
            values.reshape(-1),
        ):
            setattr(
                config,
                _ACQUISITION_CACHE_ATTRIBUTE,
                {
                    "fitted_forest_identity": id(fitted_forest),
                    "acquisition_update_serial": (
                        self._telemetry_update_serial
                    ),
                    "eta": _json_safe(self._eta),
                    "xi": _json_safe(self._xi),
                    "mean": float(mean),
                    "variance": float(variance),
                    "value": float(value),
                },
            )
        return values


class SurrogateTelemetryCallback(Callback):
    """Persist proposal-time RF state and first-completion events.

    Proposal events are flushed immediately in
    ``on_next_configurations_end``. This preserves the exact RF and Expected
    Improvement state even if the job is preempted before the proposed
    configuration finishes. A separate completion event is appended after its
    first target evaluation.
    """

    def __init__(
        self,
        output_directory: Path,
        *,
        model: Any,
        acquisition_function: Any,
        overwrite: bool = False,
    ) -> None:
        super().__init__()
        self.output_directory = Path(output_directory)
        self.path = self.output_directory / TELEMETRY_FILENAME
        self._model = model
        self._acquisition_function = acquisition_function
        self._events: list[dict[str, Any]] = []
        self._event_indices: set[int] = set()
        self._proposal_records: dict[str, dict[str, Any]] = {}
        self._completion_records: dict[str, dict[str, Any]] = {}
        self._next_event_index = 0
        self._next_configuration_index = 0
        self._last_fitted_forest: Any | None = None
        self._fit_serial = -1
        self._initialize_file(overwrite=overwrite)

    @property
    def record_count(self) -> int:
        return len(self._events)

    def _register_event(self, record: dict[str, Any]) -> None:
        event_index = int(record["event_index"])
        if event_index in self._event_indices:
            raise RuntimeError(
                f"Duplicate telemetry event index {event_index} in {self.path}."
            )

        event_type = str(record["event_type"])
        fingerprint = str(record["configuration_fingerprint"])
        if event_type == "proposal":
            if fingerprint in self._proposal_records:
                raise RuntimeError(
                    f"Duplicate proposal telemetry for {fingerprint} in "
                    f"{self.path}."
                )
            self._proposal_records[fingerprint] = record
        elif event_type == "first_completed_evaluation":
            if fingerprint in self._completion_records:
                raise RuntimeError(
                    f"Duplicate completion telemetry for {fingerprint} in "
                    f"{self.path}."
                )
            self._completion_records[fingerprint] = record
        else:
            raise RuntimeError(
                f"Unknown telemetry event type {event_type!r} in {self.path}."
            )

        self._event_indices.add(event_index)
        self._events.append(record)

    def _initialize_file(self, *, overwrite: bool) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if overwrite and self.path.exists():
            self.path.unlink()

        if not self.path.exists():
            self.path.touch()
            return

        raw_lines = self.path.read_text(encoding="utf-8").splitlines()
        nonempty_indices = [
            index for index, line in enumerate(raw_lines) if line.strip()
        ]
        last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
        recovered_truncated_tail = False
        valid_lines: list[str] = []

        for index, line in enumerate(raw_lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index != last_nonempty:
                    raise RuntimeError(
                        f"Invalid telemetry JSONL record at line {index + 1}: "
                        f"{self.path}"
                    )
                recovered_truncated_tail = True
                break

            if record.get("schema_version") != TELEMETRY_SCHEMA_VERSION:
                raise RuntimeError(
                    f"Incompatible telemetry schema in {self.path}: "
                    f"{record.get('schema_version')!r}."
                )
            self._register_event(record)
            valid_lines.append(
                json.dumps(record, sort_keys=True, separators=(",", ":"))
            )

        if recovered_truncated_tail:
            temporary = self.path.with_suffix(self.path.suffix + ".recovered")
            with temporary.open("w", encoding="utf-8") as handle:
                for line in valid_lines:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)

        if self._events:
            self._next_event_index = (
                max(int(record["event_index"]) for record in self._events) + 1
            )
        if self._proposal_records:
            self._next_configuration_index = (
                max(
                    int(record["configuration_index"])
                    for record in self._proposal_records.values()
                )
                + 1
            )
            self._fit_serial = max(
                (
                    int(record["random_forest"]["fit_serial"])
                    for record in self._proposal_records.values()
                    if record["random_forest"]["fit_serial"] is not None
                ),
                default=-1,
            )

    def _append_event(self, record: dict[str, Any]) -> None:
        serialized = json.dumps(
            _json_safe(record),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self._register_event(record)
        self._next_event_index += 1
        if record["event_type"] == "proposal":
            self._next_configuration_index += 1

    def _snapshot(self, config_selector: Any, config: Configuration) -> dict[str, Any]:
        if config_selector._model is not self._model:
            raise RuntimeError("Telemetry is not attached to SMAC's active RF model.")
        if config_selector._acquisition_function is not self._acquisition_function:
            raise RuntimeError(
                "Telemetry is not attached to SMAC's active acquisition function."
            )

        fitted_forest = getattr(self._model, "_rf", None)
        model_is_trained = (
            fitted_forest is not None
            and hasattr(fitted_forest, "estimators_")
            and len(fitted_forest.estimators_) > 0
        )
        training_rows = int(getattr(config_selector, "_previous_entries", -1))
        if training_rows < 0:
            training_rows = None
        if model_is_trained and fitted_forest is not self._last_fitted_forest:
            self._fit_serial += 1
            self._last_fitted_forest = fitted_forest

        instance_features = getattr(self._model, "_instance_features", None)
        instance_count = (
            len(instance_features) if instance_features is not None else 0
        )
        rf_options = getattr(self._model, "_rf_opts", {})
        requested_depth = _json_safe(rf_options.get("max_depth"))
        snapshot: dict[str, Any] = {
            "proposal_origin": config.origin,
            "model_is_trained": model_is_trained,
            "model_training_rows": training_rows,
            "unavailable_reason": (
                None if model_is_trained else "model_not_fitted_before_proposal"
            ),
            "prediction": {
                "mean_par10": None,
                "variance": None,
                "standard_deviation": None,
                "marginalized_over_training_instances": True,
                "training_instance_count": instance_count,
            },
            "acquisition": {
                "name": getattr(self._acquisition_function, "name", None),
                "value": None,
                "eta": _json_safe(
                    getattr(self._acquisition_function, "_eta", None)
                ),
                "xi": _json_safe(
                    getattr(self._acquisition_function, "_xi", None)
                ),
                "value_source": None,
                "uses_exact_current_state": False,
            },
            "random_forest": {
                "requested_max_depth": requested_depth,
                "fit_serial": self._fit_serial if model_is_trained else None,
                "tree_count": 0,
                "actual_tree_depths": [],
                "actual_tree_depth_min": None,
                "actual_tree_depth_max": None,
                "actual_tree_depth_mean": None,
            },
            "telemetry_error": None,
        }
        if not model_is_trained:
            return snapshot

        try:
            if not isinstance(self._acquisition_function, TelemetryEI):
                raise TypeError(
                    "Expected TelemetryEI, received "
                    f"{type(self._acquisition_function).__name__}."
                )

            cached = getattr(
                config,
                _ACQUISITION_CACHE_ATTRIBUTE,
                None,
            )
            cache_is_current = (
                isinstance(cached, Mapping)
                and cached.get("fitted_forest_identity")
                == id(fitted_forest)
                and cached.get("acquisition_update_serial")
                == self._acquisition_function._telemetry_update_serial
                and _same_json_value(
                    cached.get("eta"),
                    getattr(self._acquisition_function, "_eta", None),
                )
                and _same_json_value(
                    cached.get("xi"),
                    getattr(self._acquisition_function, "_xi", None),
                )
            )
            if cache_is_current:
                mean = _one_float(cached["mean"])
                variance = _one_float(cached["variance"])
                acquisition_value = _one_float(cached["value"])
                acquisition_value_source = (
                    "acquisition_function_evaluation_cache"
                )
            else:
                config_array = convert_configurations_to_array([config])
                means, variances = self._model.predict_marginalized(
                    config_array
                )
                mean = _one_float(means)
                variance = _one_float(variances)
                acquisition_value = (
                    self._acquisition_function.value_from_moments(
                        mean,
                        variance,
                    )
                )
                acquisition_value_source = (
                    "closed_form_from_recorded_prediction"
                )

            if variance < 0.0:
                raise ValueError(f"SMAC RF returned negative variance {variance}.")

            tree_depths = [
                int(estimator.get_depth())
                for estimator in fitted_forest.estimators_
            ]
            expected_tree_count = int(rf_options["n_estimators"])
            if len(tree_depths) != expected_tree_count:
                raise RuntimeError(
                    f"Expected {expected_tree_count} fitted trees, found "
                    f"{len(tree_depths)}."
                )
            if requested_depth is not None and any(
                depth > requested_depth for depth in tree_depths
            ):
                raise RuntimeError(
                    f"A fitted tree exceeds requested depth {requested_depth}."
                )

            snapshot["prediction"] = {
                "mean_par10": mean,
                "variance": variance,
                "standard_deviation": math.sqrt(variance),
                "marginalized_over_training_instances": True,
                "training_instance_count": instance_count,
            }
            snapshot["acquisition"]["value"] = acquisition_value
            snapshot["acquisition"]["value_source"] = (
                acquisition_value_source
            )
            snapshot["acquisition"]["uses_exact_current_state"] = True
            snapshot["random_forest"] = {
                "requested_max_depth": requested_depth,
                "fit_serial": self._fit_serial,
                "tree_count": len(tree_depths),
                "actual_tree_depths": tree_depths,
                "actual_tree_depth_min": min(tree_depths),
                "actual_tree_depth_max": max(tree_depths),
                "actual_tree_depth_mean": float(np.mean(tree_depths)),
            }
        except Exception as error:
            snapshot["telemetry_error"] = (
                f"{type(error).__name__}: {error}"
            )

        return snapshot

    def _proposal_record(
        self,
        config: Configuration,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_type": "proposal",
            "event_index": self._next_event_index,
            "configuration_index": self._next_configuration_index,
            "configuration_fingerprint": configuration_fingerprint(config),
            "configuration": canonical_configuration(config),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            **snapshot,
        }

    def _unavailable_resume_snapshot(
        self,
        config: Configuration,
    ) -> dict[str, Any]:
        instance_features = getattr(self._model, "_instance_features", None)
        return {
            "proposal_origin": config.origin,
            "model_is_trained": False,
            "model_training_rows": None,
            "unavailable_reason": "proposal_state_unavailable_after_resume",
            "prediction": {
                "mean_par10": None,
                "variance": None,
                "standard_deviation": None,
                "marginalized_over_training_instances": True,
                "training_instance_count": (
                    len(instance_features)
                    if instance_features is not None
                    else 0
                ),
            },
            "acquisition": {
                "name": getattr(self._acquisition_function, "name", None),
                "value": None,
                "eta": None,
                "xi": _json_safe(
                    getattr(self._acquisition_function, "_xi", None)
                ),
                "value_source": None,
                "uses_exact_current_state": False,
            },
            "random_forest": {
                "requested_max_depth": _json_safe(
                    getattr(self._model, "_rf_opts", {}).get("max_depth")
                ),
                "fit_serial": None,
                "tree_count": 0,
                "actual_tree_depths": [],
                "actual_tree_depth_min": None,
                "actual_tree_depth_max": None,
                "actual_tree_depth_mean": None,
            },
            "telemetry_error": None,
        }

    def on_next_configurations_end(
        self,
        config_selector: Any,
        config: Configuration,
    ) -> None:
        fingerprint = configuration_fingerprint(config)
        if fingerprint in self._proposal_records:
            return

        snapshot = self._snapshot(config_selector, config)
        if snapshot["model_is_trained"] and snapshot["telemetry_error"] is not None:
            raise RuntimeError(
                "Could not collect trained SMAC RF telemetry: "
                f"{snapshot['telemetry_error']}"
            )
        self._append_event(self._proposal_record(config, snapshot))

    def on_tell_end(self, smbo: Any, info: Any, value: Any) -> None:
        fingerprint = configuration_fingerprint(info.config)
        if fingerprint in self._completion_records:
            return

        if fingerprint not in self._proposal_records:
            unavailable = self._unavailable_resume_snapshot(info.config)
            self._append_event(
                self._proposal_record(info.config, unavailable)
            )

        completion = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_type": "first_completed_evaluation",
            "event_index": self._next_event_index,
            "configuration_fingerprint": fingerprint,
            "runhistory_config_id": int(
                smbo.runhistory.get_config_id(info.config)
            ),
            "instance": _json_safe(info.instance),
            "seed": _json_safe(info.seed),
            "budget": _json_safe(info.budget),
            "cost": _json_safe(value.cost),
            "status": getattr(value.status, "name", str(value.status)),
            "walltime": float(value.time),
            "cpu_time": float(value.cpu_time),
            "runhistory_submitted": int(smbo.runhistory.submitted),
            "runhistory_finished": int(smbo.runhistory.finished),
        }
        self._append_event(completion)

    def audit(self, runhistory: Any) -> dict[str, Any]:
        evaluated_configurations = {
            configuration_fingerprint(config): config
            for config in runhistory.get_configs()
        }
        evaluated_fingerprints = set(evaluated_configurations)
        proposal_fingerprints = set(self._proposal_records)
        completion_fingerprints = set(self._completion_records)
        missing_proposals = sorted(
            evaluated_fingerprints - proposal_fingerprints
        )
        missing_completions = sorted(
            evaluated_fingerprints - completion_fingerprints
        )
        if missing_proposals or missing_completions:
            raise RuntimeError(
                "Incomplete telemetry for evaluated configurations: "
                f"missing proposals={missing_proposals}, "
                f"missing completions={missing_completions}."
            )

        evaluated_proposals = [
            self._proposal_records[fingerprint]
            for fingerprint in evaluated_fingerprints
        ]
        errored_records = [
            record
            for record in evaluated_proposals
            if record["telemetry_error"] is not None
        ]
        if errored_records:
            raise RuntimeError(
                f"Telemetry failed for {len(errored_records)} evaluated "
                "configuration(s); inspect configuration_telemetry.jsonl."
            )

        unavailable_resume_records = [
            record
            for record in evaluated_proposals
            if record["unavailable_reason"]
            == "proposal_state_unavailable_after_resume"
        ]
        if unavailable_resume_records:
            raise RuntimeError(
                "Proposal-time telemetry is unavailable for "
                f"{len(unavailable_resume_records)} evaluated configuration(s) "
                "after resume. Restart this output once with --overwrite."
            )

        stale_completions: list[str] = []
        for fingerprint in evaluated_fingerprints:
            completion = self._completion_records[fingerprint]
            config = evaluated_configurations[fingerprint]
            actual_config_id = int(runhistory.get_config_id(config))
            if int(completion["runhistory_config_id"]) != actual_config_id:
                stale_completions.append(fingerprint)
                continue

            matching_trial_found = False
            for trial_key in runhistory:
                if int(trial_key.config_id) != actual_config_id:
                    continue
                trial_value = runhistory[trial_key]
                if not all(
                    (
                        _same_json_value(
                            completion["instance"],
                            trial_key.instance,
                        ),
                        _same_json_value(completion["seed"], trial_key.seed),
                        _same_json_value(completion["budget"], trial_key.budget),
                        _same_json_value(completion["cost"], trial_value.cost),
                        completion["status"]
                        == getattr(
                            trial_value.status,
                            "name",
                            str(trial_value.status),
                        ),
                    )
                ):
                    continue
                matching_trial_found = True
                break
            if not matching_trial_found:
                stale_completions.append(fingerprint)

        if stale_completions:
            raise RuntimeError(
                "Completion telemetry does not match persisted runhistory for "
                f"{len(stale_completions)} configuration(s): "
                f"{sorted(stale_completions)}. Restart once with --overwrite."
            )

        expected_event_indices = list(range(len(self._events)))
        actual_event_indices = sorted(self._event_indices)
        if actual_event_indices != expected_event_indices:
            raise RuntimeError(
                "Telemetry event indices are not contiguous from zero: "
                f"{actual_event_indices}."
            )

        trained_records = [
            record
            for record in evaluated_proposals
            if record["model_is_trained"]
        ]
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "telemetry_file": TELEMETRY_FILENAME,
            "event_records": len(self._events),
            "proposal_records": len(self._proposal_records),
            "first_completion_records": len(self._completion_records),
            "unique_evaluated_configurations": len(evaluated_fingerprints),
            "trained_model_evaluated_configurations": len(trained_records),
            "pre_model_evaluated_configurations": (
                len(evaluated_proposals) - len(trained_records)
            ),
            "missing_proposals": missing_proposals,
            "missing_first_completions": missing_completions,
            "proposed_but_not_evaluated": sorted(
                proposal_fingerprints - evaluated_fingerprints
            ),
            "completion_without_evaluated_runhistory": sorted(
                completion_fingerprints - evaluated_fingerprints
            ),
            "stale_completion_records": len(stale_completions),
            "telemetry_error_records": len(errored_records),
            "unavailable_resume_records": len(unavailable_resume_records),
        }

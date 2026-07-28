"""Shared analytics for completed ACLib fixed-depth surrogate experiments."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from ConfigSpace import Configuration
from epm.experiment_utils.data_handling import unwarp

from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)


ANALYTICS_VERSION = 1
DEPTHS = (5, 10, 15, 20, 30)
SMAC_SEEDS = (0, 1, 2)
N_TRIALS = 5_000
VALIDATION_QUANTILE_SEED = 0


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def canonical_configuration(
    configuration: Configuration | Mapping[str, Any],
) -> dict[str, Any]:
    return {
        str(name): _json_safe(value)
        for name, value in sorted(dict(configuration).items())
    }


def configuration_fingerprint(
    configuration: Configuration | Mapping[str, Any],
) -> str:
    serialized = json.dumps(
        canonical_configuration(configuration),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid JSONL at {path}:{line_number}."
            ) from error
    return records


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class FullTrainingValidator:
    """Evaluate configurations on every ACLib training instance.

    For the stochastic quantile surrogates, quantile seed zero is the
    deterministic median. Using it for every configuration makes validation
    comparable across SMAC seeds and matches the initial-configuration screen.
    """

    def __init__(
        self,
        benchmark_key: str,
        experiment_directory: Path,
        *,
        quantile_seed: int = VALIDATION_QUANTILE_SEED,
    ) -> None:
        self.spec = get_benchmark_spec(benchmark_key)
        self.experiment_directory = Path(experiment_directory)
        self.quantile_seed = int(quantile_seed)
        self.data = load_benchmark_data(self.spec)
        self.cache_file = (
            self.experiment_directory
            / "analytics_cache"
            / f"full_training_validation_qseed_{self.quantile_seed}.json"
        )
        self._benchmark: ACLibSurrogateBenchmark | None = None
        self._feature_matrix = np.asarray(
            [
                self.data.features[instance]
                for instance in self.data.training_instances
            ],
            dtype=np.float32,
        )
        self._asset_signature = asset_metadata(self.spec)
        self.results = self._load_cache()

    def _cache_identity(self) -> dict[str, Any]:
        return {
            "analytics_version": ANALYTICS_VERSION,
            "benchmark": self.spec.key,
            "validation_quantile_seed": self.quantile_seed,
            "training_instance_count": len(self.data.training_instances),
            "asset_signature": self._asset_signature,
        }

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_file.is_file():
            return {}
        payload = _read_json(self.cache_file)
        expected = self._cache_identity()
        found = {
            key: payload.get(key)
            for key in expected
        }
        if found != expected:
            raise RuntimeError(
                f"Incompatible validation cache {self.cache_file}. "
                "Move or remove it before recomputing."
            )
        return {
            str(fingerprint): dict(result)
            for fingerprint, result in payload.get("results", {}).items()
        }

    def _save_cache(self) -> None:
        _atomic_json(
            self.cache_file,
            {
                **self._cache_identity(),
                "results": self.results,
            },
        )

    @property
    def benchmark(self) -> ACLibSurrogateBenchmark:
        if self._benchmark is None:
            self._benchmark = ACLibSurrogateBenchmark(self.spec)
        return self._benchmark

    def _model_matrix(self, configuration: Mapping[str, Any]) -> np.ndarray:
        first_instance = self.data.training_instances[0]
        first_row = self.benchmark._model_input(
            configuration,
            first_instance,
        ).reshape(-1)
        parameter_count = len(self.data.configspace)
        parameters = first_row[:parameter_count]
        repeated_parameters = np.repeat(
            parameters[np.newaxis, :],
            len(self._feature_matrix),
            axis=0,
        )
        return np.hstack(
            (repeated_parameters, self._feature_matrix)
        ).astype(np.float32)

    def evaluate(self, configuration: Mapping[str, Any]) -> dict[str, Any]:
        fingerprint = configuration_fingerprint(configuration)
        if fingerprint in self.results:
            return self.results[fingerprint]

        model = self.benchmark.surrogate.model
        if not hasattr(model, "predict"):
            raise RuntimeError(
                f"Unsupported ACLib model type {type(model).__name__}."
            )
        raw_logged = model.predict(
            self._model_matrix(configuration),
            seed=self.quantile_seed,
            num_samples=1,
        )
        raw_predictions = np.asarray(
            unwarp(raw_logged, quality=False),
            dtype=float,
        ).reshape(-1)
        if len(raw_predictions) != len(self.data.training_instances):
            raise RuntimeError(
                "Validation prediction count does not match training instances."
            )
        if not np.all(np.isfinite(raw_predictions)):
            raise RuntimeError("Non-finite ACLib validation prediction.")

        timed_out = raw_predictions > self.spec.cutoff
        par10 = np.where(
            timed_out,
            self.spec.timeout_cost,
            raw_predictions,
        )
        result = {
            "mean_par10": float(np.mean(par10)),
            "median_par10": float(np.median(par10)),
            "timeout_count": int(np.count_nonzero(timed_out)),
            "training_instance_count": len(self.data.training_instances),
        }
        self.results[fingerprint] = result
        return result

    def evaluate_many(
        self,
        configurations: Mapping[str, Mapping[str, Any]],
        *,
        checkpoint_every: int = 100,
    ) -> dict[str, dict[str, Any]]:
        missing = [
            (fingerprint, configuration)
            for fingerprint, configuration in configurations.items()
            if fingerprint not in self.results
        ]
        if not missing:
            print(
                f"Validation cache hit: {len(configurations)} configurations "
                f"from {self.cache_file}."
            )
            return self.results

        print(
            f"Validating {len(missing)} new configurations on all "
            f"{len(self.data.training_instances)} training instances "
            f"(quantile seed {self.quantile_seed})."
        )
        started = time.perf_counter()
        for index, (fingerprint, configuration) in enumerate(
            missing,
            start=1,
        ):
            if configuration_fingerprint(configuration) != fingerprint:
                raise RuntimeError("Configuration fingerprint mismatch.")
            self.evaluate(configuration)
            if index % checkpoint_every == 0 or index == len(missing):
                self._save_cache()
                elapsed = time.perf_counter() - started
                print(
                    f"  validated {index}/{len(missing)} new configurations "
                    f"in {elapsed:.1f}s"
                )
        return self.results


@dataclass(frozen=True)
class AnalysisBundle:
    benchmark_key: str
    display_name: str
    experiment_directory: Path
    n_trials: int
    training_instance_count: int
    validation_quantile_seed: int
    trajectory_events: pd.DataFrame
    final_incumbents: pd.DataFrame
    proposal_diagnostics: pd.DataFrame
    ei_batch_variance: pd.DataFrame
    validation_cache_file: Path


def _validated_run_directory(
    results_directory: Path,
    depth: int,
    smac_seed: int,
    *,
    require_complete: bool,
) -> Path:
    directory = results_directory / f"depth_{depth}" / str(smac_seed)
    partial_required = (
        "completed.json",
        "intensifier.json",
        "runhistory.json",
        "configuration_telemetry.jsonl",
    )
    required = (
        partial_required
        + (
            "summary.json",
            "trajectory.json",
            "configuration_telemetry_summary.json",
        )
        if require_complete
        else partial_required
    )
    missing = [
        filename
        for filename in required
        if not (directory / filename).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Incomplete run {directory}; missing {missing}."
        )
    completion = _read_json(directory / "completed.json")
    if completion.get("state") not in {"running", "complete"}:
        raise RuntimeError(
            f"Run has unexpected state {completion.get('state')!r}: "
            f"{directory}."
        )
    if not require_complete:
        return directory

    summary = _read_json(directory / "summary.json")
    telemetry_summary = _read_json(
        directory / "configuration_telemetry_summary.json"
    )
    if completion.get("state") != "complete":
        raise RuntimeError(f"Run is not complete: {directory}.")
    if int(summary.get("finished_trials", -1)) != N_TRIALS:
        raise RuntimeError(f"Run has wrong trial count: {directory}.")
    if telemetry_summary.get("missing_proposals") != []:
        raise RuntimeError(f"Missing proposal telemetry: {directory}.")
    if telemetry_summary.get("missing_first_completions") != []:
        raise RuntimeError(f"Missing completion telemetry: {directory}.")
    if telemetry_summary.get("telemetry_error_records") != 0:
        raise RuntimeError(f"Telemetry errors exist: {directory}.")
    return directory


def _add_configuration(
    configurations: dict[str, dict[str, Any]],
    configuration: Mapping[str, Any],
    fingerprint: str | None = None,
) -> str:
    canonical = canonical_configuration(configuration)
    calculated = configuration_fingerprint(canonical)
    if fingerprint is not None and calculated != fingerprint:
        raise RuntimeError("Stored telemetry fingerprint is inconsistent.")
    previous = configurations.setdefault(calculated, canonical)
    if previous != canonical:
        raise RuntimeError("Fingerprint collision between configurations.")
    return calculated


def build_analysis_bundle(
    experiment_directory: Path,
    benchmark_key: str,
    *,
    selected_seed: int = 0,
    validation_quantile_seed: int = VALIDATION_QUANTILE_SEED,
    trial_limit: int | str | None = None,
) -> AnalysisBundle:
    experiment_directory = Path(experiment_directory).resolve()
    results_directory = experiment_directory / "results"
    if selected_seed not in SMAC_SEEDS:
        raise ValueError(f"selected_seed must be one of {SMAC_SEEDS}.")

    if trial_limit is not None and trial_limit != "minimum":
        if not isinstance(trial_limit, int) or isinstance(trial_limit, bool):
            raise ValueError(
                "trial_limit must be None, a positive integer, or 'minimum'."
            )
        if trial_limit < 1:
            raise ValueError("trial_limit must be positive.")

    require_complete = trial_limit is None
    run_payloads: dict[tuple[int, int], dict[str, Any]] = {}
    for depth in DEPTHS:
        for smac_seed in SMAC_SEEDS:
            directory = _validated_run_directory(
                results_directory,
                depth,
                smac_seed,
                require_complete=require_complete,
            )
            runhistory = _read_json(directory / "runhistory.json")
            trajectory_file = directory / "trajectory.json"
            if trajectory_file.is_file():
                trajectory = _read_json(trajectory_file)
            else:
                trajectory = _read_json(
                    directory / "intensifier.json"
                ).get("trajectory", [])
            telemetry = (
                _read_jsonl(directory / "configuration_telemetry.jsonl")
                if smac_seed == selected_seed
                else []
            )
            run_payloads[(depth, smac_seed)] = {
                "directory": directory,
                "runhistory": runhistory,
                "trajectory": trajectory,
                "telemetry": telemetry,
                "finished_trials": len(runhistory.get("data", [])),
            }

    minimum_finished_trials = min(
        int(payload["finished_trials"])
        for payload in run_payloads.values()
    )
    if trial_limit == "minimum":
        effective_trials = minimum_finished_trials
    elif trial_limit is None:
        effective_trials = N_TRIALS
    else:
        effective_trials = int(trial_limit)
    if effective_trials < 1:
        raise RuntimeError("At least one run has no completed trials.")
    if effective_trials > minimum_finished_trials:
        raise RuntimeError(
            f"Requested {effective_trials} trials, but the least advanced "
            f"run has only {minimum_finished_trials}."
        )

    configurations: dict[str, dict[str, Any]] = {}
    for payload in run_payloads.values():
        runhistory = payload["runhistory"]
        for event in payload["trajectory"]:
            if int(event["trial"]) > effective_trials:
                continue
            if len(event["config_ids"]) != 1:
                raise RuntimeError(
                    "Expected exactly one incumbent in single-objective run."
                )
            config_id = str(int(event["config_ids"][0]))
            _add_configuration(
                configurations,
                runhistory["configs"][config_id],
            )

        telemetry = payload["telemetry"]
        if telemetry:
            completions = {
                str(event["configuration_fingerprint"]): event
                for event in telemetry
                if event.get("event_type") == "first_completed_evaluation"
            }
            for event in telemetry:
                if event.get("event_type") != "proposal":
                    continue
                fingerprint = str(event["configuration_fingerprint"])
                completion = completions.get(fingerprint)
                if completion is None:
                    continue
                if (
                    int(completion["runhistory_finished"])
                    > effective_trials
                ):
                    continue
                _add_configuration(
                    configurations,
                    event["configuration"],
                    fingerprint,
                )

    validator = FullTrainingValidator(
        benchmark_key,
        experiment_directory,
        quantile_seed=validation_quantile_seed,
    )
    validation = validator.evaluate_many(configurations)

    trajectory_rows: list[dict[str, Any]] = []
    for (depth, smac_seed), payload in run_payloads.items():
        runhistory = payload["runhistory"]
        events = sorted(
            (
                event
                for event in payload["trajectory"]
                if int(event["trial"]) <= effective_trials
            ),
            key=lambda event: int(event["trial"]),
        )
        if not events:
            raise RuntimeError(
                f"No trajectory event by trial {effective_trials} for "
                f"depth={depth}, seed={smac_seed}."
            )
        for event in events:
            config_id = int(event["config_ids"][0])
            configuration = runhistory["configs"][str(config_id)]
            fingerprint = configuration_fingerprint(configuration)
            trajectory_rows.append(
                {
                    "depth": depth,
                    "smac_seed": smac_seed,
                    "trial": int(event["trial"]),
                    "config_id": config_id,
                    "configuration_fingerprint": fingerprint,
                    "full_training_par10": float(
                        validation[fingerprint]["mean_par10"]
                    ),
                    "smac_incumbent_subset_cost": float(event["costs"][0]),
                }
            )
        last = trajectory_rows[-1]
        if (
            last["depth"] != depth
            or last["smac_seed"] != smac_seed
        ):
            raise RuntimeError("Trajectory construction order mismatch.")
        if last["trial"] < effective_trials:
            trajectory_rows.append(
                {
                    **last,
                    "trial": effective_trials,
                }
            )

    trajectory_events = pd.DataFrame(trajectory_rows).sort_values(
        ["depth", "smac_seed", "trial"]
    )
    final_incumbents = (
        trajectory_events.sort_values("trial")
        .groupby(["depth", "smac_seed"], as_index=False)
        .tail(1)
        .sort_values(["depth", "smac_seed"])
        .reset_index(drop=True)
    )

    proposal_rows: list[dict[str, Any]] = []
    for (depth, smac_seed), payload in run_payloads.items():
        if smac_seed != selected_seed:
            continue
        telemetry = payload["telemetry"]
        completions = {
            str(event["configuration_fingerprint"]): event
            for event in telemetry
            if event.get("event_type") == "first_completed_evaluation"
        }
        for event in telemetry:
            if event.get("event_type") != "proposal":
                continue
            if not event.get("model_is_trained"):
                continue
            fingerprint = str(event["configuration_fingerprint"])
            completion = completions.get(fingerprint)
            if completion is None:
                continue
            if (
                int(completion["runhistory_finished"])
                > effective_trials
            ):
                continue
            prediction = float(event["prediction"]["mean_par10"])
            variance = float(event["prediction"]["variance"])
            standard_deviation = float(
                event["prediction"]["standard_deviation"]
            )
            actual = float(validation[fingerprint]["mean_par10"])
            signed_error = actual - prediction
            standardized_error = (
                signed_error / standard_deviation
                if standard_deviation > 0
                else math.nan
            )
            proposal_rows.append(
                {
                    "depth": depth,
                    "smac_seed": smac_seed,
                    "proposal_trial": int(
                        completion["runhistory_finished"]
                    ),
                    "model_training_rows": int(
                        event["model_training_rows"]
                    ),
                    "fit_serial": int(
                        event["random_forest"]["fit_serial"]
                    ),
                    "configuration_fingerprint": fingerprint,
                    "expected_improvement": float(
                        event["acquisition"]["value"]
                    ),
                    "predicted_full_training_par10": prediction,
                    "prediction_variance": variance,
                    "prediction_standard_deviation": standard_deviation,
                    "actual_full_training_par10": actual,
                    "signed_error": signed_error,
                    "absolute_error": abs(signed_error),
                    "standardized_error": standardized_error,
                    "average_actual_tree_depth": float(
                        event["random_forest"]["actual_tree_depth_mean"]
                    ),
                }
            )

    proposal_diagnostics = pd.DataFrame(proposal_rows).sort_values(
        ["depth", "proposal_trial"]
    )
    ei_batch_variance = (
        proposal_diagnostics.groupby(
            [
                "depth",
                "smac_seed",
                "fit_serial",
                "model_training_rows",
            ],
            as_index=False,
        )
        .agg(
            selected_proposal_count=(
                "expected_improvement",
                "size",
            ),
            selected_proposal_ei_mean=(
                "expected_improvement",
                "mean",
            ),
            selected_proposal_ei_variance=(
                "expected_improvement",
                lambda values: float(np.var(values, ddof=0)),
            ),
        )
        .sort_values(["depth", "model_training_rows"])
    )

    spec = get_benchmark_spec(benchmark_key)
    return AnalysisBundle(
        benchmark_key=benchmark_key,
        display_name=spec.display_name,
        experiment_directory=experiment_directory,
        n_trials=effective_trials,
        training_instance_count=len(validator.data.training_instances),
        validation_quantile_seed=validation_quantile_seed,
        trajectory_events=trajectory_events.reset_index(drop=True),
        final_incumbents=final_incumbents,
        proposal_diagnostics=proposal_diagnostics.reset_index(drop=True),
        ei_batch_variance=ei_batch_variance.reset_index(drop=True),
        validation_cache_file=validator.cache_file,
    )


def trajectory_series(
    trajectory_events: pd.DataFrame,
    depth: int,
    smac_seed: int,
    *,
    n_trials: int = N_TRIALS,
) -> np.ndarray:
    selected = trajectory_events[
        (trajectory_events["depth"] == depth)
        & (trajectory_events["smac_seed"] == smac_seed)
    ].sort_values("trial")
    if selected.empty:
        raise RuntimeError(
            f"No trajectory for depth={depth}, seed={smac_seed}."
        )
    trials = selected["trial"].to_numpy(dtype=int)
    values = selected["full_training_par10"].to_numpy(dtype=float)
    grid = np.arange(1, n_trials + 1)
    positions = np.searchsorted(trials, grid, side="right") - 1
    if np.any(positions < 0):
        raise RuntimeError("Trajectory does not begin at trial one.")
    return values[positions]


def write_analysis_tables(bundle: AnalysisBundle) -> tuple[Path, ...]:
    output_directory = bundle.experiment_directory / "analytics_cache"
    output_directory.mkdir(parents=True, exist_ok=True)
    tables = (
        (
            output_directory / "incumbent_trajectory_events.csv",
            bundle.trajectory_events,
        ),
        (
            output_directory / "final_incumbents.csv",
            bundle.final_incumbents,
        ),
        (
            output_directory
            / "selected_seed_proposal_diagnostics.csv",
            bundle.proposal_diagnostics,
        ),
        (
            output_directory
            / "selected_seed_ei_batch_variance.csv",
            bundle.ei_batch_variance,
        ),
    )
    for path, dataframe in tables:
        dataframe.to_csv(path, index=False)
    return tuple(path for path, _ in tables)


def default_depth_colors() -> dict[int, str]:
    return {
        5: "#4c78a8",
        10: "#f58518",
        15: "#54a24b",
        20: "#e45756",
        30: "#b279a2",
    }


def plot_incumbents_by_seed(
    bundle: AnalysisBundle,
    *,
    colors: Mapping[int, str] | None = None,
    yscale: str = "linear",
) -> list[Any]:
    import matplotlib.pyplot as plt

    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be either 'linear' or 'log'.")
    colors = dict(colors or default_depth_colors())
    figures = []
    trials = np.arange(1, bundle.n_trials + 1)
    for smac_seed in SMAC_SEEDS:
        figure, axis = plt.subplots(figsize=(11, 5.5))
        for depth in DEPTHS:
            axis.step(
                trials,
                trajectory_series(
                    bundle.trajectory_events,
                    depth,
                    smac_seed,
                    n_trials=bundle.n_trials,
                ),
                where="post",
                color=colors[depth],
                label=f"depth {depth}",
            )
        axis.set(
            xlabel="Completed SMAC trial",
            ylabel="Incumbent mean PAR10 on all training instances",
            title=(
                f"{bundle.display_name}: incumbent trajectory, "
                f"SMAC seed {smac_seed} ({yscale} performance scale)"
            ),
        )
        axis.set_yscale(yscale)
        axis.grid(alpha=0.25)
        axis.legend(ncol=len(DEPTHS))
        figure.tight_layout()
        figures.append(figure)
    return figures


def plot_incumbent_confidence_intervals(
    bundle: AnalysisBundle,
    *,
    colors: Mapping[int, str] | None = None,
) -> Any:
    import matplotlib.pyplot as plt
    from scipy.stats import t

    colors = dict(colors or default_depth_colors())
    figure, axis = plt.subplots(figsize=(11, 6))
    trials = np.arange(1, bundle.n_trials + 1)
    for depth in DEPTHS:
        matrix = np.vstack(
            [
                trajectory_series(
                    bundle.trajectory_events,
                    depth,
                    smac_seed,
                    n_trials=bundle.n_trials,
                )
                for smac_seed in SMAC_SEEDS
            ]
        )
        mean = np.mean(matrix, axis=0)
        half_width = (
            t.ppf(0.975, len(SMAC_SEEDS) - 1)
            * np.std(matrix, axis=0, ddof=1)
            / math.sqrt(len(SMAC_SEEDS))
        )
        axis.plot(
            trials,
            mean,
            color=colors[depth],
            label=f"depth {depth}",
        )
        axis.fill_between(
            trials,
            mean - half_width,
            mean + half_width,
            color=colors[depth],
            alpha=0.18,
        )
    axis.set(
        xlabel="Completed SMAC trial",
        ylabel="Mean incumbent PAR10 on all training instances",
        title=(
            f"{bundle.display_name}: mean incumbent trajectory with "
            "95% confidence intervals"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=len(DEPTHS))
    figure.tight_layout()
    return figure


def best_depth_window_counts(
    bundle: AnalysisBundle,
    *,
    window_size: int = 1_000,
) -> tuple[dict[int, np.ndarray], np.ndarray, list[str]]:
    """Count tie-adjusted best-depth occupancy in consecutive trial windows."""
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    window_starts = np.arange(0, bundle.n_trials, window_size)
    window_ends = np.minimum(
        window_starts + window_size,
        bundle.n_trials,
    )
    window_labels = [
        f"{start + 1:,}–{end:,}"
        for start, end in zip(window_starts, window_ends)
    ]
    counts_by_seed: dict[int, np.ndarray] = {}
    for smac_seed in SMAC_SEEDS:
        values = np.vstack(
            [
                trajectory_series(
                    bundle.trajectory_events,
                    depth,
                    smac_seed,
                    n_trials=bundle.n_trials,
                )
                for depth in DEPTHS
            ]
        )
        best_values = np.min(values, axis=0, keepdims=True)
        tied = np.isclose(
            values,
            best_values,
            rtol=1e-10,
            atol=1e-12,
        )
        trial_credit = tied / tied.sum(axis=0, keepdims=True)
        counts = np.vstack(
            [
                trial_credit[:, start:end].sum(axis=1)
                for start, end in zip(window_starts, window_ends)
            ]
        )
        expected = window_ends - window_starts
        if not np.allclose(counts.sum(axis=1), expected):
            raise RuntimeError(
                "Tie-adjusted best-depth window counts have wrong totals."
            )
        counts_by_seed[smac_seed] = counts

    mean_counts = np.mean(
        np.stack(list(counts_by_seed.values())),
        axis=0,
    )
    return counts_by_seed, mean_counts, window_labels


def plot_best_depth_window_counts(
    bundle: AnalysisBundle,
    *,
    window_size: int = 1_000,
    colors: Mapping[int, str] | None = None,
) -> list[Any]:
    """Plot per-seed and across-seed mean best-depth occupancy."""
    import matplotlib.pyplot as plt

    colors = dict(colors or default_depth_colors())
    counts_by_seed, mean_counts, labels = best_depth_window_counts(
        bundle,
        window_size=window_size,
    )
    plots = [
        (
            counts_by_seed[smac_seed],
            f"{bundle.display_name}: best incumbent depth, "
            f"SMAC seed {smac_seed}",
            "Tie-adjusted number of best-incumbent trials",
        )
        for smac_seed in SMAC_SEEDS
    ]
    plots.append(
        (
            mean_counts,
            f"{bundle.display_name}: mean best-depth occupancy "
            "across SMAC seeds",
            "Mean tie-adjusted trial count across SMAC seeds",
        )
    )

    figures = []
    group_positions = np.arange(len(labels))
    bar_width = 0.15
    for counts, title, ylabel in plots:
        figure, axis = plt.subplots(figsize=(12, 6))
        for depth_index, depth in enumerate(DEPTHS):
            offsets = (
                group_positions
                + (depth_index - (len(DEPTHS) - 1) / 2) * bar_width
            )
            axis.bar(
                offsets,
                counts[:, depth_index],
                width=bar_width,
                color=colors[depth],
                label=f"depth {depth}",
            )
        axis.set(
            xticks=group_positions,
            xticklabels=labels,
            xlabel="Completed SMAC trial window",
            ylabel=ylabel,
            title=title,
            ylim=(0, window_size),
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(ncol=len(DEPTHS))
        figure.tight_layout()
        figures.append(figure)
    return figures


def plot_final_incumbents(bundle: AnalysisBundle) -> Any:
    import matplotlib.pyplot as plt

    horizon_label = (
        "final incumbent"
        if bundle.n_trials == N_TRIALS
        else f"incumbent at common trial {bundle.n_trials:,}"
    )
    figure, axis = plt.subplots(figsize=(9, 5.5))
    seed_markers = {0: "o", 1: "s", 2: "^"}
    offsets = {0: -0.22, 1: 0.0, 2: 0.22}
    for smac_seed in SMAC_SEEDS:
        selected = bundle.final_incumbents[
            bundle.final_incumbents["smac_seed"] == smac_seed
        ].sort_values("depth")
        axis.scatter(
            selected["depth"].to_numpy(dtype=float)
            + offsets[smac_seed],
            selected["full_training_par10"],
            marker=seed_markers[smac_seed],
            s=70,
            label=f"SMAC seed {smac_seed}",
        )
    axis.set(
        xticks=DEPTHS,
        xlabel="SMAC random-forest maximum depth",
        ylabel=(
            f"{horizon_label.capitalize()} mean PAR10 on all "
            "training instances"
        ),
        title=f"{bundle.display_name}: {horizon_label} by RF depth",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_selected_seed_diagnostics(
    bundle: AnalysisBundle,
    *,
    selected_seed: int,
    colors: Mapping[int, str] | None = None,
) -> Any:
    import matplotlib.pyplot as plt

    if selected_seed not in SMAC_SEEDS:
        raise ValueError(f"selected_seed must be one of {SMAC_SEEDS}.")
    colors = dict(colors or default_depth_colors())
    figure, axes = plt.subplots(
        8,
        1,
        figsize=(13, 30),
        sharex=True,
    )
    trials = np.arange(1, bundle.n_trials + 1)
    diagnostics = bundle.proposal_diagnostics[
        bundle.proposal_diagnostics["smac_seed"] == selected_seed
    ]
    batch_variance = bundle.ei_batch_variance[
        bundle.ei_batch_variance["smac_seed"] == selected_seed
    ]

    for depth in DEPTHS:
        color = colors[depth]
        axes[0].step(
            trials,
            trajectory_series(
                bundle.trajectory_events,
                depth,
                selected_seed,
                n_trials=bundle.n_trials,
            ),
            where="post",
            color=color,
            label=f"depth {depth}",
        )
        selected = diagnostics[
            diagnostics["depth"] == depth
        ].sort_values("proposal_trial")
        x = selected["proposal_trial"]
        axes[1].plot(
            x,
            selected["expected_improvement"],
            color=color,
            alpha=0.8,
        )
        axes[2].plot(
            x,
            selected["predicted_full_training_par10"],
            color=color,
            alpha=0.8,
        )
        axes[3].plot(
            x,
            selected["prediction_variance"],
            color=color,
            alpha=0.8,
        )
        axes[4].plot(
            x,
            selected["absolute_error"],
            color=color,
            alpha=0.8,
        )
        axes[5].plot(
            x,
            selected["standardized_error"],
            color=color,
            alpha=0.8,
        )
        axes[6].plot(
            x,
            selected["average_actual_tree_depth"],
            color=color,
            alpha=0.8,
        )
        selected_batches = batch_variance[
            batch_variance["depth"] == depth
        ].sort_values("model_training_rows")
        axes[7].plot(
            selected_batches["model_training_rows"],
            selected_batches["selected_proposal_ei_variance"],
            color=color,
            marker=".",
            markersize=3,
            alpha=0.85,
        )

    axes[0].set_ylabel("Incumbent\nPAR10")
    axes[0].set_title(
        "Current incumbent, re-evaluated on every training instance"
    )
    axes[0].legend(ncol=len(DEPTHS))
    axes[1].set_ylabel("EI")
    axes[1].set_title("Expected Improvement of each evaluated proposal")
    axes[2].set_ylabel("Predicted\nPAR10")
    axes[2].set_title(
        "SMAC RF prediction marginalized over training instances"
    )
    axes[3].set_ylabel("Prediction\nvariance")
    axes[3].set_title("SMAC RF predictive variance")
    axes[3].set_yscale("log")
    axes[4].set_ylabel("Absolute\nerror")
    axes[4].set_title(
        "|full-training validation PAR10 − SMAC RF prediction|"
    )
    axes[5].axhline(0.0, color="black", linewidth=0.8)
    axes[5].set_ylabel("Standardized\nerror")
    axes[5].set_title(
        "(validation PAR10 − prediction) / predictive standard deviation"
    )
    axes[6].set_ylabel("Mean tree\ndepth")
    axes[6].set_title("Average actual depth across all 100 fitted RF trees")
    axes[7].set_ylabel("Selected-\nproposal EI\nvariance")
    axes[7].set_title(
        "Population variance of EI across proposals selected from each RF fit"
    )
    axes[7].set_yscale("symlog", linthresh=1e-12)
    axes[7].set_xlabel("Completed SMAC trial")
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle(
        f"{bundle.display_name}: aligned diagnostics for SMAC seed "
        f"{selected_seed}",
        y=0.995,
        fontsize=15,
    )
    figure.tight_layout()
    return figure

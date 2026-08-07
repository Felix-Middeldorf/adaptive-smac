"""Training/test incumbent analytics for minimal native-SMAC ACLib runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epm.experiment_utils.data_handling import unwarp

from surrogate_analytics import (
    canonical_configuration,
    configuration_fingerprint,
)
from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)


DEPTHS = (5, 10, 20, 30)
SEEDS = (0, 1)
PCA_MODES = ("pca_none", "pca_4")
VALIDATION_QUANTILE_SEED = 0
ANALYTICS_VERSION = 1


@dataclass(frozen=True)
class RawRunAnalysis:
    experiment_directory: Path
    benchmark_key: str
    display_name: str
    run_summary: pd.DataFrame
    trajectories: pd.DataFrame
    final_incumbents: pd.DataFrame
    cache_files: dict[str, Path]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class FullInstanceValidator:
    """Evaluate configurations over one fixed ACLib instance split."""

    def __init__(
        self,
        benchmark_key: str,
        experiment_directory: Path,
        split: str,
    ) -> None:
        if split not in {"training", "test"}:
            raise ValueError("split must be 'training' or 'test'.")
        self.spec = get_benchmark_spec(benchmark_key)
        self.data = load_benchmark_data(self.spec)
        self.split = split
        self.instances = (
            self.data.training_instances
            if split == "training"
            else self.data.test_instances
        )
        self.features = np.asarray(
            [self.data.features[instance] for instance in self.instances],
            dtype=np.float32,
        )
        self.cache_file = (
            Path(experiment_directory)
            / "analytics_cache"
            / f"full_{split}_validation_qseed_0.json"
        )
        self._benchmark: ACLibSurrogateBenchmark | None = None
        self.results = self._load_cache()

    def _identity(self) -> dict[str, Any]:
        return {
            "analytics_version": ANALYTICS_VERSION,
            "benchmark": self.spec.key,
            "split": self.split,
            "instance_count": len(self.instances),
            "quantile_seed": VALIDATION_QUANTILE_SEED,
            "asset_signature": asset_metadata(self.spec),
        }

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_file.is_file():
            return {}
        payload = _read_json(self.cache_file)
        expected = self._identity()
        found = {key: payload.get(key) for key in expected}
        if found != expected:
            raise RuntimeError(
                f"Incompatible validation cache {self.cache_file}."
            )
        return {
            str(key): dict(value)
            for key, value in payload.get("results", {}).items()
        }

    def _save(self) -> None:
        _atomic_json(
            self.cache_file,
            {**self._identity(), "results": self.results},
        )

    @property
    def benchmark(self) -> ACLibSurrogateBenchmark:
        if self._benchmark is None:
            self._benchmark = ACLibSurrogateBenchmark(self.spec)
        return self._benchmark

    def _model_matrix(self, configuration: Mapping[str, Any]) -> np.ndarray:
        first = self.benchmark._model_input(
            configuration,
            self.instances[0],
        ).reshape(-1)
        parameter_count = len(self.data.configspace)
        parameters = first[:parameter_count]
        return np.hstack(
            (
                np.repeat(
                    parameters[np.newaxis, :],
                    len(self.instances),
                    axis=0,
                ),
                self.features,
            )
        ).astype(np.float32)

    def evaluate(self, configuration: Mapping[str, Any]) -> dict[str, Any]:
        fingerprint = configuration_fingerprint(configuration)
        if fingerprint in self.results:
            return self.results[fingerprint]
        model = self.benchmark.surrogate.model
        raw_logged = model.predict(
            self._model_matrix(configuration),
            seed=VALIDATION_QUANTILE_SEED,
            num_samples=1,
        )
        predictions = np.asarray(
            unwarp(raw_logged, quality=False),
            dtype=float,
        ).reshape(-1)
        if len(predictions) != len(self.instances):
            raise RuntimeError("Prediction count differs from instance count.")
        if not np.all(np.isfinite(predictions)):
            raise RuntimeError("Non-finite surrogate prediction.")
        timed_out = predictions > self.spec.cutoff
        par10 = np.where(timed_out, self.spec.timeout_cost, predictions)
        result = {
            "mean_par10": float(np.mean(par10)),
            "median_par10": float(np.median(par10)),
            "timeout_count": int(np.count_nonzero(timed_out)),
            "instance_count": len(self.instances),
        }
        self.results[fingerprint] = result
        return result

    def evaluate_many(
        self,
        configurations: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        missing = [
            (fingerprint, configuration)
            for fingerprint, configuration in configurations.items()
            if fingerprint not in self.results
        ]
        if not missing:
            print(
                f"{self.split.title()} validation cache hit: "
                f"{len(configurations)} incumbent configurations."
            )
            return self.results
        print(
            f"Validating {len(missing)} new incumbents on all "
            f"{len(self.instances)} {self.split} instances."
        )
        started = time.perf_counter()
        for index, (fingerprint, configuration) in enumerate(missing, 1):
            if configuration_fingerprint(configuration) != fingerprint:
                raise RuntimeError("Configuration fingerprint mismatch.")
            self.evaluate(configuration)
            if index % 50 == 0 or index == len(missing):
                self._save()
                print(
                    f"  {index}/{len(missing)} in "
                    f"{time.perf_counter() - started:.1f}s"
                )
        return self.results


def _collect_native_runs(
    experiment_directory: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    summaries: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    configurations: dict[str, dict[str, Any]] = {}
    for pca_mode in PCA_MODES:
        pca_components = None if pca_mode == "pca_none" else 4
        for depth in DEPTHS:
            for seed in SEEDS:
                directory = (
                    experiment_directory
                    / "results"
                    / pca_mode
                    / f"depth_{depth}"
                    / str(seed)
                )
                scenario = _read_json(directory / "scenario.json")
                if scenario.get("deterministic") is not True:
                    raise RuntimeError(f"Non-deterministic raw run: {directory}")
                runhistory = _read_json(directory / "runhistory.json")
                intensifier = _read_json(directory / "intensifier.json")
                finished = int(runhistory["stats"]["finished"])
                trajectory = sorted(
                    intensifier.get("trajectory", []),
                    key=lambda item: int(item["trial"]),
                )
                if not trajectory:
                    raise RuntimeError(f"No incumbent trajectory: {directory}")
                recorded_trials = runhistory.get("data", [])
                evaluated_config_ids = {
                    int(trial["config_id"])
                    for trial in recorded_trials
                }
                final_incumbent_id = int(trajectory[-1]["config_ids"][0])
                final_incumbent_trials = [
                    trial
                    for trial in recorded_trials
                    if int(trial["config_id"]) == final_incumbent_id
                ]
                final_incumbent_instances = {
                    trial["instance"]
                    for trial in final_incumbent_trials
                    if trial.get("instance") is not None
                }
                summaries.append(
                    {
                        "pca_mode": pca_mode,
                        "pca_components": pca_components,
                        "depth": depth,
                        "smac_seed": seed,
                        "finished_trials": finished,
                        "recorded_trials": len(recorded_trials),
                        "evaluated_configurations": len(evaluated_config_ids),
                        "incumbent_changes": len(trajectory),
                        "final_incumbent_config_id": final_incumbent_id,
                        "final_incumbent_trials": len(final_incumbent_trials),
                        "final_incumbent_instances": len(
                            final_incumbent_instances
                        ),
                        "complete": finished >= 5_000,
                    }
                )
                for event in trajectory:
                    if len(event["config_ids"]) != 1:
                        raise RuntimeError(
                            f"Multiple incumbents unsupported: {directory}"
                        )
                    config_id = str(int(event["config_ids"][0]))
                    canonical = canonical_configuration(
                        runhistory["configs"][config_id]
                    )
                    fingerprint = configuration_fingerprint(canonical)
                    previous = configurations.setdefault(
                        fingerprint,
                        canonical,
                    )
                    if previous != canonical:
                        raise RuntimeError("Configuration fingerprint collision.")
                    events.append(
                        {
                            "pca_mode": pca_mode,
                            "pca_components": pca_components,
                            "depth": depth,
                            "smac_seed": seed,
                            "trial": int(event["trial"]),
                            "config_id": int(config_id),
                            "configuration_fingerprint": fingerprint,
                            "smac_subset_cost": float(event["costs"][0]),
                            "finished_trials": finished,
                            "is_extension": False,
                        }
                    )
                last = events[-1]
                if last["trial"] < finished:
                    events.append(
                        {
                            **last,
                            "trial": finished,
                            "is_extension": True,
                        }
                    )
    return summaries, events, configurations


def build_raw_run_analysis(
    experiment_directory: Path,
    benchmark_key: str,
) -> RawRunAnalysis:
    experiment_directory = Path(experiment_directory).resolve()
    spec = get_benchmark_spec(benchmark_key)
    summaries, event_rows, configurations = _collect_native_runs(
        experiment_directory
    )
    validators = {
        split: FullInstanceValidator(
            benchmark_key,
            experiment_directory,
            split,
        )
        for split in ("training", "test")
    }
    validated = {
        split: validator.evaluate_many(configurations)
        for split, validator in validators.items()
    }
    for row in event_rows:
        fingerprint = row["configuration_fingerprint"]
        for split in ("training", "test"):
            result = validated[split][fingerprint]
            row[f"{split}_mean_par10"] = float(result["mean_par10"])
            row[f"{split}_median_par10"] = float(result["median_par10"])
            row[f"{split}_timeouts"] = int(result["timeout_count"])
    trajectories = pd.DataFrame(event_rows).sort_values(
        ["pca_mode", "depth", "smac_seed", "trial"]
    ).reset_index(drop=True)
    final = (
        trajectories.sort_values("trial")
        .groupby(["pca_mode", "depth", "smac_seed"], as_index=False)
        .tail(1)
        .sort_values(["pca_mode", "depth", "smac_seed"])
        .reset_index(drop=True)
    )
    final["test_minus_training_par10"] = (
        final["test_mean_par10"] - final["training_mean_par10"]
    )
    run_summary = pd.DataFrame(summaries).sort_values(
        ["pca_mode", "depth", "smac_seed"]
    )
    training_instance_count = len(validators["training"].instances)
    run_summary["available_training_instances"] = training_instance_count
    run_summary["final_incumbent_instance_coverage_percent"] = (
        100.0
        * run_summary["final_incumbent_instances"]
        / training_instance_count
    )
    return RawRunAnalysis(
        experiment_directory=experiment_directory,
        benchmark_key=benchmark_key,
        display_name=spec.display_name,
        run_summary=run_summary,
        trajectories=trajectories,
        final_incumbents=final,
        cache_files={
            split: validator.cache_file
            for split, validator in validators.items()
        },
    )


def _split_column(split: str) -> str:
    if split not in {"training", "test"}:
        raise ValueError("split must be training or test.")
    return f"{split}_mean_par10"


def _policy_label(pca_mode: str, depth: int) -> str:
    pca = "no PCA" if pca_mode == "pca_none" else "PCA=4"
    return f"Depth {depth}, {pca}"


def _line_style(pca_mode: str) -> str:
    return "-" if pca_mode == "pca_none" else "--"


def plot_incumbents_per_seed(
    analysis: RawRunAnalysis,
    split: str,
    yscale: str = "linear",
) -> list[Any]:
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    column = _split_column(split)
    colors = dict(zip(DEPTHS, plt.cm.tab10.colors[: len(DEPTHS)]))
    figures = []
    for seed in SEEDS:
        figure, axis = plt.subplots(figsize=(11, 6))
        for pca_mode in PCA_MODES:
            for depth in DEPTHS:
                selected = analysis.trajectories[
                    (analysis.trajectories["pca_mode"] == pca_mode)
                    & (analysis.trajectories["depth"] == depth)
                    & (analysis.trajectories["smac_seed"] == seed)
                ]
                axis.step(
                    selected["trial"],
                    selected[column],
                    where="post",
                    color=colors[depth],
                    linestyle=_line_style(pca_mode),
                    linewidth=1.8,
                    label=_policy_label(pca_mode, depth),
                )
        axis.set_xlabel("Completed SMAC target trials")
        axis.set_ylabel(f"Mean PAR10 on all {split} instances")
        axis.set_yscale(yscale)
        axis.set_title(
            f"{analysis.display_name}: {split} incumbent trajectory, "
            f"SMAC seed {seed} ({yscale} y-scale)"
        )
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figures.append(figure)
    return figures


def _step_value_at(
    trials: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    indices = np.searchsorted(trials, query, side="right") - 1
    if np.any(indices < 0):
        raise RuntimeError("Query precedes the first incumbent event.")
    return values[indices]


def plot_incumbents_across_seeds(
    analysis: RawRunAnalysis,
    split: str,
    yscale: str = "linear",
) -> Any:
    if yscale not in {"linear", "log"}:
        raise ValueError("yscale must be 'linear' or 'log'.")
    column = _split_column(split)
    colors = dict(zip(DEPTHS, plt.cm.tab10.colors[: len(DEPTHS)]))
    figure, axis = plt.subplots(figsize=(11, 6))
    for pca_mode in PCA_MODES:
        for depth in DEPTHS:
            seed_frames = [
                analysis.trajectories[
                    (analysis.trajectories["pca_mode"] == pca_mode)
                    & (analysis.trajectories["depth"] == depth)
                    & (analysis.trajectories["smac_seed"] == seed)
                ].sort_values("trial")
                for seed in SEEDS
            ]
            horizon = min(int(frame["trial"].max()) for frame in seed_frames)
            query = np.unique(
                np.concatenate(
                    [
                        frame.loc[frame["trial"] <= horizon, "trial"].to_numpy()
                        for frame in seed_frames
                    ]
                    + [np.asarray([horizon])]
                )
            )
            query = query[
                query
                >= max(int(frame["trial"].min()) for frame in seed_frames)
            ]
            values = np.vstack(
                [
                    _step_value_at(
                        frame["trial"].to_numpy(dtype=int),
                        frame[column].to_numpy(dtype=float),
                        query,
                    )
                    for frame in seed_frames
                ]
            )
            mean = values.mean(axis=0)
            axis.step(
                query,
                mean,
                where="post",
                color=colors[depth],
                linestyle=_line_style(pca_mode),
                linewidth=1.8,
                label=_policy_label(pca_mode, depth),
            )
            axis.fill_between(
                query,
                values.min(axis=0),
                values.max(axis=0),
                step="post",
                color=colors[depth],
                alpha=0.07,
            )
    axis.set_xlabel("Completed SMAC target trials")
    axis.set_ylabel(f"Mean PAR10 on all {split} instances")
    axis.set_yscale(yscale)
    axis.set_title(
        f"{analysis.display_name}: {split} incumbent trajectories, "
        f"mean and range over SMAC seeds ({yscale} y-scale)"
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    return figure


def plot_final_train_test(
    analysis: RawRunAnalysis,
    scale: str = "linear",
) -> Any:
    if scale not in {"linear", "log"}:
        raise ValueError("scale must be 'linear' or 'log'.")
    frame = analysis.final_incumbents
    colors = dict(zip(DEPTHS, plt.cm.tab10.colors[: len(DEPTHS)]))
    figure, axis = plt.subplots(figsize=(8, 7))
    for pca_mode, marker in zip(PCA_MODES, ("o", "s")):
        selected = frame[frame["pca_mode"] == pca_mode]
        axis.scatter(
            selected["training_mean_par10"],
            selected["test_mean_par10"],
            c=[colors[int(depth)] for depth in selected["depth"]],
            marker=marker,
            s=65,
            alpha=0.8,
            label="No PCA" if pca_mode == "pca_none" else "PCA=4",
        )
    low = float(
        min(frame["training_mean_par10"].min(), frame["test_mean_par10"].min())
    )
    high = float(
        max(frame["training_mean_par10"].max(), frame["test_mean_par10"].max())
    )
    axis.plot([low, high], [low, high], color="black", linestyle=":")
    axis.set_xlabel("Final incumbent mean training PAR10")
    axis.set_ylabel("Final incumbent mean test PAR10")
    axis.set_xscale(scale)
    axis.set_yscale(scale)
    axis.set_title(
        f"{analysis.display_name}: final training versus test performance "
        f"({scale} scale)"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure

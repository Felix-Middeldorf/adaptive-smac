"""Reproducible analytics for the two 50-schedule Clasp RF experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surrogate_analytics import FullTrainingValidator
from surrogate_telemetry import configuration_fingerprint


CHECKPOINTS = (500, 2_000, 5_000)
SEEDS = (0, 1, 2)
PHASE_FACTORS = ("depth", "min_samples_split", "feature_ratio")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _phase_columns(schedule: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase, settings in enumerate(schedule["phases"]):
        for factor in PHASE_FACTORS:
            result[f"phase_{phase}_{factor}"] = settings[factor]
    return result


def build_checkpoint_table(
    experiment_directory: Path,
    benchmark_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate historical incumbents at trials 500, 2000, and 5000."""
    experiment_directory = Path(experiment_directory).resolve()
    catalog = _read_json(experiment_directory / "schedule_catalog.json")
    schedules = {
        int(schedule["index"]): schedule for schedule in catalog["schedules"]
    }
    configurations: dict[str, dict[str, Any]] = {}
    pending_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for schedule_index in range(50):
        schedule = schedules[schedule_index]
        phase_columns = _phase_columns(schedule)
        for seed in SEEDS:
            directory = (
                experiment_directory
                / "results"
                / f"schedule_{schedule_index:02d}"
                / str(seed)
            )
            completion = _read_json(directory / "completed.json")
            summary = _read_json(directory / "summary.json")
            telemetry_summary = _read_json(
                directory / "configuration_telemetry_summary.json"
            )
            if completion.get("state") != "complete":
                raise RuntimeError(f"Incomplete run: {directory}")
            if int(summary.get("finished_trials", -1)) != 5_000:
                raise RuntimeError(f"Wrong trial count: {directory}")
            if telemetry_summary.get("missing_proposals") != []:
                raise RuntimeError(f"Missing proposal telemetry: {directory}")
            if telemetry_summary.get("missing_first_completions") != []:
                raise RuntimeError(f"Missing completion telemetry: {directory}")
            if int(telemetry_summary.get("telemetry_error_records", -1)) != 0:
                raise RuntimeError(f"Telemetry errors: {directory}")

            runhistory = _read_json(directory / "runhistory.json")
            trajectory = sorted(
                _read_json(directory / "trajectory.json"),
                key=lambda event: int(event["trial"]),
            )
            transitions = _read_json(
                directory / "rf_schedule_state.json"
            )["transitions"]
            if [int(item["phase"]) for item in transitions] != [0, 1, 2]:
                raise RuntimeError(f"Incomplete phase transitions: {directory}")

            run_rows.append(
                {
                    "benchmark": benchmark_key,
                    "schedule": schedule_index,
                    "smac_seed": seed,
                    "walltime_seconds": float(
                        summary["walltime_seconds_this_process"]
                    ),
                    "configurations": int(summary["configurations"]),
                    "target_evaluations": int(
                        summary["target_evaluations_this_process"]
                    ),
                    "target_timeouts": int(
                        summary["target_timeouts_this_process"]
                    ),
                    "phase_1_activation_trial": int(
                        transitions[1]["completed_trials"]
                    ),
                    "phase_2_activation_trial": int(
                        transitions[2]["completed_trials"]
                    ),
                    **phase_columns,
                }
            )

            for checkpoint in CHECKPOINTS:
                eligible = [
                    event
                    for event in trajectory
                    if int(event["trial"]) <= checkpoint
                ]
                if not eligible:
                    raise RuntimeError(
                        f"No incumbent by trial {checkpoint}: {directory}"
                    )
                event = eligible[-1]
                if len(event["config_ids"]) != 1:
                    raise RuntimeError(
                        f"Multiple incumbents are unsupported: {directory}"
                    )
                config_id = str(int(event["config_ids"][0]))
                configuration = runhistory["configs"][config_id]
                fingerprint = configuration_fingerprint(configuration)
                configurations.setdefault(fingerprint, configuration)
                pending_rows.append(
                    {
                        "benchmark": benchmark_key,
                        "schedule": schedule_index,
                        "smac_seed": seed,
                        "checkpoint": checkpoint,
                        "incumbent_event_trial": int(event["trial"]),
                        "config_id": int(config_id),
                        "configuration_fingerprint": fingerprint,
                        "smac_incumbent_subset_cost": float(
                            event["costs"][0]
                        ),
                        **phase_columns,
                    }
                )

    validator = FullTrainingValidator(
        benchmark_key,
        experiment_directory,
        quantile_seed=0,
    )
    validation = validator.evaluate_many(configurations)
    for row in pending_rows:
        validated = validation[row["configuration_fingerprint"]]
        row["full_training_par10"] = float(validated["mean_par10"])
        row["median_training_par10"] = float(validated["median_par10"])
        row["timeout_count"] = int(validated["timeout_count"])
        row["training_instance_count"] = int(
            validated["training_instance_count"]
        )

    checkpoints = pd.DataFrame(pending_rows).sort_values(
        ["schedule", "smac_seed", "checkpoint"]
    )
    runs = pd.DataFrame(run_rows).sort_values(["schedule", "smac_seed"])
    cache = experiment_directory / "analytics_cache"
    _atomic_csv(checkpoints, cache / "checkpoint_incumbents.csv")
    _atomic_csv(runs, cache / "run_summary.csv")
    return checkpoints, runs


def schedule_summary(
    checkpoints: pd.DataFrame,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint = checkpoints.pivot(
        index=["benchmark", "schedule", "smac_seed"],
        columns="checkpoint",
        values="full_training_par10",
    ).reset_index()
    endpoint = endpoint.rename(
        columns={checkpoint: f"par10_{checkpoint}" for checkpoint in CHECKPOINTS}
    )
    endpoint["improvement_500_2000"] = (
        endpoint["par10_500"] - endpoint["par10_2000"]
    )
    endpoint["improvement_2000_5000"] = (
        endpoint["par10_2000"] - endpoint["par10_5000"]
    )
    endpoint["improvement_500_5000"] = (
        endpoint["par10_500"] - endpoint["par10_5000"]
    )
    endpoint = endpoint.merge(
        runs,
        on=["benchmark", "schedule", "smac_seed"],
        validate="one_to_one",
    )

    aggregations: dict[str, tuple[str, str]] = {}
    for checkpoint in CHECKPOINTS:
        aggregations[f"par10_{checkpoint}_mean"] = (
            f"par10_{checkpoint}",
            "mean",
        )
        aggregations[f"par10_{checkpoint}_std"] = (
            f"par10_{checkpoint}",
            "std",
        )
    for column in (
        "improvement_500_2000",
        "improvement_2000_5000",
        "improvement_500_5000",
        "walltime_seconds",
        "configurations",
    ):
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    summary = (
        endpoint.groupby(["benchmark", "schedule"], as_index=False)
        .agg(**aggregations)
        .sort_values("par10_5000_mean")
    )
    summary["final_rank"] = summary["par10_5000_mean"].rank(
        method="min"
    ).astype(int)
    return endpoint, summary


def phase_associations(endpoint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phase_metrics = {
        0: ("par10_500", False),
        1: ("improvement_500_2000", True),
        2: ("improvement_2000_5000", True),
    }
    for phase, (metric, larger_is_better) in phase_metrics.items():
        for factor in PHASE_FACTORS:
            column = f"phase_{phase}_{factor}"
            for value, group in endpoint.groupby(column):
                rows.append(
                    {
                        "benchmark": str(group["benchmark"].iloc[0]),
                        "phase": phase,
                        "factor": factor,
                        "value": value,
                        "metric": metric,
                        "larger_is_better": larger_is_better,
                        "runs": len(group),
                        "schedules": group["schedule"].nunique(),
                        "mean": float(group[metric].mean()),
                        "median": float(group[metric].median()),
                        "std": float(group[metric].std()),
                        "final_mean": float(group["par10_5000"].mean()),
                        "final_median": float(group["par10_5000"].median()),
                        "final_std": float(group["par10_5000"].std()),
                    }
                )
    return pd.DataFrame(rows)


def plot_final_performance(
    endpoint: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    display_name: str,
) -> None:
    order = summary.sort_values("par10_5000_mean")["schedule"].tolist()
    positions = {schedule: index for index, schedule in enumerate(order)}
    final = endpoint.copy()
    final["x"] = final["schedule"].map(positions)
    means = summary.set_index("schedule").loc[order, "par10_5000_mean"]

    figure, axis = plt.subplots(figsize=(15, 6))
    for seed, marker in zip(SEEDS, ("o", "s", "^")):
        selected = final[final["smac_seed"] == seed]
        axis.scatter(
            selected["x"],
            selected["par10_5000"],
            s=28,
            alpha=0.72,
            marker=marker,
            label=f"SMAC seed {seed}",
        )
    axis.plot(range(len(order)), means, color="black", linewidth=1.2, label="Mean")
    axis.set_xticks(range(len(order)))
    axis.set_xticklabels([f"{item:02d}" for item in order], rotation=90)
    axis.set_xlabel("Schedule, ordered by mean final PAR10")
    axis.set_ylabel("Full-training PAR10 at trial 5000 (lower is better)")
    axis.set_title(f"{display_name}: final performance of all 50 RF schedules")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_top_trajectories(
    endpoint: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
    display_name: str,
    top_n: int = 10,
) -> None:
    selected_schedules = summary.nsmallest(
        top_n, "par10_5000_mean"
    )["schedule"].tolist()
    means = (
        endpoint[endpoint["schedule"].isin(selected_schedules)]
        .groupby(["schedule", "smac_seed"])[
            ["par10_500", "par10_2000", "par10_5000"]
        ]
        .first()
        .groupby("schedule")
        .mean()
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    for schedule in selected_schedules:
        axis.plot(
            CHECKPOINTS,
            means.loc[schedule].to_numpy(dtype=float),
            marker="o",
            label=f"{schedule:02d}",
        )
    axis.set_xlabel("SMAC trial")
    axis.set_ylabel("Mean full-training incumbent PAR10")
    axis.set_title(f"{display_name}: checkpoint trajectories of top schedules")
    axis.grid(alpha=0.25)
    axis.legend(title="Schedule", ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def analyze_benchmark(
    experiment_directory: Path,
    benchmark_key: str,
    display_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment_directory = Path(experiment_directory).resolve()
    checkpoints, runs = build_checkpoint_table(
        experiment_directory,
        benchmark_key,
    )
    endpoint, summary = schedule_summary(checkpoints, runs)
    associations = phase_associations(endpoint)
    cache = experiment_directory / "analytics_cache"
    _atomic_csv(endpoint, cache / "schedule_seed_endpoints.csv")
    _atomic_csv(summary, cache / "schedule_summary.csv")
    _atomic_csv(associations, cache / "phase_factor_associations.csv")
    plot_final_performance(
        endpoint,
        summary,
        cache / "final_schedule_performance.png",
        display_name,
    )
    plot_top_trajectories(
        endpoint,
        summary,
        cache / "top10_checkpoint_trajectories.png",
        display_name,
    )
    return endpoint, summary, associations


def cross_benchmark_summary(
    summaries: dict[str, pd.DataFrame],
    output_directory: Path,
) -> pd.DataFrame:
    frames = []
    for label, summary in summaries.items():
        selected = summary[
            ["schedule", "par10_5000_mean", "par10_5000_std", "final_rank"]
        ].copy()
        selected = selected.rename(
            columns={
                "par10_5000_mean": f"{label}_mean",
                "par10_5000_std": f"{label}_std",
                "final_rank": f"{label}_rank",
            }
        )
        frames.append(selected)
    merged = frames[0].merge(frames[1], on="schedule", validate="one_to_one")
    rank_columns = [column for column in merged if column.endswith("_rank")]
    merged["mean_rank"] = merged[rank_columns].mean(axis=1)
    merged["worst_rank"] = merged[rank_columns].max(axis=1)
    merged = merged.sort_values(["mean_rank", "worst_rank", "schedule"])
    output_directory.mkdir(parents=True, exist_ok=True)
    _atomic_csv(merged, output_directory / "cross_benchmark_schedule_summary.csv")

    labels = list(summaries)
    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(
        merged[f"{labels[0]}_rank"],
        merged[f"{labels[1]}_rank"],
        alpha=0.75,
    )
    for row in merged.nsmallest(8, "mean_rank").itertuples():
        axis.annotate(
            f"{int(row.schedule):02d}",
            (
                getattr(row, f"{labels[0]}_rank"),
                getattr(row, f"{labels[1]}_rank"),
            ),
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set_xlabel(f"{labels[0]} final rank")
    axis.set_ylabel(f"{labels[1]} final rank")
    axis.set_title("Cross-benchmark robustness of RF schedules")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_directory / "cross_benchmark_schedule_ranks.png",
        dpi=170,
    )
    plt.close(figure)
    return merged

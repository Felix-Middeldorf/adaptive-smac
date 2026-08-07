"""Notebook-facing analytics for the two brute-force RF schedule studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surrogate_analytics import FullTrainingValidator
from surrogate_telemetry import configuration_fingerprint


CHECKPOINTS = (500, 2_000, 5_000)
SEEDS = (0, 1, 2)
DEPTHS = (5, 10, 15, 20, 30)
PHASE_FACTORS = ("depth", "min_samples_split", "feature_ratio")


@dataclass(frozen=True)
class ScheduleNotebookData:
    experiment_directory: Path
    benchmark_key: str
    schedule_rows: pd.DataFrame
    policy_rows: pd.DataFrame
    seed_rankings: pd.DataFrame
    aggregate_rankings: pd.DataFrame
    phase_effects: pd.DataFrame
    baseline_comparison: pd.DataFrame


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint_event(
    trajectory: list[dict[str, Any]],
    checkpoint: int,
) -> dict[str, Any]:
    eligible = [
        event for event in trajectory if int(event["trial"]) <= checkpoint
    ]
    if not eligible:
        raise RuntimeError(f"No incumbent exists by trial {checkpoint}.")
    event = max(eligible, key=lambda item: int(item["trial"]))
    if len(event["config_ids"]) != 1:
        raise RuntimeError("Expected one incumbent in a single-objective run.")
    return event


def build_fixed_checkpoint_table(
    fixed_directory: Path,
    benchmark_key: str,
) -> pd.DataFrame:
    """Load or calculate fixed-depth incumbent values at the three checkpoints."""
    fixed_directory = Path(fixed_directory).resolve()
    cache = fixed_directory / "analytics_cache" / "checkpoint_incumbents.csv"
    required = {
        "benchmark",
        "depth",
        "smac_seed",
        "checkpoint",
        "full_training_par10",
    }
    if cache.is_file():
        frame = pd.read_csv(cache)
        if required.issubset(frame.columns) and len(frame) == 45:
            return frame.sort_values(
                ["depth", "smac_seed", "checkpoint"]
            ).reset_index(drop=True)

    configurations: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for depth in DEPTHS:
        for seed in SEEDS:
            directory = fixed_directory / "results" / f"depth_{depth}" / str(seed)
            completion = _read_json(directory / "completed.json")
            if completion.get("state") != "complete":
                raise RuntimeError(f"Incomplete fixed-depth run: {directory}")
            runhistory = _read_json(directory / "runhistory.json")
            if int(runhistory["stats"]["finished"]) != 5_000:
                raise RuntimeError(f"Wrong trial count in {directory}.")
            trajectory = sorted(
                _read_json(directory / "trajectory.json"),
                key=lambda item: int(item["trial"]),
            )
            for checkpoint in CHECKPOINTS:
                event = _checkpoint_event(trajectory, checkpoint)
                config_id = str(int(event["config_ids"][0]))
                configuration = runhistory["configs"][config_id]
                fingerprint = configuration_fingerprint(configuration)
                configurations.setdefault(fingerprint, configuration)
                pending.append(
                    {
                        "benchmark": benchmark_key,
                        "depth": depth,
                        "smac_seed": seed,
                        "checkpoint": checkpoint,
                        "event_trial": int(event["trial"]),
                        "configuration_fingerprint": fingerprint,
                        "smac_subset_cost": float(event["costs"][0]),
                    }
                )

    validator = FullTrainingValidator(
        benchmark_key,
        fixed_directory,
        quantile_seed=0,
    )
    validated = validator.evaluate_many(configurations)
    for row in pending:
        result = validated[row["configuration_fingerprint"]]
        row["full_training_par10"] = float(result["mean_par10"])
        row["median_training_par10"] = float(result["median_par10"])
        row["timeout_count"] = int(result["timeout_count"])
        row["training_instance_count"] = int(
            result["training_instance_count"]
        )
    frame = pd.DataFrame(pending).sort_values(
        ["depth", "smac_seed", "checkpoint"]
    )
    _atomic_csv(frame, cache)
    return frame.reset_index(drop=True)


def _fixed_policy_rows(
    fixed_directory: Path,
    benchmark_key: str,
    *,
    pca_components: int | None,
    policy_prefix: str,
    policy_type: str,
) -> pd.DataFrame:
    checkpoints = build_fixed_checkpoint_table(
        fixed_directory,
        benchmark_key,
    )
    endpoint = (
        checkpoints.pivot(
            index=["benchmark", "depth", "smac_seed"],
            columns="checkpoint",
            values="full_training_par10",
        )
        .reset_index()
        .rename(
            columns={
                checkpoint: f"par10_{checkpoint}"
                for checkpoint in CHECKPOINTS
            }
        )
    )
    runtimes: list[dict[str, Any]] = []
    for depth in DEPTHS:
        for seed in SEEDS:
            summary = _read_json(
                fixed_directory
                / "results"
                / f"depth_{depth}"
                / str(seed)
                / "summary.json"
            )
            runtimes.append(
                {
                    "depth": depth,
                    "smac_seed": seed,
                    "walltime_seconds": float(
                        summary["walltime_seconds_this_process"]
                    ),
                    "configurations": float(
                        summary["configuration_telemetry"][
                            "unique_evaluated_configurations"
                        ]
                    ),
                }
            )
    endpoint = endpoint.merge(
        pd.DataFrame(runtimes),
        on=["depth", "smac_seed"],
        validate="one_to_one",
    )
    endpoint["policy_id"] = endpoint["depth"].map(
        lambda depth: f"{policy_prefix}_d{int(depth):02d}"
    )
    endpoint["policy_label"] = endpoint["depth"].map(
        lambda depth: (
            f"Fixed d{int(depth)} "
            f"({'no PCA' if pca_components is None else 'PCA=4'})"
        )
    )
    endpoint["policy_type"] = policy_type
    endpoint["pca_components"] = (
        "none" if pca_components is None else str(pca_components)
    )
    endpoint["schedule"] = np.nan
    endpoint["phase_0_depth"] = endpoint["depth"]
    endpoint["phase_1_depth"] = endpoint["depth"]
    endpoint["phase_2_depth"] = endpoint["depth"]
    endpoint["phase_0_min_samples_split"] = 1
    endpoint["phase_1_min_samples_split"] = 1
    endpoint["phase_2_min_samples_split"] = 1
    endpoint["phase_0_feature_ratio"] = 5.0 / 6.0
    endpoint["phase_1_feature_ratio"] = 5.0 / 6.0
    endpoint["phase_2_feature_ratio"] = 5.0 / 6.0
    return endpoint


def _schedule_policy_rows(experiment_directory: Path) -> pd.DataFrame:
    path = (
        experiment_directory
        / "analytics_cache"
        / "schedule_seed_endpoints.csv"
    )
    frame = pd.read_csv(path)
    expected = 50 * len(SEEDS)
    if len(frame) != expected:
        raise RuntimeError(
            f"Expected {expected} schedule/seed rows in {path}, found {len(frame)}."
        )
    frame["policy_id"] = frame["schedule"].map(
        lambda index: f"schedule_{int(index):02d}"
    )
    frame["policy_label"] = frame["schedule"].map(
        lambda index: f"Schedule {int(index):02d}"
    )
    frame["policy_type"] = "three-phase schedule (PCA=4)"
    frame["pca_components"] = "4"
    frame["depth"] = np.nan
    return frame


def _add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["improvement_500_2000"] = (
        result["par10_500"] - result["par10_2000"]
    )
    result["improvement_2000_5000"] = (
        result["par10_2000"] - result["par10_5000"]
    )
    result["improvement_500_5000"] = (
        result["par10_500"] - result["par10_5000"]
    )
    return result


def _seed_rankings(policy_rows: pd.DataFrame) -> pd.DataFrame:
    ranked = policy_rows.copy()
    ranked["seed_rank"] = ranked.groupby("smac_seed")[
        "par10_5000"
    ].rank(method="min", ascending=True).astype(int)
    return ranked.sort_values(
        ["smac_seed", "seed_rank", "policy_id"]
    ).reset_index(drop=True)


def _aggregate_rankings(seed_rankings: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        seed_rankings.groupby(
            ["policy_id", "policy_label", "policy_type", "pca_components"],
            as_index=False,
        )
        .agg(
            final_mean=("par10_5000", "mean"),
            final_std=("par10_5000", "std"),
            final_median=("par10_5000", "median"),
            final_best=("par10_5000", "min"),
            final_worst=("par10_5000", "max"),
            mean_seed_rank=("seed_rank", "mean"),
            worst_seed_rank=("seed_rank", "max"),
            par10_500_mean=("par10_500", "mean"),
            par10_2000_mean=("par10_2000", "mean"),
            improvement_500_2000_mean=("improvement_500_2000", "mean"),
            improvement_2000_5000_mean=("improvement_2000_5000", "mean"),
            improvement_500_5000_mean=("improvement_500_5000", "mean"),
            walltime_hours_mean=(
                "walltime_seconds",
                lambda values: float(values.mean() / 3600.0),
            ),
            configurations_mean=("configurations", "mean"),
        )
        .sort_values(["final_mean", "final_std", "policy_id"])
        .reset_index(drop=True)
    )
    aggregate["overall_rank"] = np.arange(1, len(aggregate) + 1)
    return aggregate


def _phase_effects(schedule_rows: pd.DataFrame) -> pd.DataFrame:
    phase_metric = {
        0: ("par10_500", "lower is better"),
        1: ("improvement_500_2000", "higher is better"),
        2: ("improvement_2000_5000", "higher is better"),
    }
    rows: list[dict[str, Any]] = []
    for phase, (metric, direction) in phase_metric.items():
        for factor in PHASE_FACTORS:
            column = f"phase_{phase}_{factor}"
            for value, group in schedule_rows.groupby(column):
                rows.append(
                    {
                        "phase": phase,
                        "factor": factor,
                        "value": value,
                        "metric": metric,
                        "direction": direction,
                        "runs": len(group),
                        "schedules": group["schedule"].nunique(),
                        "mean": float(group[metric].mean()),
                        "median": float(group[metric].median()),
                        "std": float(group[metric].std()),
                        "final_mean": float(group["par10_5000"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _baseline_comparison(policy_rows: pd.DataFrame) -> pd.DataFrame:
    schedules = policy_rows[
        policy_rows["policy_type"] == "three-phase schedule (PCA=4)"
    ].copy()
    fixed_pca = policy_rows[
        policy_rows["policy_type"] == "fixed depth (PCA=4)"
    ]
    fixed_no_pca = policy_rows[
        policy_rows["policy_type"] == "fixed depth (no PCA)"
    ]
    best_pca = fixed_pca.groupby("smac_seed")["par10_5000"].min()
    best_no_pca = fixed_no_pca.groupby("smac_seed")["par10_5000"].min()
    schedules["delta_to_best_fixed_pca4"] = schedules.apply(
        lambda row: row["par10_5000"] - best_pca.loc[row["smac_seed"]],
        axis=1,
    )
    schedules["delta_to_best_fixed_no_pca"] = schedules.apply(
        lambda row: row["par10_5000"] - best_no_pca.loc[row["smac_seed"]],
        axis=1,
    )
    return (
        schedules.groupby(["policy_id", "policy_label"], as_index=False)
        .agg(
            mean_delta_to_best_fixed_pca4=(
                "delta_to_best_fixed_pca4",
                "mean",
            ),
            wins_vs_best_fixed_pca4=(
                "delta_to_best_fixed_pca4",
                lambda values: int((values < 0).sum()),
            ),
            mean_delta_to_best_fixed_no_pca=(
                "delta_to_best_fixed_no_pca",
                "mean",
            ),
            wins_vs_best_fixed_no_pca=(
                "delta_to_best_fixed_no_pca",
                lambda values: int((values < 0).sum()),
            ),
        )
        .sort_values(
            [
                "wins_vs_best_fixed_pca4",
                "mean_delta_to_best_fixed_pca4",
            ],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def build_notebook_data(
    experiment_directory: Path,
    benchmark_key: str,
) -> ScheduleNotebookData:
    experiment_directory = Path(experiment_directory).resolve()
    benchmark_directory = experiment_directory.parent
    schedules = _schedule_policy_rows(experiment_directory)
    fixed_no_pca = _fixed_policy_rows(
        benchmark_directory / "03_fixed_deterministic",
        benchmark_key,
        pca_components=None,
        policy_prefix="fixed_np",
        policy_type="fixed depth (no PCA)",
    )
    fixed_pca = _fixed_policy_rows(
        benchmark_directory / "04_fixed_deterministic_pca4",
        benchmark_key,
        pca_components=4,
        policy_prefix="fixed_p4",
        policy_type="fixed depth (PCA=4)",
    )
    policy_rows = _add_derived_columns(
        pd.concat(
            [schedules, fixed_no_pca, fixed_pca],
            ignore_index=True,
            sort=False,
        )
    )
    if policy_rows["policy_id"].nunique() != 60 or len(policy_rows) != 180:
        raise RuntimeError("Expected 60 policies and 180 policy/seed rows.")
    seed_rankings = _seed_rankings(policy_rows)
    aggregate = _aggregate_rankings(seed_rankings)
    phase_effects = _phase_effects(schedules)
    baseline = _baseline_comparison(policy_rows)
    return ScheduleNotebookData(
        experiment_directory=experiment_directory,
        benchmark_key=benchmark_key,
        schedule_rows=schedules,
        policy_rows=policy_rows,
        seed_rankings=seed_rankings,
        aggregate_rankings=aggregate,
        phase_effects=phase_effects,
        baseline_comparison=baseline,
    )


def top_five_by_seed(data: ScheduleNotebookData) -> pd.DataFrame:
    columns = [
        "smac_seed",
        "seed_rank",
        "policy_id",
        "policy_label",
        "policy_type",
        "par10_5000",
        "par10_500",
        "improvement_500_5000",
    ]
    return (
        data.seed_rankings.sort_values(
            ["smac_seed", "par10_5000", "policy_id"]
        )
        .groupby("smac_seed", as_index=False, group_keys=False)
        .head(5)[columns]
        .reset_index(drop=True)
    )


def top_five_overall(data: ScheduleNotebookData) -> pd.DataFrame:
    columns = [
        "overall_rank",
        "policy_id",
        "policy_label",
        "policy_type",
        "final_mean",
        "final_std",
        "final_worst",
        "mean_seed_rank",
        "worst_seed_rank",
        "improvement_500_5000_mean",
    ]
    return data.aggregate_rankings.head(5)[columns].copy()


def top_policy_definitions(
    data: ScheduleNotebookData,
    top_n: int = 10,
) -> pd.DataFrame:
    selected = data.aggregate_rankings.head(top_n)[
        ["overall_rank", "policy_id", "final_mean", "final_std"]
    ]
    definitions = (
        data.policy_rows.sort_values("smac_seed")
        .drop_duplicates("policy_id")
        [
            [
                "policy_id",
                "policy_type",
                "phase_0_depth",
                "phase_0_min_samples_split",
                "phase_0_feature_ratio",
                "phase_1_depth",
                "phase_1_min_samples_split",
                "phase_1_feature_ratio",
                "phase_2_depth",
                "phase_2_min_samples_split",
                "phase_2_feature_ratio",
            ]
        ]
    )
    return selected.merge(
        definitions,
        on="policy_id",
        validate="one_to_one",
    )


def plot_top_five_by_seed(data: ScheduleNotebookData) -> Any:
    top = top_five_by_seed(data)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=False)
    colors = {
        "three-phase schedule (PCA=4)": "#4C78A8",
        "fixed depth (PCA=4)": "#F58518",
        "fixed depth (no PCA)": "#54A24B",
    }
    for seed, axis in zip(SEEDS, axes):
        selected = top[top["smac_seed"] == seed].sort_values(
            "par10_5000",
            ascending=False,
        )
        axis.barh(
            selected["policy_label"],
            selected["par10_5000"],
            color=[colors[item] for item in selected["policy_type"]],
        )
        axis.set_title(f"SMAC seed {seed}")
        axis.set_xlabel("Final full-training PAR10")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Top five policies within each SMAC seed (lower is better)")
    figure.tight_layout()
    return figure


def plot_overall_ranking(
    data: ScheduleNotebookData,
    top_n: int = 15,
) -> Any:
    selected = data.aggregate_rankings.head(top_n).sort_values(
        "final_mean",
        ascending=False,
    )
    figure, axis = plt.subplots(figsize=(11, 7))
    axis.errorbar(
        selected["final_mean"],
        selected["policy_label"],
        xerr=selected["final_std"].fillna(0),
        fmt="o",
        color="#4C78A8",
        capsize=3,
        label="Mean ± one seed standard deviation",
    )
    for seed, marker in zip(SEEDS, ("x", "+", "1")):
        values = data.policy_rows[
            (data.policy_rows["policy_id"].isin(selected["policy_id"]))
            & (data.policy_rows["smac_seed"] == seed)
        ]
        lookup = values.set_index("policy_id")["par10_5000"]
        axis.scatter(
            [lookup.loc[item] for item in selected["policy_id"]],
            selected["policy_label"],
            marker=marker,
            alpha=0.75,
            label=f"Seed {seed}",
        )
    axis.set_xlabel("Final full-training PAR10")
    axis.set_title("Best policies across all three SMAC seeds")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def plot_checkpoint_trajectories(
    data: ScheduleNotebookData,
    top_n: int = 10,
) -> Any:
    selected = data.aggregate_rankings.head(top_n)["policy_id"].tolist()
    figure, axis = plt.subplots(figsize=(11, 7))
    for policy in selected:
        rows = data.policy_rows[data.policy_rows["policy_id"] == policy]
        means = np.asarray(
            [
                rows[f"par10_{checkpoint}"].mean()
                for checkpoint in CHECKPOINTS
            ]
        )
        lows = np.asarray(
            [
                rows[f"par10_{checkpoint}"].min()
                for checkpoint in CHECKPOINTS
            ]
        )
        highs = np.asarray(
            [
                rows[f"par10_{checkpoint}"].max()
                for checkpoint in CHECKPOINTS
            ]
        )
        label = str(rows["policy_label"].iloc[0])
        axis.plot(CHECKPOINTS, means, marker="o", label=label)
        axis.fill_between(CHECKPOINTS, lows, highs, alpha=0.08)
    axis.set_xlabel("SMAC trial")
    axis.set_ylabel("Full-training incumbent PAR10")
    axis.set_title(
        "Checkpoint trajectories of top policies: mean and seed range"
    )
    axis.set_xticks(CHECKPOINTS)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    return figure


def plot_rank_stability(
    data: ScheduleNotebookData,
    top_n: int = 20,
) -> Any:
    selected = data.aggregate_rankings.head(top_n)["policy_id"].tolist()
    labels = (
        data.aggregate_rankings.set_index("policy_id")
        .loc[selected, "policy_label"]
        .tolist()
    )
    matrix = (
        data.seed_rankings[
            data.seed_rankings["policy_id"].isin(selected)
        ]
        .pivot(index="policy_id", columns="smac_seed", values="seed_rank")
        .loc[selected, list(SEEDS)]
        .to_numpy(dtype=float)
    )
    figure, axis = plt.subplots(figsize=(7, 9))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if matrix[row, column] > 30 else "black",
            )
    axis.set_xticks(range(len(SEEDS)), [f"Seed {seed}" for seed in SEEDS])
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Seed-specific rank of the overall top policies")
    figure.colorbar(image, ax=axis, label="Rank (lower is better)")
    figure.tight_layout()
    return figure


def plot_early_vs_final(data: ScheduleNotebookData) -> Any:
    aggregate = data.aggregate_rankings
    figure, axis = plt.subplots(figsize=(9, 7))
    styles = {
        "three-phase schedule (PCA=4)": ("#4C78A8", "o"),
        "fixed depth (PCA=4)": ("#F58518", "s"),
        "fixed depth (no PCA)": ("#54A24B", "^"),
    }
    for policy_type, group in aggregate.groupby("policy_type"):
        color, marker = styles[policy_type]
        axis.scatter(
            group["par10_500_mean"],
            group["final_mean"],
            color=color,
            marker=marker,
            alpha=0.75,
            label=policy_type,
        )
    annotate = aggregate.head(8)
    for row in annotate.itertuples():
        axis.annotate(
            row.policy_label,
            (row.par10_500_mean, row.final_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean incumbent PAR10 at trial 500")
    axis.set_ylabel("Mean final incumbent PAR10 at trial 5000")
    axis.set_title("Early performance versus final performance")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def plot_phase_effects(data: ScheduleNotebookData) -> Any:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    for phase in range(3):
        for column, factor in enumerate(PHASE_FACTORS):
            axis = axes[phase, column]
            selected = data.phase_effects[
                (data.phase_effects["phase"] == phase)
                & (data.phase_effects["factor"] == factor)
            ].sort_values("value")
            axis.bar(
                selected["value"].astype(str),
                selected["mean"],
                yerr=selected["std"],
                capsize=2,
                color="#72B7B2",
            )
            metric = str(selected["metric"].iloc[0])
            direction = str(selected["direction"].iloc[0])
            axis.set_title(f"Phase {phase}: {factor}")
            axis.set_ylabel(f"{metric} ({direction})")
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Marginal association of each phase setting with its phase outcome"
    )
    figure.tight_layout()
    return figure


def plot_runtime_performance(data: ScheduleNotebookData) -> Any:
    aggregate = data.aggregate_rankings
    figure, axis = plt.subplots(figsize=(9, 7))
    for policy_type, group in aggregate.groupby("policy_type"):
        axis.scatter(
            group["walltime_hours_mean"],
            group["final_mean"],
            alpha=0.75,
            label=policy_type,
        )
    for row in aggregate.head(8).itertuples():
        axis.annotate(
            row.policy_label,
            (row.walltime_hours_mean, row.final_mean),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean wall time per 5,000-trial run (hours)")
    axis.set_ylabel("Mean final full-training PAR10")
    axis.set_title("Runtime–performance trade-off")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    return figure


def findings_markdown(data: ScheduleNotebookData) -> str:
    winner = data.aggregate_rankings.iloc[0]
    stable = data.aggregate_rankings.sort_values(
        ["worst_seed_rank", "mean_seed_rank", "final_mean"]
    ).iloc[0]
    comparison = data.baseline_comparison
    wins_all = int((comparison["wins_vs_best_fixed_pca4"] == 3).sum())
    wins_majority = int((comparison["wins_vs_best_fixed_pca4"] >= 2).sum())
    schedules = data.aggregate_rankings[
        data.aggregate_rankings["policy_type"]
        == "three-phase schedule (PCA=4)"
    ]
    correlation = schedules[
        ["par10_500_mean", "final_mean"]
    ].corr(method="spearman").iloc[0, 1]
    runtime_correlation = schedules[
        ["walltime_hours_mean", "final_mean"]
    ].corr(method="spearman").iloc[0, 1]
    return "\n".join(
        [
            f"- **Best mean final policy:** {winner.policy_label} "
            f"with PAR10 {winner.final_mean:.4g} ± {winner.final_std:.4g}.",
            f"- **Best worst-seed robustness:** {stable.policy_label}; "
            f"its worst seed rank is {int(stable.worst_seed_rank)}.",
            f"- **Schedules versus the best fixed PCA=4 depth:** "
            f"{wins_all} schedules win on all three seeds and "
            f"{wins_majority} win on at least two seeds.",
            f"- **Early/final Spearman correlation among schedules:** "
            f"{correlation:.3f}. Values near one mean early ranking is "
            "strongly predictive; values near zero favor adaptive decisions.",
            f"- **Runtime/final Spearman correlation among schedules:** "
            f"{runtime_correlation:.3f}. A negative value means slower "
            "schedules tended to finish better.",
            "- The no-PCA fixed controls from `03_fixed_deterministic` are "
            "included as requested, but schedule-versus-fixed conclusions "
            "should primarily use the PCA=4 fixed controls because all "
            "three-phase schedules use PCA=4.",
        ]
    )

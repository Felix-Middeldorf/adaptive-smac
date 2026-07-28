from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "smac_output"
PLOT_DIR = HERE / "plots"
BENCHMARK_SEEDS = (40, 42)
SMAC_SEEDS = tuple(range(10))
DEPTHS = (5, 10, 15, 20, 25, 30)
N_TRIALS = 3_500
DETAIL_SMAC_SEED = 0
PROPOSAL_BIN_WIDTH = 100
COLORS = dict(zip(DEPTHS, sns.color_palette("colorblind", n_colors=len(DEPTHS))))

RUN_COLUMNS = (
    "benchmark_seed", "smac_seed", "depth", "best_regret",
    "final_best_regret", "acquisition_proposals", "matched_proposals", "path",
)
PROPOSAL_COLUMNS = (
    "benchmark_seed", "smac_seed", "depth", "proposal_index",
    "completed_trials_before_proposal", "observed_trial", "config_id",
    "observed_instance", "observed_cost", "predicted_mean",
    "predicted_variance", "predicted_std", "ei", "absolute_error",
    "standardized_error",
)


def ordered_runhistory_rows(runhistory_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(
        runhistory_data["data"],
        key=lambda row: (row["starttime"], row["endtime"]),
    )
    return [
        {
            "trial": trial,
            "config_id": int(row["config_id"]),
            "instance": row["instance"],
            "cost": float(row["cost"]),
            "starttime": float(row["starttime"]),
            "endtime": float(row["endtime"]),
        }
        for trial, row in enumerate(rows, start=1)
    ]


def first_online_observation(
    proposal: dict[str, Any],
    trial_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    config_id = proposal.get("runhistory_config_id")
    completed_before = int(proposal["completed_trials_before_proposal"])
    for row in trial_rows:
        if row["config_id"] == config_id and row["trial"] > completed_before:
            return row
    return None


def load_experiment(
    *,
    output_dir: Path = OUTPUT_DIR,
    expected_trials: int = N_TRIALS,
    require_complete: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_records: list[dict[str, Any]] = []
    proposal_records: list[dict[str, Any]] = []
    problems: list[tuple[Any, ...]] = []

    for benchmark_seed in BENCHMARK_SEEDS:
        for smac_seed in SMAC_SEEDS:
            for depth in DEPTHS:
                directory = (
                    output_dir / f"benchmark_seed_{benchmark_seed}"
                    / f"fixed_depth_{depth}" / str(smac_seed)
                )
                trajectory_path = directory / "trajectory.json"
                diagnostics_path = directory / "proposal_diagnostics.json"
                runhistory_path = directory / "runhistory.json"
                required = (trajectory_path, diagnostics_path, runhistory_path)
                if not all(path.exists() for path in required):
                    problems.append(
                        (benchmark_seed, smac_seed, depth, "missing files", str(directory))
                    )
                    continue

                try:
                    trajectory = json.loads(trajectory_path.read_text())
                    diagnostics = json.loads(diagnostics_path.read_text())
                    runhistory = json.loads(runhistory_path.read_text())
                    best_regret = np.asarray(trajectory["best_regret"], dtype=float)
                    trial_rows = ordered_runhistory_rows(runhistory)
                except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    problems.append(
                        (benchmark_seed, smac_seed, depth, str(error), str(directory))
                    )
                    continue

                checks = {
                    "benchmark seed": trajectory.get("benchmark_seed") == benchmark_seed,
                    "SMAC seed": trajectory.get("smac_seed") == smac_seed,
                    "depth": trajectory.get("max_depth") == depth,
                    "reported trials": trajectory.get("n_trials") == expected_trials,
                    "trajectory length": len(best_regret) == expected_trials,
                    "runhistory length": len(trial_rows) == expected_trials,
                    "finite trajectory": bool(np.isfinite(best_regret).all()),
                    "monotone trajectory": bool(np.all(np.diff(best_regret) <= 1e-10)),
                    "diagnostic seed": (
                        diagnostics.get("benchmark_seed") == benchmark_seed
                        and diagnostics.get("smac_seed") == smac_seed
                    ),
                    "diagnostic depth": diagnostics.get("max_depth") == depth,
                }
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    problems.append(
                        (benchmark_seed, smac_seed, depth, ", ".join(failed), str(directory))
                    )
                    continue

                selected = [
                    proposal for proposal in diagnostics["proposals"]
                    if proposal.get("selected_by_acquisition")
                ]
                matched = 0
                for proposal in selected:
                    observation = first_online_observation(proposal, trial_rows)
                    if observation is None:
                        continue
                    predicted_mean = float(
                        proposal["predicted_cost_mean_marginalized"]
                    )
                    predicted_variance = float(
                        proposal["predicted_variance_marginalized"]
                    )
                    predicted_std = float(
                        proposal["predicted_std_marginalized"]
                    )
                    observed_cost = observation["cost"]
                    absolute_error = abs(observed_cost - predicted_mean)
                    standardized_error = absolute_error / max(predicted_std, 1e-8)
                    proposal_records.append(
                        {
                            "benchmark_seed": benchmark_seed,
                            "smac_seed": smac_seed,
                            "depth": depth,
                            "proposal_index": int(proposal["proposal_index"]),
                            "completed_trials_before_proposal": int(
                                proposal["completed_trials_before_proposal"]
                            ),
                            "observed_trial": int(observation["trial"]),
                            "config_id": int(proposal["runhistory_config_id"]),
                            "observed_instance": observation["instance"],
                            "observed_cost": observed_cost,
                            "predicted_mean": predicted_mean,
                            "predicted_variance": predicted_variance,
                            "predicted_std": predicted_std,
                            "ei": float(proposal["acquisition_value"]),
                            "absolute_error": absolute_error,
                            "standardized_error": standardized_error,
                        }
                    )
                    matched += 1

                if matched != len(selected):
                    problems.append(
                        (
                            benchmark_seed,
                            smac_seed,
                            depth,
                            f"{len(selected) - matched} unmatched acquisition proposals",
                            str(directory),
                        )
                    )
                run_records.append(
                    {
                        "benchmark_seed": benchmark_seed,
                        "smac_seed": smac_seed,
                        "depth": depth,
                        "best_regret": best_regret,
                        "final_best_regret": float(best_regret[-1]),
                        "acquisition_proposals": len(selected),
                        "matched_proposals": matched,
                        "path": trajectory_path,
                    }
                )

    runs = pd.DataFrame(run_records, columns=RUN_COLUMNS)
    proposals = pd.DataFrame(proposal_records, columns=PROPOSAL_COLUMNS)
    problem_table = pd.DataFrame(
        problems,
        columns=("benchmark_seed", "smac_seed", "depth", "problem", "directory"),
    )
    expected_runs = len(BENCHMARK_SEEDS) * len(SMAC_SEEDS) * len(DEPTHS)
    if require_complete:
        if len(runs) != expected_runs or not problem_table.empty:
            raise RuntimeError(
                f"Loaded {len(runs)} / {expected_runs} runs with "
                f"{len(problem_table)} reported problems."
            )
    return runs, proposals, problem_table


def coverage_table(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    return (
        runs.pivot_table(
            index=["benchmark_seed", "smac_seed"],
            columns="depth",
            values="matched_proposals",
            aggfunc="first",
        )
        .reindex(columns=DEPTHS)
        .rename_axis(columns="fixed_depth")
    )


def _plot_raw_metric(
    axis,
    values: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    *,
    linthresh: float,
) -> None:
    for depth in DEPTHS:
        depth_values = values.loc[values["depth"].eq(depth)].sort_values(
            "observed_trial"
        )
        if depth_values.empty:
            continue
        axis.plot(
            depth_values["observed_trial"],
            depth_values[column],
            color=COLORS[depth],
            linewidth=0.75,
            marker="o",
            markersize=2.0,
            alpha=0.70,
            label=f"Depth {depth}",
        )
    axis.set_yscale("symlog", linthresh=linthresh)
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def plot_detailed_seed(
    runs: pd.DataFrame,
    proposals: pd.DataFrame,
    benchmark_seed: int,
    *,
    smac_seed: int = DETAIL_SMAC_SEED,
    plot_dir: Path = PLOT_DIR,
) -> Path | None:
    selected_runs = runs.query(
        "benchmark_seed == @benchmark_seed and smac_seed == @smac_seed"
    )
    selected_proposals = proposals.query(
        "benchmark_seed == @benchmark_seed and smac_seed == @smac_seed"
    )
    if selected_runs.empty:
        return None

    fig, axes = plt.subplots(5, 1, figsize=(18, 25), sharex=True)
    for depth in DEPTHS:
        row = selected_runs.loc[selected_runs["depth"].eq(depth)]
        if row.empty:
            continue
        trajectory = row.iloc[0]["best_regret"]
        axes[0].plot(
            np.arange(1, len(trajectory) + 1),
            trajectory,
            color=COLORS[depth],
            linewidth=1.5,
            drawstyle="steps-post",
            label=f"Depth {depth}",
        )
    axes[0].set_yscale("symlog", linthresh=1e-8)
    axes[0].set_ylabel("Best-so-far regret")
    axes[0].set_title("Best-so-far regret")
    _plot_raw_metric(
        axes[1], selected_proposals, "ei", "Expected Improvement",
        "EI of acquisition-selected configuration", linthresh=1e-8,
    )
    _plot_raw_metric(
        axes[2], selected_proposals, "predicted_std", "Predicted std",
        "Surrogate uncertainty", linthresh=1e-8,
    )
    _plot_raw_metric(
        axes[3], selected_proposals, "absolute_error", "Absolute error",
        "Online absolute prediction error", linthresh=1e-6,
    )
    _plot_raw_metric(
        axes[4], selected_proposals, "standardized_error",
        "Absolute error / predicted std", "Online standardized prediction error",
        linthresh=1e-3,
    )
    for axis in axes:
        axis.set_xlim(1, N_TRIALS)
    axes[-1].set_xlabel("Completed SMAC trial")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(
        f"Benchmark seed {benchmark_seed}, SMAC seed {smac_seed}: raw diagnostics",
        y=1.002,
        fontsize=18,
    )
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"benchmark_seed_{benchmark_seed}_smac_seed_{smac_seed}_raw.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    return path


def _trailing_proposal_trial_average(
    values: pd.DataFrame,
    metric: str,
    window_trials: int,
) -> pd.Series:
    """Average proposals in [trial-window+1, trial] on a complete trial grid."""
    per_trial = values.groupby("observed_trial")[metric].mean()
    trial_grid = per_trial.reindex(pd.RangeIndex(1, N_TRIALS + 1))
    return trial_grid.rolling(window=window_trials, min_periods=1).mean()


def plot_detailed_seed_running_average(
    runs: pd.DataFrame,
    proposals: pd.DataFrame,
    benchmark_seed: int,
    *,
    smac_seed: int = DETAIL_SMAC_SEED,
    window_trials: int,
    plot_dir: Path = PLOT_DIR,
) -> Path | None:
    """Plot five trailing averages using a window measured in SMAC trials."""
    if window_trials < 1:
        raise ValueError("window_trials must be at least one.")
    selected_runs = runs.query(
        "benchmark_seed == @benchmark_seed and smac_seed == @smac_seed"
    )
    selected_proposals = proposals.query(
        "benchmark_seed == @benchmark_seed and smac_seed == @smac_seed"
    )
    if selected_runs.empty:
        return None

    metric_specs = (
        ("ei", "Expected Improvement", "Trailing-average EI", 1e-8),
        (
            "predicted_std", "Predicted std",
            "Trailing-average surrogate uncertainty", 1e-8,
        ),
        (
            "absolute_error", "Absolute error",
            "Trailing-average online absolute prediction error", 1e-6,
        ),
        (
            "standardized_error", "Absolute error / predicted std",
            "Trailing-average online standardized prediction error", 1e-3,
        ),
    )
    trial_grid = np.arange(1, N_TRIALS + 1)
    fig, axes = plt.subplots(5, 1, figsize=(18, 25), sharex=True)

    for depth in DEPTHS:
        run_row = selected_runs.loc[selected_runs["depth"].eq(depth)]
        if run_row.empty:
            continue
        trajectory = pd.Series(
            run_row.iloc[0]["best_regret"],
            index=pd.RangeIndex(1, N_TRIALS + 1),
            dtype=float,
        )
        rolling_regret = trajectory.rolling(
            window=window_trials,
            min_periods=1,
        ).mean()
        axes[0].plot(
            trial_grid,
            rolling_regret,
            color=COLORS[depth],
            linewidth=1.5,
            label=f"Depth {depth}",
        )

        depth_proposals = selected_proposals.loc[
            selected_proposals["depth"].eq(depth)
        ]
        if depth_proposals.empty:
            continue
        for axis, (metric, _, _, _) in zip(axes[1:], metric_specs):
            rolling_metric = _trailing_proposal_trial_average(
                depth_proposals,
                metric,
                window_trials,
            )
            axis.plot(
                trial_grid,
                rolling_metric.to_numpy(),
                color=COLORS[depth],
                linewidth=1.2,
                alpha=0.85,
                label=f"Depth {depth}",
            )

    axes[0].set_yscale("symlog", linthresh=1e-8)
    axes[0].set_ylabel("Best-so-far regret")
    axes[0].set_title("Trailing-average best-so-far regret")
    for axis, (_, ylabel, title, linthresh) in zip(axes[1:], metric_specs):
        axis.set_yscale("symlog", linthresh=linthresh)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    for axis in axes:
        axis.set_xlim(1, N_TRIALS)
    axes[-1].set_xlabel("Completed SMAC trial")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    fig.suptitle(
        f"Benchmark seed {benchmark_seed}, SMAC seed {smac_seed}: "
        f"trailing {window_trials}-trial averages",
        y=1.002,
        fontsize=18,
    )
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = (
        plot_dir
        / f"benchmark_seed_{benchmark_seed}_smac_seed_{smac_seed}_"
        f"rolling_{window_trials}_trials.png"
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    return path


def trajectory_confidence_intervals(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    rows = []
    for run in runs.itertuples(index=False):
        for trial, regret in enumerate(run.best_regret, start=1):
            rows.append(
                (run.benchmark_seed, run.smac_seed, run.depth, trial, float(regret))
            )
    long = pd.DataFrame(
        rows,
        columns=("benchmark_seed", "smac_seed", "depth", "trial", "best_regret"),
    )
    summary = (
        long.groupby(["benchmark_seed", "depth", "trial"], as_index=False)
        .agg(
            mean=("best_regret", "mean"),
            std=("best_regret", "std"),
            n=("best_regret", "size"),
        )
    )
    summary["ci95"] = (
        t.ppf(0.975, summary["n"] - 1)
        * summary["std"].fillna(0.0)
        / np.sqrt(summary["n"])
    )
    return summary


def proposal_confidence_intervals(
    proposals: pd.DataFrame,
    metric: str,
    *,
    bin_width: int = PROPOSAL_BIN_WIDTH,
) -> pd.DataFrame:
    if proposals.empty:
        return pd.DataFrame()
    values = proposals.copy()
    values["bin_end"] = np.minimum(
        ((values["observed_trial"] - 1) // bin_width + 1) * bin_width,
        N_TRIALS,
    ).astype(int)
    per_seed = (
        values.groupby(
            ["benchmark_seed", "smac_seed", "depth", "bin_end"],
            as_index=False,
        )[metric]
        .mean()
    )
    summary = (
        per_seed.groupby(["benchmark_seed", "depth", "bin_end"], as_index=False)
        .agg(mean=(metric, "mean"), std=(metric, "std"), n=(metric, "size"))
    )
    summary["ci95"] = (
        t.ppf(0.975, summary["n"] - 1)
        * summary["std"].fillna(0.0)
        / np.sqrt(summary["n"])
    )
    return summary


def _plot_ci_metric(
    axis,
    summary: pd.DataFrame,
    benchmark_seed: int,
    *,
    x_column: str,
    ylabel: str,
    title: str,
    linthresh: float,
) -> None:
    benchmark = summary.loc[summary["benchmark_seed"].eq(benchmark_seed)]
    for depth in DEPTHS:
        values = benchmark.loc[benchmark["depth"].eq(depth)].sort_values(x_column)
        if values.empty:
            continue
        x = values[x_column].to_numpy()
        mean = values["mean"].to_numpy()
        ci = values["ci95"].fillna(0.0).to_numpy()
        axis.plot(x, mean, color=COLORS[depth], linewidth=1.6, label=f"Depth {depth}")
        axis.fill_between(
            x,
            np.maximum(0.0, mean - ci),
            mean + ci,
            color=COLORS[depth],
            alpha=0.13,
            linewidth=0,
        )
    axis.set_yscale("symlog", linthresh=linthresh)
    axis.set_ylabel(ylabel)
    axis.set_title(title)


def plot_aggregate_confidence_intervals(
    runs: pd.DataFrame,
    proposals: pd.DataFrame,
    benchmark_seed: int,
    *,
    plot_dir: Path = PLOT_DIR,
) -> Path | None:
    trajectory_ci = trajectory_confidence_intervals(runs)
    if trajectory_ci.empty:
        return None
    metric_specs = (
        ("ei", "Expected Improvement", "Binned mean EI", 1e-8),
        ("predicted_std", "Predicted std", "Binned mean surrogate uncertainty", 1e-8),
        ("absolute_error", "Absolute error", "Binned mean online absolute error", 1e-6),
        (
            "standardized_error", "Absolute error / predicted std",
            "Binned mean online standardized error", 1e-3,
        ),
    )
    proposal_summaries = {
        metric: proposal_confidence_intervals(proposals, metric)
        for metric, _, _, _ in metric_specs
    }

    fig, axes = plt.subplots(5, 1, figsize=(18, 25), sharex=True)
    _plot_ci_metric(
        axes[0], trajectory_ci, benchmark_seed, x_column="trial",
        ylabel="Mean best regret", title="Mean best-so-far regret and 95% CI",
        linthresh=1e-8,
    )
    for axis, (metric, ylabel, title, linthresh) in zip(axes[1:], metric_specs):
        _plot_ci_metric(
            axis,
            proposal_summaries[metric],
            benchmark_seed,
            x_column="bin_end",
            ylabel=ylabel,
            title=f"{title} and 95% CI across SMAC seeds",
            linthresh=linthresh,
        )
    for axis in axes:
        axis.set_xlim(1, N_TRIALS)
    axes[-1].set_xlabel("Completed SMAC trial / non-overlapping bin end")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(
        f"Benchmark seed {benchmark_seed}: aggregate diagnostics over SMAC seeds",
        y=1.002,
        fontsize=18,
    )
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"benchmark_seed_{benchmark_seed}_aggregate_ci.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    return path


def seed_level_summary(runs: pd.DataFrame, proposals: pd.DataFrame) -> pd.DataFrame:
    if runs.empty or proposals.empty:
        return pd.DataFrame()
    proposal_summary = (
        proposals.groupby(["benchmark_seed", "smac_seed", "depth"], as_index=False)
        .agg(
            median_ei=("ei", "median"),
            median_predicted_std=("predicted_std", "median"),
            median_absolute_error=("absolute_error", "median"),
            median_standardized_error=("standardized_error", "median"),
        )
    )
    final_regret = runs[
        ["benchmark_seed", "smac_seed", "depth", "final_best_regret"]
    ]
    return final_regret.merge(
        proposal_summary,
        on=["benchmark_seed", "smac_seed", "depth"],
        how="inner",
        validate="one_to_one",
    )


def plot_seed_level_boxplots(
    summary: pd.DataFrame,
    benchmark_seed: int,
    *,
    plot_dir: Path = PLOT_DIR,
) -> Path | None:
    values = summary.loc[summary["benchmark_seed"].eq(benchmark_seed)]
    if values.empty:
        return None
    metrics = (
        ("final_best_regret", "Final best regret", "Final best-so-far regret"),
        ("median_ei", "Median EI", "Per-run median EI"),
        ("median_predicted_std", "Median predicted std", "Per-run median uncertainty"),
        ("median_absolute_error", "Median absolute error", "Per-run median absolute error"),
        (
            "median_standardized_error", "Median standardized error",
            "Per-run median standardized error",
        ),
    )
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        sns.boxplot(
            data=values,
            x="depth",
            y=metric,
            order=DEPTHS,
            hue="depth",
            palette=COLORS,
            dodge=False,
            legend=False,
            showfliers=False,
            ax=axis,
        )
        sns.stripplot(
            data=values,
            x="depth",
            y=metric,
            order=DEPTHS,
            color="black",
            alpha=0.55,
            size=3.2,
            jitter=0.13,
            ax=axis,
        )
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.set_xlabel("Fixed depth")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    fig.suptitle(
        f"Benchmark seed {benchmark_seed}: distributions across SMAC seeds",
        y=1.02,
        fontsize=17,
    )
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"benchmark_seed_{benchmark_seed}_seed_boxplots.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.show()
    return path

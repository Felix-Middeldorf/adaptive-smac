"""Analytics for interrupted ACLib adaptive-depth policy experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from surrogate_analytics import (
    DEPTHS,
    N_TRIALS,
    SMAC_SEEDS,
    FullTrainingValidator,
    canonical_configuration,
    configuration_fingerprint,
)


POLICIES = (
    "saturation_k50",
    "saturation_k100",
    "saturation_k250",
    "rotating_saturation_k50",
    "error_variance_trend_25",
    "oracle_incumbent_depth",
)
POLICY_LABELS = {
    "saturation_k50": "Saturation k=50",
    "saturation_k100": "Saturation k=100",
    "saturation_k250": "Saturation k=250",
    "rotating_saturation_k50": "Rotating saturation k=50",
    "error_variance_trend_25": "Error/variance trend",
    "oracle_incumbent_depth": "Oracle depth schedule",
}
FIXED_LABELS = {depth: f"Fixed depth {depth}" for depth in DEPTHS}
METHOD_ORDER = tuple(FIXED_LABELS.values()) + tuple(
    POLICY_LABELS[policy] for policy in POLICIES
)
POLICY_COLORS = {
    "Saturation k=50": "#4c78a8",
    "Saturation k=100": "#f58518",
    "Saturation k=250": "#54a24b",
    "Rotating saturation k=50": "#e45756",
    "Error/variance trend": "#b279a2",
    "Oracle depth schedule": "#ff9da6",
}
FIXED_COLORS = {
    f"Fixed depth {depth}": color
    for depth, color in zip(
        DEPTHS,
        ("#252525", "#525252", "#737373", "#969696", "#bdbdbd"),
    )
}


@dataclass(frozen=True)
class PolicyComparison:
    benchmark_key: str
    display_name: str
    policy_directory: Path
    initial_directory: Path
    trajectories: pd.DataFrame
    endpoints: pd.DataFrame
    availability: pd.DataFrame
    common_horizon: pd.DataFrame
    validation_cache_file: Path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _observed_telemetry_trials(directory: Path) -> int:
    path = directory / "configuration_telemetry.jsonl"
    if not path.is_file():
        return 0
    maximum = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = record.get("runhistory_finished")
            if isinstance(value, int):
                maximum = max(maximum, value)
    return maximum


def _step_value(events: pd.DataFrame, trial: int) -> float:
    preceding = events[events["trial"] <= trial].sort_values("trial")
    if preceding.empty:
        return np.nan
    return float(preceding.iloc[-1]["full_training_par10"])


def build_policy_comparison(
    policy_directory: Path,
    benchmark_key: str,
    initial_directory: Path,
) -> PolicyComparison:
    policy_directory = Path(policy_directory).resolve()
    initial_directory = Path(initial_directory).resolve()
    fixed_cache = initial_directory / "analytics_cache"
    fixed_events_file = fixed_cache / "incumbent_trajectory_events.csv"
    if not fixed_events_file.is_file():
        raise RuntimeError(
            f"Execute the fixed-depth analytics notebook first; missing "
            f"{fixed_events_file}."
        )

    fixed = pd.read_csv(fixed_events_file)
    expected_fixed = {
        (depth, seed) for depth in DEPTHS for seed in SMAC_SEEDS
    }
    found_fixed = set(
        zip(fixed["depth"].astype(int), fixed["smac_seed"].astype(int))
    )
    if found_fixed != expected_fixed:
        raise RuntimeError("The fixed-depth trajectory cache is incomplete.")
    fixed["method"] = fixed["depth"].map(FIXED_LABELS)
    fixed["method_key"] = fixed["depth"].map(
        lambda depth: f"fixed_depth_{int(depth)}"
    )
    fixed["family"] = "fixed depth"
    fixed["run_complete"] = True
    fixed["observed_endpoint_trial"] = N_TRIALS

    availability_rows: list[dict[str, Any]] = []
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    configurations: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        for seed in SMAC_SEEDS:
            directory = policy_directory / "results" / policy / str(seed)
            observed_trials = _observed_telemetry_trials(directory)
            try:
                runhistory = _read_json(directory / "runhistory.json")
                intensifier = _read_json(directory / "intensifier.json")
                finished_trials = len(runhistory.get("data", []))
                events = list(intensifier.get("trajectory", []))
                if finished_trials < 1 or not events:
                    raise RuntimeError("no persisted trials or trajectory")
                usable_events = [
                    event
                    for event in events
                    if int(event["trial"]) <= finished_trials
                ]
                if not usable_events:
                    raise RuntimeError("no trajectory within persisted trials")
                for event in usable_events:
                    if len(event["config_ids"]) != 1:
                        raise RuntimeError("multi-incumbent trajectory")
                    config_id = str(int(event["config_ids"][0]))
                    configuration = canonical_configuration(
                        runhistory["configs"][config_id]
                    )
                    configurations.setdefault(
                        configuration_fingerprint(configuration),
                        configuration,
                    )
                payloads[(policy, seed)] = {
                    "runhistory": runhistory,
                    "events": usable_events,
                    "finished_trials": finished_trials,
                }
                status = "recoverable partial run"
                reason = ""
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                finished_trials = observed_trials
                status = "trajectory unavailable"
                reason = (
                    "runhistory/intensifier corrupted during cluster SIGBUS: "
                    f"{type(error).__name__}"
                )
            availability_rows.append(
                {
                    "policy": POLICY_LABELS[policy],
                    "policy_key": policy,
                    "smac_seed": seed,
                    "trajectory_status": status,
                    "observed_trials": finished_trials,
                    "reason": reason,
                }
            )

    validator = FullTrainingValidator(
        benchmark_key,
        policy_directory,
        quantile_seed=0,
    )
    validation = validator.evaluate_many(configurations)

    policy_rows: list[dict[str, Any]] = []
    for (policy, seed), payload in payloads.items():
        runhistory = payload["runhistory"]
        finished_trials = int(payload["finished_trials"])
        rows_for_run: list[dict[str, Any]] = []
        for event in sorted(payload["events"], key=lambda item: int(item["trial"])):
            config_id = int(event["config_ids"][0])
            configuration = runhistory["configs"][str(config_id)]
            fingerprint = configuration_fingerprint(configuration)
            rows_for_run.append(
                {
                    "method": POLICY_LABELS[policy],
                    "method_key": policy,
                    "family": "adaptive policy",
                    "smac_seed": seed,
                    "trial": int(event["trial"]),
                    "config_id": config_id,
                    "configuration_fingerprint": fingerprint,
                    "full_training_par10": float(
                        validation[fingerprint]["mean_par10"]
                    ),
                    "smac_incumbent_subset_cost": float(event["costs"][0]),
                    "run_complete": False,
                    "observed_endpoint_trial": finished_trials,
                }
            )
        if rows_for_run[-1]["trial"] < finished_trials:
            rows_for_run.append(
                {
                    **rows_for_run[-1],
                    "trial": finished_trials,
                }
            )
        policy_rows.extend(rows_for_run)

    trajectory_columns = [
        "method",
        "method_key",
        "family",
        "smac_seed",
        "trial",
        "config_id",
        "configuration_fingerprint",
        "full_training_par10",
        "smac_incumbent_subset_cost",
        "run_complete",
        "observed_endpoint_trial",
    ]
    trajectories = pd.concat(
        [
            fixed[trajectory_columns],
            pd.DataFrame(policy_rows, columns=trajectory_columns),
        ],
        ignore_index=True,
    ).sort_values(["smac_seed", "method", "trial"])

    endpoints = (
        trajectories.sort_values("trial")
        .groupby(["method", "smac_seed"], as_index=False)
        .tail(1)
        .sort_values(["smac_seed", "method"])
        .reset_index(drop=True)
    )
    endpoints["method"] = pd.Categorical(
        endpoints["method"],
        categories=METHOD_ORDER,
        ordered=True,
    )
    endpoints = endpoints.sort_values(["smac_seed", "method"])

    common_rows: list[dict[str, Any]] = []
    for seed in SMAC_SEEDS:
        adaptive = endpoints[
            (endpoints["smac_seed"] == seed)
            & (endpoints["family"] == "adaptive policy")
        ]
        if adaptive.empty:
            continue
        horizon = int(adaptive["trial"].min())
        for method in METHOD_ORDER:
            selected = trajectories[
                (trajectories["smac_seed"] == seed)
                & (trajectories["method"] == method)
            ]
            if selected.empty:
                continue
            common_rows.append(
                {
                    "smac_seed": seed,
                    "common_trial": horizon,
                    "method": method,
                    "family": selected.iloc[0]["family"],
                    "full_training_par10": _step_value(selected, horizon),
                }
            )
    common_horizon = pd.DataFrame(common_rows)
    if not common_horizon.empty:
        common_horizon["method"] = pd.Categorical(
            common_horizon["method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
        common_horizon = common_horizon.sort_values(["smac_seed", "method"])

    spec_name = (
        "Clasp Queens"
        if benchmark_key == "clasp_queens"
        else "Clasp weighted-sequence"
    )
    comparison = PolicyComparison(
        benchmark_key=benchmark_key,
        display_name=spec_name,
        policy_directory=policy_directory,
        initial_directory=initial_directory,
        trajectories=trajectories.reset_index(drop=True),
        endpoints=endpoints.reset_index(drop=True),
        availability=pd.DataFrame(availability_rows),
        common_horizon=common_horizon.reset_index(drop=True),
        validation_cache_file=validator.cache_file,
    )
    write_policy_tables(comparison)
    return comparison


def write_policy_tables(comparison: PolicyComparison) -> tuple[Path, ...]:
    output = comparison.policy_directory / "analytics_cache"
    output.mkdir(parents=True, exist_ok=True)
    tables = (
        ("combined_incumbent_trajectory_events.csv", comparison.trajectories),
        ("observed_endpoint_incumbents.csv", comparison.endpoints),
        ("policy_trajectory_availability.csv", comparison.availability),
        ("common_horizon_incumbents.csv", comparison.common_horizon),
    )
    paths: list[Path] = []
    for filename, table in tables:
        path = output / filename
        table.to_csv(path, index=False)
        paths.append(path)
    return tuple(paths)


def plot_incumbent_trajectories(
    comparison: PolicyComparison,
    *,
    logarithmic: bool = False,
) -> list[Any]:
    figures = []
    for seed in SMAC_SEEDS:
        figure, axis = plt.subplots(figsize=(13, 7))
        selected = comparison.trajectories[
            comparison.trajectories["smac_seed"] == seed
        ]
        for method in METHOD_ORDER:
            events = selected[selected["method"] == method].sort_values("trial")
            if events.empty:
                continue
            fixed = events.iloc[0]["family"] == "fixed depth"
            axis.step(
                events["trial"],
                events["full_training_par10"],
                where="post",
                label=method,
                color=(
                    FIXED_COLORS[method]
                    if fixed
                    else POLICY_COLORS[method]
                ),
                linestyle="--" if fixed else "-",
                linewidth=1.3 if fixed else 2.0,
                alpha=0.8 if fixed else 0.95,
            )
            if not fixed:
                endpoint = events.iloc[-1]
                axis.scatter(
                    [endpoint["trial"]],
                    [endpoint["full_training_par10"]],
                    color=POLICY_COLORS[method],
                    marker="x",
                    s=45,
                    zorder=5,
                )
        scale = "log" if logarithmic else "linear"
        axis.set(
            xlabel="Completed SMAC trial",
            ylabel="Mean incumbent PAR10 on all training instances",
            title=(
                f"{comparison.display_name}: fixed-depth and adaptive-policy "
                f"incumbents, SMAC seed {seed} ({scale} scale)"
            ),
            xlim=(1, N_TRIALS),
            yscale=scale,
        )
        axis.grid(alpha=0.2)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        figures.append(figure)
    return figures


def _plot_method_bars(
    table: pd.DataFrame,
    comparison: PolicyComparison,
    *,
    title_suffix: str,
    trial_column: str | None,
) -> Any:
    figure, axes = plt.subplots(1, 3, figsize=(19, 8), sharey=True)
    for axis, seed in zip(axes, SMAC_SEEDS):
        selected = table[table["smac_seed"] == seed].copy()
        selected["method"] = pd.Categorical(
            selected["method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
        selected = selected.sort_values("method")
        colors = [
            FIXED_COLORS.get(str(method), POLICY_COLORS.get(str(method)))
            for method in selected["method"]
        ]
        labels = [str(method) for method in selected["method"]]
        y = np.arange(len(selected))
        axis.barh(y, selected["full_training_par10"], color=colors)
        axis.set_yticks(y, labels=labels)
        axis.invert_yaxis()
        subtitle = (
            f"trial {int(selected[trial_column].iloc[0]):,}"
            if trial_column is not None
            else "run-specific observed endpoints"
        )
        axis.set(
            xlabel="Mean incumbent PAR10 (lower is better)",
            title=f"SMAC seed {seed}\n{subtitle}",
        )
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle(
        f"{comparison.display_name}: {title_suffix}",
        y=1.01,
    )
    figure.tight_layout()
    return figure


def plot_observed_endpoint_incumbents(comparison: PolicyComparison) -> Any:
    return _plot_method_bars(
        comparison.endpoints,
        comparison,
        title_suffix="incumbent at each run's last persisted trial",
        trial_column=None,
    )


def plot_common_horizon_incumbents(comparison: PolicyComparison) -> Any:
    return _plot_method_bars(
        comparison.common_horizon,
        comparison,
        title_suffix="incumbent comparison at a common per-seed horizon",
        trial_column="common_trial",
    )

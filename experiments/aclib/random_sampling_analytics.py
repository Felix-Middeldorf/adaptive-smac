"""Reusable analytics for ACLib random-configuration samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplingResults:
    directory: Path
    metadata: dict[str, Any]
    configurations: pd.DataFrame
    costs: np.ndarray

    @property
    def identity(self) -> dict[str, Any]:
        return self.metadata["identity"]

    @property
    def quantile_seeds(self) -> tuple[int, ...]:
        return tuple(self.identity["quantile_seeds"])

    @property
    def instances(self) -> tuple[str, ...]:
        return tuple(self.metadata["training_instances"])

    @property
    def timeout_cost(self) -> float:
        return float(self.identity["timeout_cost"])

    @property
    def median_costs(self) -> np.ndarray:
        return np.asarray(self.costs[:, 0, :], dtype=float)

    @property
    def median_config_means(self) -> np.ndarray:
        return np.mean(self.median_costs, axis=1)

    @property
    def stochastic_config_means(self) -> np.ndarray:
        if len(self.quantile_seeds) < 2:
            return np.empty((self.costs.shape[0], 0), dtype=float)
        return np.mean(
            np.asarray(self.costs[:, 1:, :], dtype=float),
            axis=2,
        )


def load_sampling_results(directory: str | Path = ".") -> SamplingResults:
    directory = Path(directory).resolve()
    result_directory = directory / "results"
    metadata_path = result_directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} is missing. Run or submit the sampling job first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("state") != "complete":
        raise RuntimeError(
            "The sampling job is not complete; see results/progress.json."
        )
    configurations = pd.read_json(
        result_directory / "configurations.jsonl",
        lines=True,
    )
    costs = np.load(result_directory / "costs.npy", mmap_mode="r")
    expected_shape = (
        int(metadata["identity"]["n_configurations"]),
        len(metadata["identity"]["quantile_seeds"]),
        int(metadata["identity"]["n_training_instances"]),
    )
    if costs.shape != expected_shape:
        raise RuntimeError(f"Cost shape {costs.shape} != {expected_shape}.")
    if not np.isfinite(costs).all():
        raise RuntimeError("The stored cost tensor contains incomplete values.")
    return SamplingResults(directory, metadata, configurations, costs)


def overview_table(results: SamplingResults) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark": results.identity["display_name"],
                "configurations": results.costs.shape[0],
                "training_instances": results.costs.shape[2],
                "quantile_modes": results.costs.shape[1],
                "median_seed": results.quantile_seeds[0],
                "stochastic_draws": results.costs.shape[1] - 1,
                "target_predictions": int(np.prod(results.costs.shape)),
                "cutoff": results.identity["cutoff"],
                "PAR10_timeout_cost": results.timeout_cost,
                "configspace_seed": results.identity["configspace_seed"],
            }
        ]
    ).set_index("benchmark")


def configuration_distribution_table(
    results: SamplingResults,
) -> pd.DataFrame:
    values = results.median_config_means
    return pd.DataFrame(
        {
            "value": [
                np.min(values),
                np.quantile(values, 0.01),
                np.quantile(values, 0.10),
                np.quantile(values, 0.25),
                np.median(values),
                np.mean(values),
                np.quantile(values, 0.75),
                np.quantile(values, 0.90),
                np.quantile(values, 0.99),
                np.max(values),
                np.std(values),
            ]
        },
        index=[
            "minimum",
            "q01",
            "q10",
            "q25",
            "median",
            "mean",
            "q75",
            "q90",
            "q99",
            "maximum",
            "standard deviation",
        ],
    )


def plot_value_distribution(
    results: SamplingResults,
) -> tuple[plt.Figure, np.ndarray]:
    values = results.median_config_means
    ordered = np.sort(values)
    ecdf = np.arange(1, len(ordered) + 1) / len(ordered)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(values, bins=45, color="tab:blue", alpha=0.8)
    axes[0].set_title("Distribution of full-training mean PAR10")
    axes[0].set_xlabel("Mean PAR10 over training instances")
    axes[0].set_ylabel("Random configurations")
    axes[1].plot(ordered, ecdf, color="tab:orange")
    axes[1].set_title("Empirical CDF of configuration performance")
    axes[1].set_xlabel("Mean PAR10 over training instances")
    axes[1].set_ylabel("Fraction of configurations")
    if np.min(values) > 0:
        for axis in axes:
            axis.set_xscale("log")
    figure.suptitle(
        f"{results.identity['display_name']}: deterministic EPM median (seed 0)"
    )
    figure.tight_layout()
    return figure, axes


def ranked_configurations(
    results: SamplingResults,
    *,
    n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = results.median_costs
    table = pd.DataFrame(
        {
            "configuration_index": np.arange(values.shape[0]),
            "mean_PAR10": np.mean(values, axis=1),
            "median_instance_PAR10": np.median(values, axis=1),
            "instance_std": np.std(values, axis=1),
            "instance_IQR": np.quantile(values, 0.75, axis=1)
            - np.quantile(values, 0.25, axis=1),
            "timeout_fraction": np.mean(
                values == results.timeout_cost, axis=1
            ),
        }
    ).sort_values("mean_PAR10", ignore_index=True)
    return table.head(n), table.tail(n).sort_values(
        "mean_PAR10", ascending=False, ignore_index=True
    )


def within_configuration_variability_table(
    results: SamplingResults,
) -> pd.DataFrame:
    values = results.median_costs
    mean = np.mean(values, axis=1)
    std = np.std(values, axis=1)
    iqr = np.quantile(values, 0.75, axis=1) - np.quantile(
        values, 0.25, axis=1
    )
    metrics = {
        "instance standard deviation": std,
        "instance IQR": iqr,
        "coefficient of variation": std / np.maximum(mean, 1e-12),
        "timeout fraction": np.mean(
            values == results.timeout_cost, axis=1
        ),
    }
    return pd.DataFrame(
        {
            name: [
                np.mean(metric),
                np.median(metric),
                np.quantile(metric, 0.90),
                np.max(metric),
            ]
            for name, metric in metrics.items()
        },
        index=["mean across configs", "median", "q90", "maximum"],
    ).T


def plot_instance_variability(
    results: SamplingResults,
) -> tuple[plt.Figure, np.ndarray]:
    values = results.median_costs
    means = np.mean(values, axis=1)
    standard_deviations = np.std(values, axis=1)
    timeout_fractions = np.mean(values == results.timeout_cost, axis=1)
    coefficients = standard_deviations / np.maximum(means, 1e-12)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    points = axes[0].scatter(
        means,
        standard_deviations,
        c=timeout_fractions,
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    axes[0].set_title("Mean performance versus instance variability")
    axes[0].set_xlabel("Mean PAR10")
    axes[0].set_ylabel("PAR10 standard deviation across instances")
    if np.min(means) > 0:
        axes[0].set_xscale("log")
    if np.min(standard_deviations) > 0:
        axes[0].set_yscale("log")
    figure.colorbar(points, ax=axes[0], label="Timeout fraction")
    axes[1].hist(coefficients, bins=45, color="tab:green", alpha=0.8)
    axes[1].set_title("Relative variability across instances")
    axes[1].set_xlabel("Instance coefficient of variation")
    axes[1].set_ylabel("Random configurations")
    figure.suptitle(
        f"{results.identity['display_name']}: within-configuration variability"
    )
    figure.tight_layout()
    return figure, axes


def stochastic_variability_table(results: SamplingResults) -> pd.DataFrame:
    stochastic = results.stochastic_config_means
    if stochastic.shape[1] == 0:
        return pd.DataFrame()
    per_config_std = np.std(stochastic, axis=1)
    per_config_mean = np.mean(stochastic, axis=1)
    relative = per_config_std / np.maximum(per_config_mean, 1e-12)
    return pd.DataFrame(
        {
            "full-training mean across stochastic draws": [
                np.mean(per_config_mean),
                np.median(per_config_mean),
                np.quantile(per_config_mean, 0.90),
                np.max(per_config_mean),
            ],
            "per-config stochastic standard deviation": [
                np.mean(per_config_std),
                np.median(per_config_std),
                np.quantile(per_config_std, 0.90),
                np.max(per_config_std),
            ],
            "per-config stochastic coefficient of variation": [
                np.mean(relative),
                np.median(relative),
                np.quantile(relative, 0.90),
                np.max(relative),
            ],
        },
        index=["mean across configs", "median", "q90", "maximum"],
    ).T


def quantile_seed_table(results: SamplingResults) -> pd.DataFrame:
    config_means = np.mean(
        np.asarray(results.costs, dtype=float),
        axis=2,
    )
    rows = []
    for seed_index, seed in enumerate(results.quantile_seeds):
        values = config_means[:, seed_index]
        rows.append(
            {
                "quantile_seed": seed,
                "kind": "median" if seed == 0 else "stochastic draw",
                "mean_across_configurations": np.mean(values),
                "std_across_configurations": np.std(values),
                "q10": np.quantile(values, 0.10),
                "median": np.median(values),
                "q90": np.quantile(values, 0.90),
            }
        )
    return pd.DataFrame(rows).set_index("quantile_seed")


def plot_stochastic_variability(
    results: SamplingResults,
) -> tuple[plt.Figure, np.ndarray]:
    stochastic = results.stochastic_config_means
    if stochastic.shape[1] == 0:
        raise ValueError("No stochastic quantile draws are available.")
    median = results.median_config_means
    stochastic_mean = np.mean(stochastic, axis=1)
    stochastic_std = np.std(stochastic, axis=1)
    relative = stochastic_std / np.maximum(stochastic_mean, 1e-12)
    rank_correlation = pd.Series(median).rank().corr(
        pd.Series(stochastic_mean).rank()
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(median, stochastic_mean, s=18, alpha=0.65)
    lower = min(np.min(median), np.min(stochastic_mean))
    upper = max(np.max(median), np.max(stochastic_mean))
    axes[0].plot([lower, upper], [lower, upper], "--", color="black", lw=1)
    axes[0].set_title(
        f"Median surface vs stochastic mean\nrank correlation={rank_correlation:.3f}"
    )
    axes[0].set_xlabel("Quantile seed 0: full-training mean PAR10")
    axes[0].set_ylabel("Mean over stochastic quantile draws")
    if lower > 0:
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
    axes[1].hist(relative, bins=45, color="tab:red", alpha=0.8)
    axes[1].set_title("EPM draw-to-draw variability by configuration")
    axes[1].set_xlabel("Stochastic coefficient of variation")
    axes[1].set_ylabel("Random configurations")
    figure.suptitle(
        f"{results.identity['display_name']}: target-surrogate stochasticity"
    )
    figure.tight_layout()
    return figure, axes


def instance_table(results: SamplingResults, *, n: int = 15) -> pd.DataFrame:
    values = results.median_costs
    table = pd.DataFrame(
        {
            "instance": [
                Path(instance).name for instance in results.instances
            ],
            "mean_across_configurations": np.mean(values, axis=0),
            "median_across_configurations": np.median(values, axis=0),
            "configuration_IQR": np.quantile(values, 0.75, axis=0)
            - np.quantile(values, 0.25, axis=0),
            "timeout_fraction": np.mean(
                values == results.timeout_cost, axis=0
            ),
        }
    )
    return table.sort_values(
        "mean_across_configurations", ascending=False, ignore_index=True
    ).head(n)


def plot_instance_effects(
    results: SamplingResults,
    *,
    n_hardest: int = 20,
) -> tuple[plt.Figure, np.ndarray]:
    values = results.median_costs
    instance_means = np.mean(values, axis=0)
    names = np.asarray([Path(item).name for item in results.instances])
    hardest = np.argsort(instance_means)[-n_hardest:]
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(instance_means, bins=45, color="tab:purple", alpha=0.8)
    axes[0].set_title("Difficulty distribution across training instances")
    axes[0].set_xlabel("Mean PAR10 across random configurations")
    axes[0].set_ylabel("Training instances")
    if np.min(instance_means) > 0:
        axes[0].set_xscale("log")
    order = hardest[np.argsort(instance_means[hardest])]
    axes[1].barh(
        np.arange(len(order)),
        instance_means[order],
        color="tab:brown",
    )
    axes[1].set_yticks(np.arange(len(order)), labels=names[order], fontsize=7)
    axes[1].set_title(f"{n_hardest} hardest instances in the sample")
    axes[1].set_xlabel("Mean PAR10 across random configurations")
    figure.suptitle(
        f"{results.identity['display_name']}: instance effects at quantile seed 0"
    )
    figure.tight_layout()
    return figure, axes


def variance_decomposition(results: SamplingResults) -> pd.DataFrame:
    median = results.median_costs
    stochastic = results.stochastic_config_means
    rows = [
        {
            "source": "between configurations",
            "variance": np.var(np.mean(median, axis=1)),
            "measurement": "Variance of full-training mean PAR10",
        },
        {
            "source": "within configuration, between instances",
            "variance": np.mean(np.var(median, axis=1)),
            "measurement": "Mean per-configuration instance variance",
        },
    ]
    if stochastic.shape[1]:
        rows.append(
            {
                "source": "within configuration, between EPM draws",
                "variance": np.mean(np.var(stochastic, axis=1)),
                "measurement": (
                    "Mean per-configuration variance of full-training means"
                ),
            }
        )
    return pd.DataFrame(rows).set_index("source")


def plot_sampling_convergence(
    results: SamplingResults,
) -> tuple[plt.Figure, plt.Axes]:
    values = results.median_config_means
    checkpoints = np.unique(
        np.geomspace(10, len(values), num=45).astype(int)
    )
    rows = []
    for count in checkpoints:
        prefix = values[:count]
        rows.append(
            (
                count,
                np.min(prefix),
                np.quantile(prefix, 0.10),
                np.median(prefix),
                np.quantile(prefix, 0.90),
            )
        )
    frame = pd.DataFrame(
        rows, columns=["samples", "best", "q10", "median", "q90"]
    )
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(frame["samples"], frame["best"], label="best", lw=2)
    axis.plot(frame["samples"], frame["median"], label="median", lw=2)
    axis.fill_between(
        frame["samples"],
        frame["q10"],
        frame["q90"],
        alpha=0.2,
        label="q10–q90",
    )
    axis.set_xscale("log")
    if np.min(values) > 0:
        axis.set_yscale("log")
    axis.set_xlabel("Number of sampled configurations")
    axis.set_ylabel("Full-training mean PAR10")
    axis.set_title(
        f"{results.identity['display_name']}: stability as the sample grows"
    )
    axis.legend()
    figure.tight_layout()
    return figure, axis


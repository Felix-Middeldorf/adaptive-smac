"""Small plotting helpers for minimal ACLib surrogate validations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_validation(directory: str | Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    results = Path(directory) / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    parity = pd.read_csv(results / "wrapper_parity.csv")
    agreement = pd.read_csv(results / "archive_agreement.csv")
    return summary, parity, agreement


def overview_table(summary: dict[str, Any]) -> pd.DataFrame:
    agreement = summary["archive_agreement"]
    return pd.DataFrame(
        {
            "value": {
                "wrapper points": summary["wrapper_parity"]["tested_points"],
                "wrapper points passed": summary["wrapper_parity"]["all_passed"],
                "archived comparison points": agreement["points"],
                "Spearman PAR10": agreement["spearman_par10"],
                "median absolute log10 error": agreement[
                    "median_absolute_log10_error"
                ],
                "within factor 2": agreement["fraction_within_factor_2"],
                "within factor 5": agreement["fraction_within_factor_5"],
                "actual timeout/failure fraction": agreement[
                    "actual_timeout_or_failure_fraction"
                ],
                "predicted timeout fraction": agreement[
                    "predicted_timeout_fraction"
                ],
            }
        }
    )


def plot_archive_agreement(
    summary: dict[str, Any],
    agreement: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    display_name = summary["benchmark"]["display_name"]
    actual = agreement["actual_par10"].to_numpy(float)
    predicted = agreement["predicted_median_par10"].to_numpy(float)
    order = np.argsort(predicted)
    bins = np.array_split(order, 10)
    bin_predicted = [np.median(predicted[index]) for index in bins]
    bin_actual = [np.median(actual[index]) for index in bins]

    rng = np.random.RandomState(20_260_731)
    selected = rng.choice(
        len(agreement),
        size=min(5_000, len(agreement)),
        replace=False,
    )
    lower = min(actual.min(), predicted.min())
    upper = max(actual.max(), predicted.max())

    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].scatter(
        predicted[selected],
        actual[selected],
        s=10,
        alpha=0.25,
    )
    axes[0].plot([lower, upper], [lower, upper], "--", color="black", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("EPM median PAR10")
    axes[0].set_ylabel("Archived real PAR10")
    axes[0].set_title("Identical configuration–instance pairs")

    axes[1].plot(bin_predicted, bin_actual, "o-", label="Archived median")
    axes[1].plot([lower, upper], [lower, upper], "--", color="black", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Median EPM PAR10 in prediction decile")
    axes[1].set_ylabel("Median archived PAR10 in prediction decile")
    axes[1].set_title("Decile calibration")
    axes[1].legend()
    figure.suptitle(f"{display_name}: EPM agreement with archived real runs")
    figure.tight_layout()
    return figure, axes


def plot_error_and_timeout(
    summary: dict[str, Any],
    agreement: pd.DataFrame,
) -> tuple[plt.Figure, np.ndarray]:
    display_name = summary["benchmark"]["display_name"]
    confusion = summary["archive_agreement"]["timeout_confusion"]
    matrix = np.asarray(
        [
            [confusion["true_negative"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_positive"]],
        ],
        dtype=int,
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].hist(
        agreement["absolute_log10_error"],
        bins=50,
        color="tab:blue",
        alpha=0.8,
    )
    axes[0].axvline(np.log10(2), color="black", ls="--", label="factor 2")
    axes[0].axvline(np.log10(10), color="tab:red", ls="--", label="factor 10")
    axes[0].set_xlabel("Absolute log10 PAR10 error")
    axes[0].set_ylabel("Archived runs")
    axes[0].set_title("Multiplicative prediction error")
    axes[0].legend()

    image = axes[1].imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axes[1].text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
            )
    axes[1].set_xticks([0, 1], ["Predicted solved", "Predicted timeout"])
    axes[1].set_yticks([0, 1], ["Actually solved", "Actual failure/timeout"])
    axes[1].set_title("Timeout/failure classification")
    figure.colorbar(image, ax=axes[1], label="Archived runs")
    figure.suptitle(f"{display_name}: error and timeout checks")
    figure.tight_layout()
    return figure, axes

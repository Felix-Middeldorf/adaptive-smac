#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Compare five-seed real Clasp distributions with EPM distributions."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from epm.experiment_utils.data_handling import unwarp
from scipy.stats import ks_2samp, wasserstein_distance

from run_validation import (
    DEFAULT_RANDOM_SEED,
    RESULTS,
    ACLibSurrogateBenchmark,
    archived_cost,
    get_benchmark_spec,
    iter_archive_records,
    model_matrix,
    normalize_archive_config,
    predict_raw,
    prediction_costs,
)


COMPARISON_VERSION = 1
DEFAULT_DRAWS = 1_000
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def _quantile_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, QUANTILES)
    return {
        f"{prefix}_q{int(quantile * 100):02d}": float(value)
        for quantile, value in zip(QUANTILES, quantiles)
    }


def find_five_seed_pairs() -> tuple[
    list[tuple[str, str]],
    dict[str, dict[str, Any]],
]:
    pair_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    configurations: dict[str, dict[str, Any]] = {}
    for record in iter_archive_records():
        key = (record.fingerprint, record.instance)
        pair_seeds[key].add(record.solver_seed)
        configurations.setdefault(record.fingerprint, record.config)
    pairs = sorted(
        key for key, seeds in pair_seeds.items() if len(seeds) == 5
    )
    if len(pairs) != 737:
        raise RuntimeError(
            f"Expected 737 pairs with five distinct seeds, found {len(pairs)}."
        )
    return pairs, configurations


def collect_real_costs(
    selected_pairs: set[tuple[str, str]],
    timeout_cost: float,
) -> tuple[
    dict[tuple[str, str], dict[int, list[float]]],
    int,
]:
    observations: dict[
        tuple[str, str], dict[int, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    repeated_same_seed_records = 0
    for record in iter_archive_records():
        key = (record.fingerprint, record.instance)
        if key not in selected_pairs:
            continue
        values = observations[key][record.solver_seed]
        if values:
            repeated_same_seed_records += 1
        values.append(archived_cost(record, timeout_cost))
    return observations, repeated_same_seed_records


def run(
    *,
    draws: int,
    random_seed: int,
    output_directory: Path,
    overwrite: bool,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "five_seed_summary.json"
    if summary_path.is_file() and not overwrite:
        print(f"Existing five-seed comparison found: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.time()
    benchmark = ACLibSurrogateBenchmark(get_benchmark_spec("clasp_queens"))

    print("Finding configuration-instance pairs with five distinct seeds ...")
    pairs, configurations = find_five_seed_pairs()
    observations, repeated_same_seed_records = collect_real_costs(
        set(pairs),
        benchmark.spec.timeout_cost,
    )

    actual_costs = np.empty((len(pairs), 5), dtype=np.float64)
    solver_seeds = np.empty((len(pairs), 5), dtype=np.int64)
    normalized_configs: list[dict[str, Any]] = []
    instances: list[str] = []
    for pair_index, (fingerprint, instance) in enumerate(pairs):
        by_seed = observations[(fingerprint, instance)]
        if len(by_seed) != 5:
            raise RuntimeError("Five-seed pair lost a seed while collecting.")
        ordered_seeds = sorted(by_seed)
        solver_seeds[pair_index] = ordered_seeds
        actual_costs[pair_index] = [
            float(np.mean(by_seed[seed])) for seed in ordered_seeds
        ]
        normalized_configs.append(
            normalize_archive_config(configurations[fingerprint])
        )
        instances.append(instance)

    print(
        f"Drawing {draws} surrogate outcomes for each of {len(pairs)} pairs ..."
    )
    X = model_matrix(benchmark, normalized_configs, instances)
    # ExternalRFRQuantile draws ``num_samples`` quantile levels from the
    # RandomState initialized by one nonzero seed. Passing the complete matrix
    # produces the same quantile grid for every pair in one efficient call.
    quantile_levels = np.random.RandomState(random_seed).uniform(
        low=0.0,
        high=1.0,
        size=draws,
    )
    raw_logged_draws = benchmark.surrogate.model.predict(
        X,
        seed=int(random_seed),
        num_samples=draws,
    )
    raw_draws = np.asarray(
        unwarp(raw_logged_draws, quality=False),
        dtype=np.float64,
    )
    if raw_draws.shape != (len(pairs), draws):
        raise RuntimeError(
            f"Unexpected EPM draw shape {raw_draws.shape}; "
            f"expected {(len(pairs), draws)}."
        )
    surrogate_draws = prediction_costs(
        raw_draws,
        benchmark.spec.cutoff,
        benchmark.spec.timeout_cost,
    ).astype(np.float32)

    median_raw = predict_raw(benchmark, X, seed=0)
    median_costs = prediction_costs(
        median_raw,
        benchmark.spec.cutoff,
        benchmark.spec.timeout_cost,
    )

    rows: list[dict[str, Any]] = []
    for pair_index, ((fingerprint, instance), actual, predicted) in enumerate(
        zip(pairs, actual_costs, surrogate_draws)
    ):
        predicted = np.asarray(predicted, dtype=np.float64)
        actual_timeout_fraction = float(
            np.mean(actual == benchmark.spec.timeout_cost)
        )
        predicted_timeout_fraction = float(
            np.mean(predicted == benchmark.spec.timeout_cost)
        )
        predicted_q05, predicted_q95 = np.quantile(
            predicted, [0.05, 0.95]
        )
        predicted_q25, predicted_q75 = np.quantile(
            predicted, [0.25, 0.75]
        )
        ks_result = ks_2samp(actual, predicted, method="auto")
        row = {
            "pair_index": pair_index,
            "configuration_fingerprint": fingerprint,
            "instance": instance,
            "distinct_real_seeds": 5,
            "surrogate_draws": draws,
            "actual_mean_par10": float(np.mean(actual)),
            "actual_std_par10": float(np.std(actual)),
            "actual_variance_par10": float(np.var(actual)),
            "actual_minimum_par10": float(np.min(actual)),
            "actual_maximum_par10": float(np.max(actual)),
            "actual_timeout_fraction": actual_timeout_fraction,
            "predicted_mean_par10": float(np.mean(predicted)),
            "predicted_std_par10": float(np.std(predicted)),
            "predicted_variance_par10": float(np.var(predicted)),
            "predicted_minimum_par10": float(np.min(predicted)),
            "predicted_maximum_par10": float(np.max(predicted)),
            "predicted_timeout_fraction": predicted_timeout_fraction,
            "predicted_seed_zero_median_par10": float(
                median_costs[pair_index]
            ),
            "wasserstein_par10": float(
                wasserstein_distance(actual, predicted)
            ),
            "ks_statistic": float(ks_result.statistic),
            "ks_pvalue": float(ks_result.pvalue),
            "real_values_inside_predicted_50_interval": float(
                np.mean(
                    (actual >= predicted_q25)
                    & (actual <= predicted_q75)
                )
            ),
            "real_values_inside_predicted_90_interval": float(
                np.mean(
                    (actual >= predicted_q05)
                    & (actual <= predicted_q95)
                )
            ),
            **_quantile_columns("actual", actual),
            **_quantile_columns("predicted", predicted),
        }
        for quantile in QUANTILES:
            suffix = f"q{int(quantile * 100):02d}"
            row[f"absolute_{suffix}_error"] = abs(
                row[f"predicted_{suffix}"] - row[f"actual_{suffix}"]
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    _atomic_csv(output_directory / "five_seed_pairs.csv", frame)
    _atomic_npy(
        output_directory / "five_seed_actual_costs.npy",
        actual_costs.astype(np.float32),
    )
    _atomic_npy(
        output_directory / "five_seed_surrogate_draws.npy",
        surrogate_draws,
    )
    _atomic_npy(
        output_directory / "five_seed_solver_seeds.npy",
        solver_seeds,
    )
    _atomic_npy(
        output_directory / "five_seed_quantile_levels.npy",
        quantile_levels.astype(np.float32),
    )

    quantile_errors = {
        f"q{int(quantile * 100):02d}": {
            "mean_absolute_error": float(
                frame[
                    f"absolute_q{int(quantile * 100):02d}_error"
                ].mean()
            ),
            "median_absolute_error": float(
                frame[
                    f"absolute_q{int(quantile * 100):02d}_error"
                ].median()
            ),
        }
        for quantile in QUANTILES
    }
    summary = {
        "comparison_version": COMPARISON_VERSION,
        "pairs_with_five_distinct_real_seeds": len(pairs),
        "unique_configurations": len({fingerprint for fingerprint, _ in pairs}),
        "unique_instances": len({instance for _, instance in pairs}),
        "real_observations_per_pair": 5,
        "surrogate_draws_per_pair": draws,
        "random_seed": random_seed,
        "same_seed_duplicate_records_averaged": repeated_same_seed_records,
        "mean_wasserstein_par10": float(frame["wasserstein_par10"].mean()),
        "median_wasserstein_par10": float(
            frame["wasserstein_par10"].median()
        ),
        "mean_ks_statistic": float(frame["ks_statistic"].mean()),
        "median_ks_statistic": float(frame["ks_statistic"].median()),
        "mean_real_50_interval_coverage": float(
            frame["real_values_inside_predicted_50_interval"].mean()
        ),
        "mean_real_90_interval_coverage": float(
            frame["real_values_inside_predicted_90_interval"].mean()
        ),
        "spearman_mean_par10": float(
            frame["actual_mean_par10"].corr(
                frame["predicted_mean_par10"], method="spearman"
            )
        ),
        "spearman_std_par10": float(
            frame["actual_std_par10"].corr(
                frame["predicted_std_par10"], method="spearman"
            )
        ),
        "quantile_absolute_errors": quantile_errors,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED + 10)
    parser.add_argument("--output-directory", type=Path, default=RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.draws < 20:
        parser.error("--draws must be at least 20")
    run(
        draws=args.draws,
        random_seed=args.random_seed,
        output_directory=args.output_directory.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

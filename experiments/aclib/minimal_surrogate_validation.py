"""Minimal integrity checks for the downloaded ACLib surrogate benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from ConfigSpace.read_and_write import pcs
from epm.experiment_utils.data_handling import unwarp
from epm.webserver.flask_server_helper import convert_params_to_vec, handle_request

from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    get_benchmark_spec,
    load_benchmark_data,
    par10_cost,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_DATA_ROOT = REPOSITORY_ROOT / "external" / "aclib-performance-data"
DEFAULT_WRAPPER_POINTS = 50
DEFAULT_RANDOM_SEED = 20_260_731
SOLVED_STATUSES = {"SAT", "UNSAT", "SUCCESS"}
FAILED_STATUSES = {"TIMEOUT", "CRASHED", "ABORT"}


@dataclass(frozen=True)
class ValidationDefinition:
    benchmark_key: str
    archive_name: str
    archive_configspace: str
    directory: Path

    @property
    def archive_directory(self) -> Path:
        return PERFORMANCE_DATA_ROOT / self.archive_name

    @property
    def results_directory(self) -> Path:
        return self.directory / "results"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def normalize_archive_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key)[1:] if str(key).startswith("-") else str(key): value
        for key, value in config.items()
    }


def _read_archive_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_number"] = line_number
            records.append(row)
    if not records:
        raise RuntimeError(f"No records found in {path}.")
    return records


def _archive_cost(record: Mapping[str, Any], timeout_cost: float) -> float:
    status = str(record["status"]).upper()
    runtime = float(record["time"])
    if status in SOLVED_STATUSES:
        if not math.isfinite(runtime) or runtime <= 0:
            raise ValueError(
                f"Invalid solved runtime at line {record['_line_number']}: {runtime}"
            )
        return runtime
    if status in FAILED_STATUSES:
        return timeout_cost
    raise ValueError(
        f"Unexpected status at line {record['_line_number']}: {status!r}"
    )


def _read_instance_list(path: Path) -> set[str]:
    return {
        Path(line.split()[0]).name
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _archive_feature_instances(path: Path) -> tuple[set[str], int]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        instances = {Path(row[0]).name for row in reader if row}
    return instances, len(header) - 1


def run_identity_checks(
    definition: ValidationDefinition,
    benchmark: ACLibSurrogateBenchmark,
) -> dict[str, Any]:
    archive = definition.archive_directory
    archive_space_file = archive / definition.archive_configspace
    required = (
        archive / "random_train.json",
        archive / "training.txt",
        archive / "test.txt",
        archive / "features.txt",
        archive_space_file,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing official performance-data files:\n" + "\n".join(missing)
        )

    with archive_space_file.open("r", encoding="utf-8") as handle:
        archive_space = pcs.read(handle)
    packaged_space = benchmark.surrogate.cs

    archive_training = _read_instance_list(archive / "training.txt")
    archive_test = _read_instance_list(archive / "test.txt")
    packaged_data = load_benchmark_data(benchmark.spec)
    packaged_training = {Path(name).name for name in packaged_data.training_instances}
    packaged_test = {Path(name).name for name in packaged_data.test_instances}
    archive_features, archive_feature_dimensions = _archive_feature_instances(
        archive / "features.txt"
    )
    packaged_features = {
        Path(name).name for name in benchmark.surrogate.inst_feat_dict
    }

    checks = {
        "archive_directory": str(archive),
        "configspaces_equal": archive_space == packaged_space,
        "archive_hyperparameters": len(archive_space),
        "packaged_hyperparameters": len(packaged_space),
        "archive_conditions": len(archive_space.conditions),
        "packaged_conditions": len(packaged_space.conditions),
        "archive_forbiddens": len(archive_space.forbidden_clauses),
        "packaged_forbiddens": len(packaged_space.forbidden_clauses),
        "training_instance_sets_equal": archive_training == packaged_training,
        "test_instance_sets_equal": archive_test == packaged_test,
        "feature_instance_sets_equal": archive_features == packaged_features,
        "archive_feature_dimensions": archive_feature_dimensions,
        "packaged_feature_dimensions": benchmark.spec.expected_features,
        "feature_dimensions_equal": (
            archive_feature_dimensions == benchmark.spec.expected_features
        ),
    }
    required_checks = (
        "configspaces_equal",
        "training_instance_sets_equal",
        "test_instance_sets_equal",
        "feature_instance_sets_equal",
        "feature_dimensions_equal",
    )
    checks["all_passed"] = all(bool(checks[key]) for key in required_checks)
    if not checks["all_passed"]:
        raise RuntimeError(f"Archive/EPM identity checks failed: {checks}")
    return checks


def _model_matrix(
    benchmark: ACLibSurrogateBenchmark,
    records: list[dict[str, Any]],
) -> np.ndarray:
    return np.vstack(
        [
            benchmark._model_input(
                normalize_archive_config(record["config"]),
                str(record["instance"]),
            ).reshape(-1)
            for record in records
        ]
    ).astype(np.float32, copy=False)


def _predict_median(
    benchmark: ACLibSurrogateBenchmark,
    matrix: np.ndarray,
    *,
    chunk_size: int = 2_000,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, len(matrix), chunk_size):
        logged = benchmark.surrogate.model.predict(
            matrix[start : start + chunk_size],
            seed=0,
            num_samples=1,
        )
        predictions.append(
            np.asarray(
                unwarp(logged, quality=False),
                dtype=np.float64,
            ).reshape(-1)
        )
    values = np.concatenate(predictions)
    if not np.isfinite(values).all() or (values < 0).any():
        raise RuntimeError("The EPM returned invalid median predictions.")
    return values


def run_wrapper_parity(
    benchmark: ACLibSurrogateBenchmark,
    records: list[dict[str, Any]],
    *,
    count: int,
    random_seed: int,
) -> pd.DataFrame:
    if count < 1 or count > len(records):
        raise ValueError(f"wrapper point count must be in [1, {len(records)}].")
    indices = random.Random(random_seed).sample(range(len(records)), count)
    rows: list[dict[str, Any]] = []

    for index in indices:
        record = records[index]
        archive_params = [
            item
            for key, value in record["config"].items()
            for item in (str(key), str(value))
        ]
        reference_parameters = convert_params_to_vec(
            params=list(archive_params),
            cs=benchmark.surrogate.cs,
            encode=benchmark.surrogate.encode,
            impute_with=benchmark.surrogate.impute_with,
        ).reshape(-1)
        reference_features = np.asarray(
            benchmark.surrogate.inst_feat_dict[str(record["instance"])]
        ).reshape(-1)
        reference_input = np.hstack(
            (reference_parameters, reference_features)
        ).astype(np.float32)
        custom_input = benchmark._model_input(
            normalize_archive_config(record["config"]),
            str(record["instance"]),
        ).reshape(-1)

        request = {
            "instance_name": str(record["instance"]),
            "instance_info": "None",
            "cutoff": benchmark.spec.cutoff,
            "runlength": 0,
            "seed": 0,
            "params": list(archive_params),
        }
        prediction_array, status_array = handle_request(
            request,
            benchmark.surrogate,
        )
        reference_prediction = float(np.asarray(prediction_array).reshape(-1)[0])
        reference_status = str(np.asarray(status_array).reshape(-1)[0])
        reference_cost = par10_cost(
            reference_prediction,
            reference_status,
            benchmark.spec.timeout_cost,
        )
        custom_cost, custom_info = benchmark.evaluate(
            normalize_archive_config(record["config"]),
            str(record["instance"]),
            seed=0,
        )

        input_error = float(np.max(np.abs(reference_input - custom_input)))
        prediction_error = abs(
            reference_prediction
            - float(custom_info["surrogate_prediction"])
        )
        cost_error = abs(reference_cost - custom_cost)
        statuses_equal = (
            reference_status.upper()
            == str(custom_info["surrogate_status"]).upper()
        )
        passed = (
            input_error <= 1e-7
            and prediction_error <= 1e-7
            and cost_error <= 1e-7
            and statuses_equal
        )
        rows.append(
            {
                "random_train_line": int(record["_line_number"]),
                "instance": str(record["instance"]),
                "input_max_absolute_error": input_error,
                "reference_prediction": reference_prediction,
                "custom_prediction": custom_info["surrogate_prediction"],
                "prediction_absolute_error": prediction_error,
                "reference_status": reference_status,
                "custom_status": custom_info["surrogate_status"],
                "statuses_equal": statuses_equal,
                "reference_par10": reference_cost,
                "custom_par10": custom_cost,
                "cost_absolute_error": cost_error,
                "passed": passed,
            }
        )

    frame = pd.DataFrame(rows)
    if not frame["passed"].all():
        raise RuntimeError("Custom wrapper does not reproduce the official path.")
    return frame


def evaluate_archive_agreement(
    benchmark: ACLibSurrogateBenchmark,
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    matrix = _model_matrix(benchmark, records)
    raw_predictions = _predict_median(benchmark, matrix)
    predicted_par10 = np.where(
        raw_predictions > benchmark.spec.cutoff,
        benchmark.spec.timeout_cost,
        raw_predictions,
    )
    actual_par10 = np.asarray(
        [
            _archive_cost(record, benchmark.spec.timeout_cost)
            for record in records
        ],
        dtype=np.float64,
    )
    actual_statuses = np.asarray(
        [str(record["status"]).upper() for record in records]
    )
    actual_timeout = ~np.isin(actual_statuses, list(SOLVED_STATUSES))
    predicted_timeout = raw_predictions > benchmark.spec.cutoff
    absolute_log_error = np.abs(
        np.log10(predicted_par10) - np.log10(actual_par10)
    )

    return pd.DataFrame(
        {
            "random_train_line": [
                int(record["_line_number"]) for record in records
            ],
            "instance": [str(record["instance"]) for record in records],
            "solver_seed": [int(record["seed"]) for record in records],
            "actual_status": actual_statuses,
            "actual_runtime": [
                float(record["time"]) for record in records
            ],
            "actual_par10": actual_par10,
            "predicted_median_runtime": raw_predictions,
            "predicted_median_par10": predicted_par10,
            "actual_timeout_or_failure": actual_timeout,
            "predicted_timeout": predicted_timeout,
            "absolute_log10_error": absolute_log_error,
            "multiplicative_error": np.power(10.0, absolute_log_error),
        }
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _quantiles(values: pd.Series) -> dict[str, float]:
    return {
        "q10": float(values.quantile(0.10)),
        "q25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
    }


def summarize(
    definition: ValidationDefinition,
    benchmark: ACLibSurrogateBenchmark,
    identity: Mapping[str, Any],
    parity: pd.DataFrame,
    agreement: pd.DataFrame,
) -> dict[str, Any]:
    actual_timeout = agreement["actual_timeout_or_failure"].astype(bool)
    predicted_timeout = agreement["predicted_timeout"].astype(bool)
    true_positive = int((actual_timeout & predicted_timeout).sum())
    false_positive = int((~actual_timeout & predicted_timeout).sum())
    false_negative = int((actual_timeout & ~predicted_timeout).sum())
    true_negative = int((~actual_timeout & ~predicted_timeout).sum())
    actual = agreement["actual_par10"]
    predicted = agreement["predicted_median_par10"]
    errors = agreement["absolute_log10_error"]
    near_cutoff = (
        agreement["predicted_median_runtime"]
        >= benchmark.spec.cutoff * (1.0 - 1e-6)
    ) & (
        agreement["predicted_median_runtime"]
        <= benchmark.spec.cutoff * (1.0 + 1e-6)
    )
    spearman = float(actual.corr(predicted, method="spearman"))
    spearman_log = float(
        np.log10(actual).corr(np.log10(predicted), method="spearman")
    )

    return {
        "validation_version": 1,
        "benchmark": {
            "key": definition.benchmark_key,
            "display_name": benchmark.spec.display_name,
            "archive_name": definition.archive_name,
            "model_type": benchmark.model_type,
            "cutoff": benchmark.spec.cutoff,
            "par10_timeout_cost": benchmark.spec.timeout_cost,
            "deterministic_scenario": benchmark.spec.deterministic,
        },
        "data_identity": dict(identity),
        "wrapper_parity": {
            "tested_points": len(parity),
            "all_passed": bool(parity["passed"].all()),
            "maximum_input_absolute_error": float(
                parity["input_max_absolute_error"].max()
            ),
            "maximum_prediction_absolute_error": float(
                parity["prediction_absolute_error"].max()
            ),
            "maximum_cost_absolute_error": float(
                parity["cost_absolute_error"].max()
            ),
        },
        "archive_agreement": {
            "comparison": (
                "One archived random training run versus the EPM median for "
                "the identical configuration-instance pair"
            ),
            "points": len(agreement),
            "spearman_par10": spearman,
            "spearman_log10_par10": spearman_log,
            "median_absolute_log10_error": float(errors.median()),
            "mean_absolute_log10_error": float(errors.mean()),
            "fraction_within_factor_2": float((errors <= math.log10(2)).mean()),
            "fraction_within_factor_5": float((errors <= math.log10(5)).mean()),
            "fraction_within_factor_10": float((errors <= 1.0).mean()),
            "actual_par10_quantiles": _quantiles(actual),
            "predicted_median_par10_quantiles": _quantiles(predicted),
            "actual_timeout_or_failure_fraction": float(actual_timeout.mean()),
            "predicted_timeout_fraction": float(predicted_timeout.mean()),
            "predicted_median_near_cutoff_fraction": float(
                near_cutoff.mean()
            ),
            "actual_timeouts_with_near_cutoff_prediction_fraction": (
                float(near_cutoff[actual_timeout].mean())
                if actual_timeout.any()
                else None
            ),
            "timeout_confusion": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": _safe_ratio(
                    true_positive, true_positive + false_positive
                ),
                "recall": _safe_ratio(
                    true_positive, true_positive + false_negative
                ),
            },
        },
        "limitations": [
            (
                "The official random_train archive contains one real run per "
                "row, while the stochastic EPM comparison uses its conditional "
                "median. Individual-row disagreement therefore includes real "
                "target stochasticity."
            ),
            (
                "These archives were used to construct the EPMs, so this is an "
                "integrity/reproduction check rather than an independent "
                "generalization test."
            ),
        ],
    }


def run_validation(
    definition: ValidationDefinition,
    *,
    wrapper_points: int = DEFAULT_WRAPPER_POINTS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = definition.results_directory
    summary_path = output / "summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_path} already exists; pass --overwrite to replace results."
        )

    benchmark = ACLibSurrogateBenchmark(
        get_benchmark_spec(definition.benchmark_key)
    )
    identity = run_identity_checks(definition, benchmark)
    records = _read_archive_records(
        definition.archive_directory / "random_train.json"
    )
    parity = run_wrapper_parity(
        benchmark,
        records,
        count=wrapper_points,
        random_seed=random_seed,
    )
    agreement = evaluate_archive_agreement(benchmark, records)
    summary = summarize(
        definition,
        benchmark,
        identity,
        parity,
        agreement,
    )

    output.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output / "wrapper_parity.csv", parity)
    _atomic_csv(output / "archive_agreement.csv", agreement)
    _atomic_json(summary_path, summary)
    return summary


def cli(definition: ValidationDefinition) -> None:
    parser = argparse.ArgumentParser(
        description=f"Validate the {definition.benchmark_key} ACLib surrogate."
    )
    parser.add_argument(
        "--wrapper-points",
        type=int,
        default=DEFAULT_WRAPPER_POINTS,
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_validation(
        definition,
        wrapper_points=args.wrapper_points,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

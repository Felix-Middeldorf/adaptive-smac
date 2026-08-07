"""Reproducible random-configuration sampling for ACLib surrogate targets."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from ConfigSpace import Configuration

from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)


N_CONFIGURATIONS = 1_000
CONFIGSPACE_SEED = 20_260_729
QUANTILE_SEEDS = (
    0,
    475_207_229,
    1_792_501_355,
    850_460_608,
    1_773_604_054,
    1_755_497_032,
)
EXPERIMENT_VERSION = 1


@dataclass(frozen=True)
class RandomSamplingDefinition:
    benchmark_key: str
    initials: str
    directory: Path

    @property
    def output_directory(self) -> Path:
        return self.directory / "results"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_configuration(config: Configuration | dict[str, Any]) -> str:
    return json.dumps(
        _jsonable(dict(config)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _jsonable(row),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def sample_unique_configurations(
    configspace: Any,
    *,
    n_configurations: int,
    seed: int,
) -> tuple[list[Configuration], int]:
    """Draw exactly ``n_configurations`` unique active configurations."""
    sampling_space = copy.deepcopy(configspace)
    sampling_space.seed(seed)
    configurations: list[Configuration] = []
    fingerprints: set[str] = set()
    draws = 0

    while len(configurations) < n_configurations:
        batch_size = min(256, max(16, n_configurations - len(configurations)))
        sampled = sampling_space.sample_configuration(size=batch_size)
        if isinstance(sampled, Configuration):
            sampled = [sampled]
        for config in sampled:
            draws += 1
            fingerprint = _canonical_configuration(config)
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint)
            configurations.append(config)
            if len(configurations) == n_configurations:
                break

    return configurations, draws


def _configuration_digest(configurations: list[Configuration]) -> str:
    digest = hashlib.sha256()
    for config in configurations:
        digest.update(_canonical_configuration(config).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _instance_digest(instances: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for instance in instances:
        digest.update(instance.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_configurations(
    path: Path,
    configspace: Any,
) -> list[Configuration]:
    configurations: list[Configuration] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        configurations.append(
            Configuration(
                configuration_space=configspace,
                values=row["configuration"],
            )
        )
    return configurations


def _build_input_matrix(
    benchmark: ACLibSurrogateBenchmark,
    config: Configuration,
    feature_matrix: np.ndarray,
    first_instance: str,
) -> np.ndarray:
    if benchmark.surrogate.encode:
        raise RuntimeError(
            "This sampler expects the downloaded ACLib EPM without one-hot "
            "encoding."
        )
    n_parameters = len(benchmark.surrogate.cs)
    first = benchmark._model_input(config, first_instance).reshape(-1)
    parameters = first[:n_parameters]
    repeated = np.repeat(
        parameters.reshape(1, -1),
        repeats=feature_matrix.shape[0],
        axis=0,
    )
    return np.hstack((repeated, feature_matrix)).astype(np.float32, copy=False)


def _evaluate_matrix(
    benchmark: ACLibSurrogateBenchmark,
    X: np.ndarray,
    quantile_seed: int,
) -> np.ndarray:
    # ACLib's legacy SurrogateModel has an incorrect status-vector branch for
    # multi-row calls: the SAT/CUTOFF labels are reversed. Keep its efficient
    # configuration encoding, but predict one row at a time exactly as the
    # official target wrapper does.
    costs = np.empty(X.shape[0], dtype=np.float32)
    valid_statuses = {"TRUE", "SAT", "SUCCESS", "CUTOFF"}
    for index, row in enumerate(X):
        predictions, statuses = benchmark.surrogate.predict(
            X=row.reshape(1, -1),
            quality=False,
            cutoff=benchmark.spec.cutoff,
            quantile_seed=quantile_seed,
        )
        prediction = float(np.asarray(predictions).reshape(-1)[0])
        status = str(np.asarray(statuses).reshape(-1)[0]).upper()
        if status not in valid_statuses:
            raise RuntimeError(f"Unexpected ACLib EPM status: {status}")
        costs[index] = (
            benchmark.spec.timeout_cost
            if status == "CUTOFF"
            else prediction
        )
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise RuntimeError("The ACLib EPM returned an invalid cost.")
    return costs


def _write_summary(
    path: Path,
    costs: np.ndarray,
    quantile_seeds: tuple[int, ...],
    timeout_cost: float,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = (
        "configuration_index",
        "quantile_seed",
        "mean_par10",
        "median_par10",
        "std_par10",
        "minimum_par10",
        "q10_par10",
        "q25_par10",
        "q75_par10",
        "q90_par10",
        "maximum_par10",
        "timeout_fraction",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for config_index in range(costs.shape[0]):
            for seed_index, quantile_seed in enumerate(quantile_seeds):
                values = np.asarray(costs[config_index, seed_index], dtype=float)
                q10, q25, q75, q90 = np.quantile(
                    values, [0.10, 0.25, 0.75, 0.90]
                )
                writer.writerow(
                    {
                        "configuration_index": config_index,
                        "quantile_seed": quantile_seed,
                        "mean_par10": float(np.mean(values)),
                        "median_par10": float(np.median(values)),
                        "std_par10": float(np.std(values)),
                        "minimum_par10": float(np.min(values)),
                        "q10_par10": float(q10),
                        "q25_par10": float(q25),
                        "q75_par10": float(q75),
                        "q90_par10": float(q90),
                        "maximum_par10": float(np.max(values)),
                        "timeout_fraction": float(
                            np.mean(values == timeout_cost)
                        ),
                    }
                )
    os.replace(temporary, path)


def run_random_sampling(
    definition: RandomSamplingDefinition,
    *,
    n_configurations: int = N_CONFIGURATIONS,
    configspace_seed: int = CONFIGSPACE_SEED,
    quantile_seeds: tuple[int, ...] = QUANTILE_SEEDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    if n_configurations < 1:
        raise ValueError("n_configurations must be positive.")
    if not quantile_seeds or quantile_seeds[0] != 0:
        raise ValueError("quantile_seeds must begin with deterministic seed 0.")
    if len(set(quantile_seeds)) != len(quantile_seeds):
        raise ValueError("quantile_seeds must be unique.")

    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    output = definition.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    configurations_path = output / "configurations.jsonl"
    costs_path = output / "costs.npy"
    progress_path = output / "progress.json"
    summary_path = output / "configuration_seed_summary.csv"

    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": spec.key,
        "display_name": spec.display_name,
        "n_configurations": n_configurations,
        "configspace_seed": configspace_seed,
        "quantile_seeds": list(quantile_seeds),
        "n_training_instances": len(data.training_instances),
        "training_instance_sha256": _instance_digest(data.training_instances),
        "training_only": True,
        "test_instances_used": 0,
        "cutoff": spec.cutoff,
        "timeout_cost": spec.timeout_cost,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "assets": asset_metadata(spec),
    }

    resume = metadata_path.exists() and not overwrite
    if resume:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity") != identity:
            raise RuntimeError(
                "Existing sampling output has a different identity. "
                "Use --overwrite or another output directory."
            )
        configurations = _load_configurations(
            configurations_path, data.configspace
        )
        if len(configurations) != n_configurations:
            raise RuntimeError("Stored configuration count is inconsistent.")
        costs = np.lib.format.open_memmap(costs_path, mode="r+")
        expected_shape = (
            n_configurations,
            len(quantile_seeds),
            len(data.training_instances),
        )
        if costs.shape != expected_shape:
            raise RuntimeError(
                f"Stored cost shape {costs.shape} != {expected_shape}."
            )
    else:
        configurations, total_draws = sample_unique_configurations(
            data.configspace,
            n_configurations=n_configurations,
            seed=configspace_seed,
        )
        _atomic_jsonl(
            configurations_path,
            (
                {
                    "configuration_index": index,
                    "configuration": dict(config),
                    "origin": config.origin,
                }
                for index, config in enumerate(configurations)
            ),
        )
        metadata = {
            "state": "running",
            "identity": identity,
            "total_configspace_draws_including_duplicates": total_draws,
            "configuration_sha256": _configuration_digest(configurations),
            "cost_tensor_axes": [
                "configuration_index",
                "quantile_seed_index",
                "training_instance_index",
            ],
            "training_instances": list(data.training_instances),
        }
        _atomic_json(metadata_path, metadata)
        costs = np.lib.format.open_memmap(
            costs_path,
            mode="w+",
            dtype=np.float32,
            shape=(
                n_configurations,
                len(quantile_seeds),
                len(data.training_instances),
            ),
        )
        costs[:] = np.nan
        costs.flush()

    completed = np.isfinite(costs).all(axis=(1, 2))
    benchmark = ACLibSurrogateBenchmark(spec)
    feature_matrix = np.asarray(
        [data.features[instance] for instance in data.training_instances],
        dtype=np.float32,
    )
    started = time.time()
    already_complete = int(np.sum(completed))

    for config_index, config in enumerate(configurations):
        if completed[config_index]:
            continue
        X = _build_input_matrix(
            benchmark,
            config,
            feature_matrix,
            data.training_instances[0],
        )
        for seed_index, quantile_seed in enumerate(quantile_seeds):
            values = _evaluate_matrix(benchmark, X, quantile_seed)
            costs[config_index, seed_index, :] = values

            if config_index == 0:
                scalar_cost, _ = benchmark.evaluate(
                    config,
                    data.training_instances[0],
                    seed=quantile_seed,
                )
                if not np.isclose(
                    values[0], scalar_cost, rtol=0.0, atol=1e-6
                ):
                    raise RuntimeError(
                        "Vectorized evaluation disagrees with the official "
                        "single-instance adapter."
                    )

        costs.flush()
        completed[config_index] = True
        finished = int(np.sum(completed))
        elapsed = time.time() - started
        _atomic_json(
            progress_path,
            {
                "state": "running",
                "completed_configurations": finished,
                "total_configurations": n_configurations,
                "fraction_complete": finished / n_configurations,
                "completed_in_this_process": finished - already_complete,
                "elapsed_seconds_this_process": elapsed,
            },
        )
        if finished % 10 == 0 or finished == n_configurations:
            rate = (finished - already_complete) / max(elapsed, 1e-9)
            print(
                f"evaluated {finished}/{n_configurations} configurations "
                f"({rate:.3f} configurations/s)",
                flush=True,
            )

    final_costs = np.asarray(costs)
    if not np.isfinite(final_costs).all():
        raise RuntimeError("Sampling ended with incomplete costs.")
    _write_summary(
        summary_path,
        final_costs,
        quantile_seeds,
        spec.timeout_cost,
    )
    elapsed = time.time() - started
    metadata["state"] = "complete"
    metadata["completed_at_unix_seconds"] = time.time()
    metadata["walltime_seconds_this_process"] = elapsed
    _atomic_json(metadata_path, metadata)
    result = {
        "state": "complete",
        "benchmark": spec.key,
        "configurations": n_configurations,
        "training_instances": len(data.training_instances),
        "quantile_seeds": list(quantile_seeds),
        "target_evaluations": (
            n_configurations
            * len(data.training_instances)
            * len(quantile_seeds)
        ),
        "costs": str(costs_path),
        "summary": str(summary_path),
    }
    _atomic_json(progress_path, result)
    return result

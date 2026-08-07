#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Validate the ACLib Clasp Queens surrogate against archived real runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from ConfigSpace.read_and_write import pcs
from epm.experiment_utils.data_handling import unwarp
from epm.webserver.flask_server_helper import (
    convert_params_to_vec,
    handle_request,
)
from epm.webserver.flask_worker_helper import (
    retrieve_credentials,
    send_shutdown_signal,
)


HERE = Path(__file__).resolve().parent
ACLIB_EXPERIMENT_ROOT = HERE.parents[1]
REPOSITORY_ROOT = HERE.parents[3]
if str(ACLIB_EXPERIMENT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ACLIB_EXPERIMENT_ROOT))

from surrogate_benchmark import (  # noqa: E402
    ACLibSurrogateBenchmark,
    get_benchmark_spec,
    load_benchmark_data,
    par10_cost,
)


ARCHIVE_ROOT = (
    REPOSITORY_ROOT / "external" / "aclib-performance-data" / "clasp_queens"
)
RESULTS = HERE / "results"
VALIDATION_VERSION = 1
DEFAULT_RANDOM_SEED = 20_260_730
SOLVED_STATUSES = {"SAT", "UNSAT", "SUCCESS"}
UNSUCCESSFUL_STATUSES = {"TIMEOUT", "CRASHED", "ABORT"}


@dataclass(frozen=True)
class ArchivedRecord:
    source: str
    line_number: int
    status: str
    runtime: float
    instance: str
    solver_seed: int
    config: dict[str, Any]
    fingerprint: str

    @property
    def record_id(self) -> str:
        payload = (
            f"{self.source}:{self.line_number}:{self.fingerprint}:"
            f"{self.instance}:{self.solver_seed}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


class StratifiedReservoir:
    """Fixed-size deterministic reservoir for each runtime/status stratum."""

    def __init__(self, quotas: Mapping[str, int], seed: int) -> None:
        self.quotas = dict(quotas)
        self.rng = random.Random(seed)
        self.seen: Counter[str] = Counter()
        self.rows: dict[str, list[ArchivedRecord]] = {
            key: [] for key in quotas
        }

    def add(self, stratum: str, record: ArchivedRecord) -> None:
        if stratum not in self.quotas:
            raise KeyError(stratum)
        self.seen[stratum] += 1
        quota = self.quotas[stratum]
        rows = self.rows[stratum]
        if len(rows) < quota:
            rows.append(record)
            return
        replacement = self.rng.randrange(self.seen[stratum])
        if replacement < quota:
            rows[replacement] = record

    def selected(self) -> list[ArchivedRecord]:
        return [
            row
            for stratum in self.quotas
            for row in self.rows[stratum]
        ]


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


def _numeric_canonical(value: Any) -> str:
    text = str(value)
    try:
        return str(Decimal(text).normalize())
    except InvalidOperation:
        return text


def normalize_archive_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the old command-line dash from archived PCS names."""
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        key = str(key)
        normalized[key[1:] if key.startswith("-") else key] = value
    return normalized


def configuration_fingerprint(config: Mapping[str, Any]) -> str:
    normalized = normalize_archive_config(config)
    canonical = sorted(
        (str(key), _numeric_canonical(value))
        for key, value in normalized.items()
    )
    serialized = json.dumps(canonical, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def iter_archive_records() -> Iterator[ArchivedRecord]:
    data_directory = ARCHIVE_ROOT / "data_train"
    files = sorted(data_directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No archived JSON files under {data_directory}")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                config = dict(row["config"])
                yield ArchivedRecord(
                    source=path.stem,
                    line_number=line_number,
                    status=str(row["status"]).upper(),
                    runtime=float(row["time"]),
                    instance=str(row["instance"]),
                    solver_seed=int(row["seed"]),
                    config=config,
                    fingerprint=configuration_fingerprint(config),
                )


def archived_cost(record: ArchivedRecord, timeout_cost: float) -> float:
    if record.status in SOLVED_STATUSES:
        if not math.isfinite(record.runtime) or record.runtime <= 0:
            raise ValueError(f"Invalid solved runtime in {record.record_id}")
        return record.runtime
    if record.status in UNSUCCESSFUL_STATUSES:
        return timeout_cost
    raise ValueError(
        f"Unexpected archived status {record.status!r} in {record.record_id}"
    )


def performance_stratum(record: ArchivedRecord) -> str:
    if record.status not in SOLVED_STATUSES:
        return "timeout_or_failure"
    if record.runtime <= 1:
        return "solved_le_1s"
    if record.runtime <= 10:
        return "solved_1_10s"
    if record.runtime <= 100:
        return "solved_10_100s"
    return "solved_100_300s"


def quotas_for(total: int) -> dict[str, int]:
    strata = (
        "solved_le_1s",
        "solved_1_10s",
        "solved_10_100s",
        "solved_100_300s",
        "timeout_or_failure",
    )
    base, remainder = divmod(total, len(strata))
    return {
        stratum: base + (index < remainder)
        for index, stratum in enumerate(strata)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_asset_checks(
    benchmark: ACLibSurrogateBenchmark,
) -> dict[str, Any]:
    spec = benchmark.spec
    archive_training = ARCHIVE_ROOT / "training.txt"
    archive_test = ARCHIVE_ROOT / "test.txt"
    archive_pcs = ARCHIVE_ROOT / "clasp-sat-params-nat.pcs"
    archive_features = ARCHIVE_ROOT / "features.txt"
    required = (
        archive_training,
        archive_test,
        archive_pcs,
        archive_features,
        ARCHIVE_ROOT / "data_train",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing performance-data inputs:\n" + "\n".join(missing))

    with archive_pcs.open("r", encoding="utf-8") as handle:
        archive_space = pcs.read(handle)
    local_space = benchmark.surrogate.cs

    with archive_features.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    feature_names = rows[0][1:]
    instance_names = [row[0] for row in rows[1:]]
    raw_features = np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]],
        dtype=np.float64,
    )
    local_feature_names = set(benchmark.surrogate.inst_feat_dict)
    if len(set(instance_names)) != len(instance_names):
        raise RuntimeError("Archive feature table has duplicate instance names.")
    if set(instance_names) != local_feature_names:
        raise RuntimeError("Archive and packaged EPM instance names differ.")

    minimum = np.min(raw_features, axis=0)
    maximum = np.max(raw_features, axis=0)
    denominator = maximum - minimum
    normalized = np.divide(
        raw_features - minimum,
        denominator,
        out=np.zeros_like(raw_features),
        where=denominator != 0,
    )
    packaged = np.asarray(
        [
            np.asarray(benchmark.surrogate.inst_feat_dict[name]).reshape(-1)
            for name in instance_names
        ],
        dtype=np.float64,
    )
    feature_max_abs_error = float(np.max(np.abs(normalized - packaged)))

    checks = {
        "archive_root": str(ARCHIVE_ROOT),
        "archive_training_instances": len(
            archive_training.read_text(encoding="utf-8").splitlines()
        ),
        "archive_test_instances": len(
            archive_test.read_text(encoding="utf-8").splitlines()
        ),
        "training_list_sha256_archive": _sha256(archive_training),
        "training_list_sha256_local": _sha256(spec.training_file),
        "test_list_sha256_archive": _sha256(archive_test),
        "test_list_sha256_local": _sha256(spec.test_file),
        "training_lists_identical": (
            archive_training.read_bytes() == spec.training_file.read_bytes()
        ),
        "test_lists_identical": (
            archive_test.read_bytes() == spec.test_file.read_bytes()
        ),
        "configspaces_equal": archive_space == local_space,
        "archive_hyperparameters": len(archive_space),
        "local_hyperparameters": len(local_space),
        "archive_conditions": len(archive_space.conditions),
        "local_conditions": len(local_space.conditions),
        "archive_forbiddens": len(archive_space.forbidden_clauses),
        "local_forbiddens": len(local_space.forbidden_clauses),
        "archive_feature_instances": len(instance_names),
        "packaged_feature_instances": len(local_feature_names),
        "feature_dimensions": len(feature_names),
        "packaged_feature_dimensions": packaged.shape[1],
        "feature_instance_names_equal": set(instance_names) == local_feature_names,
        "feature_minmax_max_absolute_error": feature_max_abs_error,
        "feature_minmax_match_within_5e_6": feature_max_abs_error <= 5e-6,
    }
    required_true = (
        "training_lists_identical",
        "test_lists_identical",
        "configspaces_equal",
        "feature_instance_names_equal",
        "feature_minmax_match_within_5e_6",
    )
    checks["all_required_checks_pass"] = all(checks[key] for key in required_true)
    if not checks["all_required_checks_pass"]:
        raise RuntimeError(f"Asset validation failed: {checks}")
    return checks


def collect_archive_index(
    point_sample_count: int,
    random_seed: int,
) -> tuple[
    list[ArchivedRecord],
    dict[str, set[str]],
    dict[str, dict[str, Any]],
    dict[str, set[str]],
    Counter[tuple[str, str]],
    Counter[str],
]:
    reservoir = StratifiedReservoir(
        quotas=quotas_for(point_sample_count),
        seed=random_seed,
    )
    coverage: dict[str, set[str]] = defaultdict(set)
    configs: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    pair_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()

    for record in iter_archive_records():
        reservoir.add(performance_stratum(record), record)
        coverage[record.fingerprint].add(record.instance)
        configs.setdefault(record.fingerprint, record.config)
        sources[record.fingerprint].add(record.source)
        pair_counts[(record.fingerprint, record.instance)] += 1
        status_counts[record.status] += 1

    selected = reservoir.selected()
    if len(selected) != point_sample_count:
        found = Counter(performance_stratum(row) for row in selected)
        raise RuntimeError(
            f"Could not fill the stratified sample: requested "
            f"{point_sample_count}, found {len(selected)} ({found})."
        )
    return selected, coverage, configs, sources, pair_counts, status_counts


def model_matrix(
    benchmark: ACLibSurrogateBenchmark,
    configs: Sequence[Mapping[str, Any]],
    instances: Sequence[str],
) -> np.ndarray:
    if len(configs) != len(instances):
        raise ValueError("Configuration and instance counts differ.")
    return np.vstack(
        [
            benchmark._model_input(config, instance).reshape(-1)
            for config, instance in zip(configs, instances)
        ]
    ).astype(np.float32, copy=False)


def predict_raw(
    benchmark: ACLibSurrogateBenchmark,
    X: np.ndarray,
    *,
    seed: int,
    chunk_size: int = 2_048,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(X), chunk_size):
        raw_logged = benchmark.surrogate.model.predict(
            X[start : start + chunk_size],
            seed=int(seed),
            num_samples=1,
        )
        values.append(
            np.asarray(
                unwarp(raw_logged, quality=False),
                dtype=np.float64,
            ).reshape(-1)
        )
    result = np.concatenate(values) if values else np.empty(0, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any():
        raise RuntimeError("EPM returned invalid predictions.")
    return result


def prediction_costs(
    raw_predictions: np.ndarray,
    cutoff: float,
    timeout_cost: float,
) -> np.ndarray:
    return np.where(raw_predictions > cutoff, timeout_cost, raw_predictions)


def run_wrapper_parity(
    benchmark: ACLibSurrogateBenchmark,
    sample: Sequence[ArchivedRecord],
    count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in sample[:count]:
        normalized_config = normalize_archive_config(record.config)
        archive_params = [
            item
            for key, value in record.config.items()
            for item in (str(key), str(value))
        ]
        reference_parameters = convert_params_to_vec(
            params=list(archive_params),
            cs=benchmark.surrogate.cs,
            encode=benchmark.surrogate.encode,
            impute_with=benchmark.surrogate.impute_with,
        ).reshape(-1)
        reference_features = np.asarray(
            benchmark.surrogate.inst_feat_dict[record.instance]
        ).reshape(-1)
        reference_input = np.hstack(
            (reference_parameters, reference_features)
        ).astype(np.float32)
        adapter_input = benchmark._model_input(
            normalized_config, record.instance
        ).reshape(-1)

        json_request = {
            "instance_name": record.instance,
            "instance_info": "None",
            "cutoff": benchmark.spec.cutoff,
            "runlength": 0,
            "seed": 0,
            "params": list(archive_params),
        }
        reference_prediction_array, reference_status_array = handle_request(
            json_request,
            benchmark.surrogate,
        )
        reference_prediction = float(
            np.asarray(reference_prediction_array).reshape(-1)[0]
        )
        reference_status = str(
            np.asarray(reference_status_array).reshape(-1)[0]
        )
        reference_cost = par10_cost(
            reference_prediction,
            reference_status,
            benchmark.spec.timeout_cost,
        )
        adapter_cost, adapter_info = benchmark.evaluate(
            normalized_config,
            record.instance,
            seed=0,
        )
        input_max_abs_error = float(
            np.max(np.abs(reference_input - adapter_input))
        )
        prediction_abs_error = abs(
            reference_prediction
            - float(adapter_info["surrogate_prediction"])
        )
        cost_abs_error = abs(reference_cost - adapter_cost)
        status_equal = (
            reference_status.upper()
            == str(adapter_info["surrogate_status"]).upper()
        )
        passed = (
            input_max_abs_error <= 1e-7
            and prediction_abs_error <= 1e-7
            and cost_abs_error <= 1e-7
            and status_equal
        )
        rows.append(
            {
                "record_id": record.record_id,
                "source": record.source,
                "instance": record.instance,
                "input_max_absolute_error": input_max_abs_error,
                "reference_prediction": reference_prediction,
                "adapter_prediction": adapter_info["surrogate_prediction"],
                "prediction_absolute_error": prediction_abs_error,
                "reference_status": reference_status,
                "adapter_status": adapter_info["surrogate_status"],
                "status_equal": status_equal,
                "reference_par10": reference_cost,
                "adapter_par10": adapter_cost,
                "cost_absolute_error": cost_abs_error,
                "passed": passed,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["passed"].all():
        raise RuntimeError("Official wrapper-core parity failed.")
    return frame


def run_command_line_wrapper_parity(
    benchmark: ACLibSurrogateBenchmark,
    sample: Sequence[ArchivedRecord],
) -> pd.DataFrame:
    """Exercise the complete legacy CLI/communicator/HTTP-server path."""
    records_by_stratum: dict[str, ArchivedRecord] = {}
    for record in sample:
        records_by_stratum.setdefault(performance_stratum(record), record)
    records = list(records_by_stratum.values())
    if len(records) != len(quotas_for(5)):
        raise RuntimeError("CLI parity sample does not cover all five strata.")

    target_directory = (
        REPOSITORY_ROOT
        / "external"
        / "aclib-surrogates"
        / "aclib2"
        / "target_algorithms"
        / "surrogate"
    )
    wrapper = target_directory / "wrapper.py"
    model_directory = target_directory / benchmark.spec.model_name
    base_command = [
        sys.executable,
        str(wrapper),
        "--runsolver-path",
        "None",
        "--idle_time",
        "100",
        "--quality",
        "0",
        "--pyrfr_wrapper",
        str(benchmark.spec.wrapper_file),
        "--pyrfr_model",
        str(benchmark.spec.model_file),
        "--config_space",
        str(benchmark.spec.configspace_file),
        "--inst_feat_dict",
        str(benchmark.spec.feature_file),
    ]
    result_pattern = re.compile(
        r"Result for ParamILS:\s*([^,]+),\s*([^,]+),"
    )
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cq-cli-wrapper-") as temporary:
        working_directory = Path(temporary)
        try:
            for index, record in enumerate(records):
                command = [
                    *base_command,
                    record.instance,
                    "None",
                    str(benchmark.spec.cutoff),
                    "0",
                    "0",
                ]
                for key, value in record.config.items():
                    command.extend((str(key), str(value)))
                stdout_path = working_directory / f"stdout_{index}.txt"
                stderr_path = working_directory / f"stderr_{index}.txt"
                with stdout_path.open("w", encoding="utf-8") as stdout, (
                    stderr_path.open("w", encoding="utf-8")
                ) as stderr:
                    completed = subprocess.run(
                        command,
                        cwd=working_directory,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=45,
                        check=False,
                    )
                output = stdout_path.read_text(encoding="utf-8")
                error_output = stderr_path.read_text(encoding="utf-8")
                match = result_pattern.search(output)
                if completed.returncode != 0 or match is None:
                    raise RuntimeError(
                        "Official command-line wrapper failed.\n"
                        f"stdout:\n{output[-2000:]}\n"
                        f"stderr:\n{error_output[-2000:]}"
                    )
                cli_status = match.group(1).strip().upper()
                cli_runtime = float(match.group(2))
                cli_cost = (
                    benchmark.spec.timeout_cost
                    if cli_status == "TIMEOUT"
                    else cli_runtime
                )
                adapter_cost, adapter_info = benchmark.evaluate(
                    normalize_archive_config(record.config),
                    record.instance,
                    seed=0,
                )
                expected_cli_status = (
                    "TIMEOUT"
                    if str(adapter_info["surrogate_status"]).upper() == "CUTOFF"
                    else "SUCCESS"
                )
                cost_absolute_error = abs(cli_cost - adapter_cost)
                passed = (
                    cli_status == expected_cli_status
                    and cost_absolute_error <= 1e-3
                )
                rows.append(
                    {
                        "record_id": record.record_id,
                        "stratum": performance_stratum(record),
                        "instance": record.instance,
                        "cli_status": cli_status,
                        "adapter_equivalent_status": expected_cli_status,
                        "status_equal": cli_status == expected_cli_status,
                        "cli_runtime_field": cli_runtime,
                        "cli_par10": cli_cost,
                        "adapter_par10": adapter_cost,
                        "cost_absolute_error": cost_absolute_error,
                        "returncode": completed.returncode,
                        "passed": passed,
                    }
                )
        finally:
            credentials = working_directory / "nameserver_creds.pkl"
            if credentials.is_file():
                try:
                    host, port, _ = retrieve_credentials(working_directory)
                    send_shutdown_signal(host, port)
                except Exception:
                    pass
    frame = pd.DataFrame(rows)
    if frame.empty or not frame["passed"].all():
        raise RuntimeError("End-to-end command-line wrapper parity failed.")
    return frame


def evaluate_point_sample(
    benchmark: ACLibSurrogateBenchmark,
    sample: Sequence[ArchivedRecord],
) -> pd.DataFrame:
    configs = [normalize_archive_config(row.config) for row in sample]
    instances = [row.instance for row in sample]
    X = model_matrix(benchmark, configs, instances)
    predictions = predict_raw(benchmark, X, seed=0)
    costs = prediction_costs(
        predictions,
        benchmark.spec.cutoff,
        benchmark.spec.timeout_cost,
    )
    rows: list[dict[str, Any]] = []
    for record, prediction, predicted_cost in zip(sample, predictions, costs):
        actual_cost = archived_cost(record, benchmark.spec.timeout_cost)
        rows.append(
            {
                "record_id": record.record_id,
                "source": record.source,
                "line_number": record.line_number,
                "stratum": performance_stratum(record),
                "configuration_fingerprint": record.fingerprint,
                "instance": record.instance,
                "solver_seed": record.solver_seed,
                "actual_status": record.status,
                "actual_runtime": record.runtime,
                "actual_par10": actual_cost,
                "predicted_median_runtime": prediction,
                "predicted_status": (
                    "CUTOFF"
                    if prediction > benchmark.spec.cutoff
                    else "TRUE"
                ),
                "predicted_median_par10": predicted_cost,
                "absolute_error": abs(predicted_cost - actual_cost),
                "absolute_log10_error": abs(
                    math.log10(predicted_cost) - math.log10(actual_cost)
                ),
            }
        )
    return pd.DataFrame(rows)


def select_repeated_pairs(
    pair_counts: Counter[tuple[str, str]],
    count: int,
    random_seed: int,
) -> list[tuple[str, str]]:
    repeated = [key for key, repeats in pair_counts.items() if repeats >= 2]
    if len(repeated) < count:
        raise RuntimeError(
            f"Requested {count} repeated pairs, only {len(repeated)} exist."
        )
    rng = random.Random(random_seed)
    rng.shuffle(repeated)
    return repeated[:count]


def collect_selected_observations(
    full_fingerprints: set[str],
    repeated_pairs: set[tuple[str, str]],
    timeout_cost: float,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], list[float]],
    Counter[str],
]:
    full_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    repeated_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    full_record_counts: Counter[str] = Counter()
    for record in iter_archive_records():
        key = (record.fingerprint, record.instance)
        cost = archived_cost(record, timeout_cost)
        if record.fingerprint in full_fingerprints:
            full_values[key].append(cost)
            full_record_counts[record.fingerprint] += 1
        if key in repeated_pairs:
            repeated_values[key].append(cost)
    return full_values, repeated_values, full_record_counts


def source_family(names: Iterable[str]) -> str:
    families = sorted({name.rstrip("0123456789") for name in names})
    return "+".join(families)


def evaluate_full_configurations(
    benchmark: ACLibSurrogateBenchmark,
    full_fingerprints: Sequence[str],
    configs: Mapping[str, Mapping[str, Any]],
    sources: Mapping[str, set[str]],
    observations: Mapping[tuple[str, str], list[float]],
    record_counts: Counter[str],
) -> pd.DataFrame:
    training_instances = load_benchmark_data(benchmark.spec).training_instances
    output: list[dict[str, Any]] = []
    for index, fingerprint in enumerate(sorted(full_fingerprints)):
        config = normalize_archive_config(configs[fingerprint])
        actual_instance_means = np.asarray(
            [
                np.mean(observations[(fingerprint, instance)])
                for instance in training_instances
            ],
            dtype=float,
        )
        if len(actual_instance_means) != 484:
            raise RuntimeError("Incomplete supposedly full configuration.")
        X = model_matrix(
            benchmark,
            [config] * len(training_instances),
            list(training_instances),
        )
        raw = predict_raw(benchmark, X, seed=0)
        predicted = prediction_costs(
            raw,
            benchmark.spec.cutoff,
            benchmark.spec.timeout_cost,
        )
        output.append(
            {
                "full_configuration_index": index,
                "configuration_fingerprint": fingerprint,
                "source_family": source_family(sources[fingerprint]),
                "source_files": ",".join(sorted(sources[fingerprint])),
                "archived_run_records": record_counts[fingerprint],
                "training_instances": len(training_instances),
                "actual_mean_par10": float(np.mean(actual_instance_means)),
                "predicted_median_mean_par10": float(np.mean(predicted)),
                "absolute_error": float(
                    abs(np.mean(predicted) - np.mean(actual_instance_means))
                ),
                "actual_instance_mean_timeout_fraction": float(
                    np.mean(actual_instance_means == benchmark.spec.timeout_cost)
                ),
                "predicted_timeout_fraction": float(
                    np.mean(predicted == benchmark.spec.timeout_cost)
                ),
            }
        )
    return pd.DataFrame(output)


def evaluate_uncertainty(
    benchmark: ACLibSurrogateBenchmark,
    selected_pairs: Sequence[tuple[str, str]],
    configs: Mapping[str, Mapping[str, Any]],
    observations: Mapping[tuple[str, str], list[float]],
    draws: int,
    random_seed: int,
) -> pd.DataFrame:
    normalized_configs = [
        normalize_archive_config(configs[fingerprint])
        for fingerprint, _ in selected_pairs
    ]
    instances = [instance for _, instance in selected_pairs]
    X = model_matrix(benchmark, normalized_configs, instances)
    rng = np.random.RandomState(random_seed)
    quantile_seeds = rng.randint(1, 2**31 - 1, size=draws)
    predicted_draws = np.empty((draws, len(selected_pairs)), dtype=float)
    for index, seed in enumerate(quantile_seeds):
        raw = predict_raw(benchmark, X, seed=int(seed))
        predicted_draws[index] = prediction_costs(
            raw,
            benchmark.spec.cutoff,
            benchmark.spec.timeout_cost,
        )

    output: list[dict[str, Any]] = []
    for pair_index, ((fingerprint, instance), predicted) in enumerate(
        zip(selected_pairs, predicted_draws.T)
    ):
        actual = np.asarray(observations[(fingerprint, instance)], dtype=float)
        q05, q25, q50, q75, q95 = np.quantile(
            predicted, [0.05, 0.25, 0.50, 0.75, 0.95]
        )
        output.append(
            {
                "pair_index": pair_index,
                "configuration_fingerprint": fingerprint,
                "instance": instance,
                "real_repetitions": len(actual),
                "surrogate_draws": draws,
                "actual_mean_par10": float(np.mean(actual)),
                "actual_std_par10": float(np.std(actual)),
                "actual_variance_par10": float(np.var(actual)),
                "actual_timeout_fraction": float(
                    np.mean(actual == benchmark.spec.timeout_cost)
                ),
                "predicted_mean_par10": float(np.mean(predicted)),
                "predicted_std_par10": float(np.std(predicted)),
                "predicted_variance_par10": float(np.var(predicted)),
                "predicted_timeout_fraction": float(
                    np.mean(predicted == benchmark.spec.timeout_cost)
                ),
                "predicted_q05": float(q05),
                "predicted_q25": float(q25),
                "predicted_q50": float(q50),
                "predicted_q75": float(q75),
                "predicted_q95": float(q95),
                "empirical_50_interval_coverage": float(
                    np.mean((actual >= q25) & (actual <= q75))
                ),
                "empirical_90_interval_coverage": float(
                    np.mean((actual >= q05) & (actual <= q95))
                ),
            }
        )
    return pd.DataFrame(output)


def _rank_correlation(
    first: pd.Series | np.ndarray,
    second: pd.Series | np.ndarray,
) -> float:
    return float(pd.Series(first).corr(pd.Series(second), method="spearman"))


def summarize(
    asset_checks: Mapping[str, Any],
    parity: pd.DataFrame,
    cli_parity: pd.DataFrame | None,
    points: pd.DataFrame,
    full: pd.DataFrame,
    uncertainty: pd.DataFrame,
    archive_status_counts: Counter[str],
    coverage: Mapping[str, set[str]],
    pair_counts: Counter[tuple[str, str]],
) -> dict[str, Any]:
    actual_timeout = ~points["actual_status"].isin(SOLVED_STATUSES)
    predicted_timeout = points["predicted_status"] == "CUTOFF"
    true_positive = int(np.sum(actual_timeout & predicted_timeout))
    false_positive = int(np.sum(~actual_timeout & predicted_timeout))
    false_negative = int(np.sum(actual_timeout & ~predicted_timeout))
    true_negative = int(np.sum(~actual_timeout & ~predicted_timeout))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else float("nan")
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    return {
        "validation_version": VALIDATION_VERSION,
        "assets": dict(asset_checks),
        "archive": {
            "data_train_records": int(sum(archive_status_counts.values())),
            "status_counts": dict(archive_status_counts),
            "unique_configurations": len(coverage),
            "unique_configuration_instance_pairs": len(pair_counts),
            "repeated_configuration_instance_pairs": int(
                sum(count >= 2 for count in pair_counts.values())
            ),
            "fully_covered_configurations": int(
                sum(len(instances) == 484 for instances in coverage.values())
            ),
        },
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
        "command_line_wrapper": {
            "tested": cli_parity is not None,
            "tested_points": 0 if cli_parity is None else len(cli_parity),
            "all_passed": (
                None if cli_parity is None else bool(cli_parity["passed"].all())
            ),
            "maximum_cost_absolute_error": (
                None
                if cli_parity is None
                else float(cli_parity["cost_absolute_error"].max())
            ),
        },
        "point_level": {
            "sampled_records": len(points),
            "mean_absolute_par10_error": float(points["absolute_error"].mean()),
            "median_absolute_par10_error": float(
                points["absolute_error"].median()
            ),
            "root_mean_squared_par10_error": float(
                np.sqrt(np.mean(np.square(points["absolute_error"])))
            ),
            "mean_absolute_log10_error": float(
                points["absolute_log10_error"].mean()
            ),
            "spearman_par10": _rank_correlation(
                points["actual_par10"],
                points["predicted_median_par10"],
            ),
            "actual_timeout_fraction": float(actual_timeout.mean()),
            "predicted_timeout_fraction": float(predicted_timeout.mean()),
            "timeout_confusion": {
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": precision,
                "recall": recall,
            },
        },
        "full_training": {
            "configurations": len(full),
            "mean_absolute_error": float(full["absolute_error"].mean()),
            "median_absolute_error": float(full["absolute_error"].median()),
            "spearman_mean_par10": _rank_correlation(
                full["actual_mean_par10"],
                full["predicted_median_mean_par10"],
            ),
        },
        "uncertainty": {
            "repeated_pairs": len(uncertainty),
            "mean_real_repetitions": float(
                uncertainty["real_repetitions"].mean()
            ),
            "mean_empirical_50_interval_coverage": float(
                uncertainty["empirical_50_interval_coverage"].mean()
            ),
            "mean_empirical_90_interval_coverage": float(
                uncertainty["empirical_90_interval_coverage"].mean()
            ),
            "spearman_standard_deviation": _rank_correlation(
                uncertainty["actual_std_par10"],
                uncertainty["predicted_std_par10"],
            ),
        },
        "interpretation": {
            "wrapper_call_path_verified": bool(parity["passed"].all()),
            "statistical_results_are_training_data_diagnostics": True,
            "new_random_configuration_generalization_tested": False,
        },
    }


def run(
    *,
    point_samples: int,
    parity_samples: int,
    uncertainty_pairs: int,
    uncertainty_draws: int,
    random_seed: int,
    output_directory: Path,
    overwrite: bool,
    skip_cli_wrapper: bool,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    if summary_path.is_file() and not overwrite:
        print(f"Existing validation found: {summary_path}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.time()
    spec = get_benchmark_spec("clasp_queens")
    benchmark = ACLibSurrogateBenchmark(spec)

    print("Checking assets and preprocessing ...")
    assets = run_asset_checks(benchmark)

    print("Indexing archived real Clasp measurements ...")
    (
        point_sample,
        coverage,
        configs,
        sources,
        pair_counts,
        status_counts,
    ) = collect_archive_index(point_samples, random_seed)
    full_fingerprints = {
        fingerprint
        for fingerprint, instances in coverage.items()
        if len(instances) == spec.expected_training_instances
    }
    repeated_pairs = select_repeated_pairs(
        pair_counts,
        uncertainty_pairs,
        random_seed + 1,
    )

    print(f"Checking official wrapper-core parity on {parity_samples} points ...")
    parity = run_wrapper_parity(
        benchmark,
        point_sample,
        parity_samples,
    )
    _atomic_csv(output_directory / "wrapper_parity.csv", parity)

    cli_parity: pd.DataFrame | None = None
    if not skip_cli_wrapper:
        print("Checking the end-to-end command-line wrapper on five points ...")
        cli_parity = run_command_line_wrapper_parity(benchmark, point_sample)
        _atomic_csv(
            output_directory / "command_line_wrapper_parity.csv",
            cli_parity,
        )

    print(f"Predicting {len(point_sample)} stratified archived records ...")
    points = evaluate_point_sample(benchmark, point_sample)
    _atomic_csv(output_directory / "point_level.csv", points)

    print(
        f"Collecting real observations for {len(full_fingerprints)} complete "
        f"configurations and {len(repeated_pairs)} repeated pairs ..."
    )
    full_observations, repeated_observations, full_record_counts = (
        collect_selected_observations(
            full_fingerprints,
            set(repeated_pairs),
            spec.timeout_cost,
        )
    )

    print("Comparing full-training mean PAR10 values ...")
    full = evaluate_full_configurations(
        benchmark,
        sorted(full_fingerprints),
        configs,
        sources,
        full_observations,
        full_record_counts,
    )
    _atomic_csv(output_directory / "full_training.csv", full)

    print(
        f"Calibrating uncertainty with {len(repeated_pairs)} repeated pairs "
        f"and {uncertainty_draws} surrogate draws ..."
    )
    uncertainty = evaluate_uncertainty(
        benchmark,
        repeated_pairs,
        configs,
        repeated_observations,
        uncertainty_draws,
        random_seed + 2,
    )
    _atomic_csv(output_directory / "uncertainty.csv", uncertainty)

    summary = summarize(
        assets,
        parity,
        cli_parity,
        points,
        full,
        uncertainty,
        status_counts,
        coverage,
        pair_counts,
    )
    summary["run"] = {
        "point_samples": point_samples,
        "parity_samples": parity_samples,
        "uncertainty_pairs": uncertainty_pairs,
        "uncertainty_draws": uncertainty_draws,
        "random_seed": random_seed,
        "elapsed_seconds": time.time() - started,
        "output_directory": str(output_directory),
        "skip_cli_wrapper": skip_cli_wrapper,
    }
    _atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-samples", type=int, default=10_000)
    parser.add_argument("--parity-samples", type=int, default=100)
    parser.add_argument("--uncertainty-pairs", type=int, default=500)
    parser.add_argument("--uncertainty-draws", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output-directory", type=Path, default=RESULTS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-cli-wrapper",
        action="store_true",
        help="Skip the five-point legacy CLI/local-server transport check.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.point_samples = min(args.point_samples, 100)
        args.parity_samples = min(args.parity_samples, 10)
        args.uncertainty_pairs = min(args.uncertainty_pairs, 20)
        args.uncertainty_draws = min(args.uncertainty_draws, 10)
        if args.output_directory == RESULTS:
            args.output_directory = HERE / "results_smoke"
    if args.parity_samples > args.point_samples:
        parser.error("--parity-samples cannot exceed --point-samples")
    for name in (
        "point_samples",
        "parity_samples",
        "uncertainty_pairs",
        "uncertainty_draws",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    run(
        point_samples=args.point_samples,
        parity_samples=args.parity_samples,
        uncertainty_pairs=args.uncertainty_pairs,
        uncertainty_draws=args.uncertainty_draws,
        random_seed=args.random_seed,
        output_directory=args.output_directory.resolve(),
        overwrite=args.overwrite,
        skip_cli_wrapper=args.skip_cli_wrapper,
    )


if __name__ == "__main__":
    main()

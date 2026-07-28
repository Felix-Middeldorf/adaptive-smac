from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

from smac import AlgorithmConfigurationFacade, Scenario

from aclib_benchmark import (
    ACLIB_ROOT,
    CUTOFF,
    HERE,
    TIMEOUT_COST,
    CplexRegions200Benchmark,
    load_benchmark_data,
    required_assets,
)


DEFAULT_N_TRIALS = 5_000
DEFAULT_N_INSTANCES = 1_000
DEFAULT_OUTPUT_ROOT = HERE / "results"
EXPERIMENT_VERSION = 1


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def scenario_name(n_instances: int, n_trials: int) -> str:
    return f"cplex_regions200_train{n_instances}_trials{n_trials}"


def run_directory(
    output_root: Path,
    smac_seed: int,
    n_instances: int,
    n_trials: int,
) -> Path:
    return output_root / scenario_name(n_instances, n_trials) / str(smac_seed)


def _completion_is_valid(path: Path, expected: dict[str, Any]) -> bool:
    completion_file = path / "completed.json"
    required = (
        path / "scenario.json",
        path / "runhistory.json",
        path / "intensifier.json",
        path / "optimization.json",
        path / "incumbent.json",
        path / "trajectory.json",
        path / "summary.json",
    )
    if not completion_file.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        completed = json.loads(completion_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return completed == expected


def _smac_state_exists(path: Path) -> bool:
    return all(
        (path / filename).is_file()
        for filename in (
            "scenario.json",
            "configspace.json",
            "runhistory.json",
            "intensifier.json",
            "optimization.json",
        )
    )


def _acquire_run_lock(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "owner metadata unavailable"
        handle.close()
        raise RuntimeError(
            f"Another process is already running this SMAC output: {path}. "
            f"Lock owner: {owner}"
        ) from error
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": time.time(),
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    return handle


def _asset_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for path in required_assets():
        stat = path.stat()
        metadata[str(path.relative_to(ACLIB_ROOT))] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return metadata


def _serialize_trajectory(smac: AlgorithmConfigurationFacade) -> list[dict[str, Any]]:
    return [
        {
            "config_ids": [int(config_id) for config_id in item.config_ids],
            "costs": [float(cost) for cost in item.costs],
            "trial": int(item.trial),
            "walltime": float(item.walltime),
        }
        for item in smac.intensifier.trajectory
    ]


def _validate_on_test_instances(
    benchmark: CplexRegions200Benchmark,
    incumbent: Any,
    test_instances: tuple[str, ...],
    output_file: Path,
) -> dict[str, Any]:
    evaluations = []
    for instance in test_instances:
        cost, info = benchmark.evaluate(incumbent, instance)
        evaluations.append(
            {
                "instance": instance,
                "cost": float(cost),
                "prediction": float(info["surrogate_prediction"]),
                "status": str(info["surrogate_status"]),
            }
        )
    costs = [row["cost"] for row in evaluations]
    result = {
        "n_instances": len(evaluations),
        "mean_par10": float(statistics.fmean(costs)),
        "median_par10": float(statistics.median(costs)),
        "min_par10": float(min(costs)),
        "max_par10": float(max(costs)),
        "timeouts": sum(row["status"].upper() == "CUTOFF" for row in evaluations),
        "evaluations": evaluations,
    }
    _atomic_json(output_file, result)
    return {key: value for key, value in result.items() if key != "evaluations"}


def run_experiment(
    *,
    smac_seed: int,
    n_trials: int = DEFAULT_N_TRIALS,
    n_instances: int = DEFAULT_N_INSTANCES,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    validate_test: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if not 1 <= n_instances <= DEFAULT_N_INSTANCES:
        raise ValueError(f"n_instances must be between 1 and {DEFAULT_N_INSTANCES}.")

    output_root = Path(output_root).resolve()
    output_path = run_directory(output_root, smac_seed, n_instances, n_trials)
    # Keep this file handle alive for the entire function. flock is released
    # automatically on close or process termination, including Slurm kills.
    run_lock = _acquire_run_lock(output_path)
    run_identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "smac_seed": int(smac_seed),
        "n_trials": int(n_trials),
        "n_training_instances": int(n_instances),
        "validated_on_test_instances": bool(validate_test),
    }
    completion = {
        "state": "complete",
        **run_identity,
        "asset_signature": _asset_metadata(),
    }
    if not overwrite and _completion_is_valid(output_path, completion):
        print(f"Complete run found; skipping {output_path}")
        summary = json.loads((output_path / "summary.json").read_text(encoding="utf-8"))
        run_lock.close()
        return summary

    # Invalidate any older completion marker before loading/continuing SMAC.
    # If this process is interrupted, a subsequent invocation will therefore
    # resume instead of mistaking the older result for the requested run.
    _atomic_json(output_path / "completed.json", {**completion, "state": "running"})

    started = time.time()
    data = load_benchmark_data()
    training_instances = data.training_instances[:n_instances]
    training_features = {
        instance: data.features[instance]
        for instance in training_instances
    }

    scenario = Scenario(
        configspace=data.configspace,
        name=scenario_name(n_instances, n_trials),
        output_directory=output_root,
        deterministic=True,
        objectives="PAR10",
        crash_cost=TIMEOUT_COST,
        n_trials=n_trials,
        use_default_config=True,
        instances=list(training_instances),
        instance_features=training_features,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != output_path:
        raise RuntimeError(
            f"Unexpected SMAC output path {scenario.output_directory}; expected {output_path}."
        )

    run_metadata = {
        **run_identity,
        "output_directory": str(output_path),
        "training_instances": list(training_instances),
        "test_instances_reserved": len(data.test_instances),
        "cutoff": CUTOFF,
        "timeout_cost": TIMEOUT_COST,
        "deterministic_quantile_seed": 0,
        "adaptive_capping": False,
        "facade": "AlgorithmConfigurationFacade",
        "assets": completion["asset_signature"],
        "versions": {
            "python": __import__("sys").version,
            "smac": _package_version("smac"),
            "ConfigSpace": _package_version("ConfigSpace"),
            "epm": _package_version("epm"),
            "pyrfr": _package_version("pyrfr"),
            "numpy": _package_version("numpy"),
        },
    }
    _atomic_json(output_path / "run_metadata.json", run_metadata)

    print(f"Loading cplex_regions200 surrogate for SMAC seed {smac_seed} ...")
    benchmark = CplexRegions200Benchmark()
    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=benchmark.target,
        overwrite=overwrite,
    )
    if not overwrite and _smac_state_exists(output_path):
        previous_scenario = Scenario.load(output_path)
        if scenario != previous_scenario:
            raise RuntimeError(
                "Existing SMAC state has different scenario/component metadata. "
                "Batch jobs cannot answer SMAC's interactive mismatch prompt. "
                "Use a new --output-root or rerun run_smac.py once with --overwrite."
            )
    incumbent = smac.optimize()

    incumbent_cost = float(smac.runhistory.average_cost(incumbent, normalize=False))
    incumbent_payload = {
        "config_id": int(smac.runhistory.get_config_id(incumbent)),
        "configuration": dict(incumbent),
        "training_cost_on_evaluated_instances": incumbent_cost,
    }
    _atomic_json(output_path / "incumbent.json", incumbent_payload)
    _atomic_json(output_path / "trajectory.json", _serialize_trajectory(smac))

    test_validation = None
    if validate_test:
        print(f"Validating incumbent on {len(data.test_instances)} held-out instances ...")
        test_validation = _validate_on_test_instances(
            benchmark,
            incumbent,
            data.test_instances,
            output_path / "test_validation.json",
        )

    summary = {
        **run_identity,
        "output_directory": str(output_path),
        "model_type": benchmark.model_type,
        "finished_trials": int(smac.runhistory.finished),
        "submitted_trials": int(smac.runhistory.submitted),
        "incumbent_cost": incumbent_cost,
        "target_evaluations_this_process": int(benchmark.evaluation_count),
        "target_timeouts_this_process": int(benchmark.timeout_count),
        "walltime_seconds_this_process": float(time.time() - started),
        "test_validation": test_validation,
    }
    _atomic_json(output_path / "summary.json", summary)
    _atomic_json(output_path / "completed.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True))
    run_lock.close()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run modern SMAC on the ACLib cplex_regions200 surrogate."
    )
    parser.add_argument("--smac-seed", type=int, default=0)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-instances", type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validate-test",
        action="store_true",
        help="Evaluate the final incumbent on all 1000 held-out test instances.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start this run from scratch instead of resuming matching SMAC state.",
    )
    args = parser.parse_args()
    run_experiment(
        smac_seed=args.smac_seed,
        n_trials=args.n_trials,
        n_instances=args.n_instances,
        output_root=args.output_root,
        validate_test=args.validate_test,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

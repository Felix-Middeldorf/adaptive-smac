#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import closing
from pathlib import Path
from typing import Any

# Make the shared ACLib adapter importable regardless of the current working
# directory. This keeps 01_initial_c directly executable without PYTHONPATH.
HERE = Path(__file__).resolve().parent
PARENT_DIRECTORY = HERE.parent
if str(PARENT_DIRECTORY) not in sys.path:
    sys.path.append(str(PARENT_DIRECTORY))

from smac import AlgorithmConfigurationFacade, Scenario

from aclib_benchmark import (
    CUTOFF,
    TIMEOUT_COST,
    CplexRegions200Benchmark,
    load_benchmark_data,
)
from run_smac import (
    _acquire_run_lock,
    _asset_metadata,
    _atomic_json,
    _completion_is_valid,
    _package_version,
    _serialize_trajectory,
    _smac_state_exists,
)


DEPTHS = (5, 10, 20, 30)
SMAC_SEEDS = tuple(range(5))
N_INSTANCES = 100
N_TRIALS = 1_000
OUTPUT_ROOT = HERE / "results"
EXPERIMENT_VERSION = 1


def run_directory(output_root: Path, depth: int, smac_seed: int) -> Path:
    return output_root / f"depth_{depth}" / str(smac_seed)


def _run_locked(
    *,
    depth: int,
    smac_seed: int,
    n_trials: int,
    output_root: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    run_identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "policy": f"fixed_depth_{depth}",
        "depth": depth,
        "smac_seed": smac_seed,
        "n_trials": n_trials,
        "n_training_instances": N_INSTANCES,
    }
    completion = {
        "state": "complete",
        **run_identity,
        "asset_signature": _asset_metadata(),
    }
    if not overwrite and _completion_is_valid(output_path, completion):
        print(f"Complete run found; skipping {output_path}")
        return json.loads((output_path / "summary.json").read_text(encoding="utf-8"))

    _atomic_json(output_path / "completed.json", {**completion, "state": "running"})
    started = time.time()

    data = load_benchmark_data()
    training_instances = data.training_instances[:N_INSTANCES]
    training_features = {
        instance: data.features[instance]
        for instance in training_instances
    }
    scenario = Scenario(
        configspace=data.configspace,
        name=f"depth_{depth}",
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

    _atomic_json(
        output_path / "run_metadata.json",
        {
            **run_identity,
            "output_directory": str(output_path),
            "training_instances": list(training_instances),
            "cutoff": CUTOFF,
            "timeout_cost": TIMEOUT_COST,
            "deterministic_quantile_seed": 0,
            "facade": "AlgorithmConfigurationFacade",
            "smac_surrogate": {
                "max_depth": depth,
                "all_other_parameters": "AlgorithmConfigurationFacade defaults",
            },
            "assets": completion["asset_signature"],
            "versions": {
                "smac": _package_version("smac"),
                "ConfigSpace": _package_version("ConfigSpace"),
                "epm": _package_version("epm"),
                "pyrfr": _package_version("pyrfr"),
                "numpy": _package_version("numpy"),
            },
        },
    )

    print(f"Loading ACLib model for depth={depth}, SMAC seed={smac_seed} ...")
    benchmark = CplexRegions200Benchmark()
    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        max_depth=depth,
        pca_components=None
    )
    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=benchmark.target,
        model=model,
        overwrite=True,
    )
    if not overwrite and _smac_state_exists(output_path):
        previous_scenario = Scenario.load(output_path)
        if scenario != previous_scenario:
            raise RuntimeError(
                "Existing SMAC state is incompatible with this run. Use a new "
                "output root or run once with --overwrite."
            )

    incumbent = smac.optimize()
    incumbent_cost = float(smac.runhistory.average_cost(incumbent, normalize=False))
    _atomic_json(
        output_path / "incumbent.json",
        {
            "config_id": int(smac.runhistory.get_config_id(incumbent)),
            "configuration": dict(incumbent),
            "training_cost_on_evaluated_instances": incumbent_cost,
        },
    )
    _atomic_json(output_path / "trajectory.json", _serialize_trajectory(smac))

    summary = {
        **run_identity,
        "output_directory": str(output_path),
        "model_type": benchmark.model_type,
        "finished_trials": int(smac.runhistory.finished),
        "incumbent_cost": incumbent_cost,
        "target_evaluations_this_process": int(benchmark.evaluation_count),
        "target_timeouts_this_process": int(benchmark.timeout_count),
        "walltime_seconds_this_process": float(time.time() - started),
    }
    _atomic_json(output_path / "summary.json", summary)
    _atomic_json(output_path / "completed.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_fixed_depth(
    *,
    depth: int,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    output_root: Path = OUTPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}.")
    if smac_seed < 0:
        raise ValueError("smac_seed must be non-negative.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")

    output_root = Path(output_root).resolve()
    output_path = run_directory(output_root, depth, smac_seed)
    with closing(_acquire_run_lock(output_path)):
        return _run_locked(
            depth=depth,
            smac_seed=smac_seed,
            n_trials=n_trials,
            output_root=output_root,
            output_path=output_path,
            overwrite=overwrite,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, choices=DEPTHS, required=True)
    parser.add_argument("--smac-seed", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_fixed_depth(
        depth=args.depth,
        smac_seed=args.smac_seed,
        n_trials=args.n_trials,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Compare the original and batched get_x_best implementations."""

from __future__ import annotations

import argparse
import functools
import importlib.metadata
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"
PARENT_DIRECTORY = HERE.parent

# Direct invocations must use the local checkout too. The Slurm launcher also
# puts this path first in PYTHONPATH before the worker imports this module.
sys.path.insert(0, str(LOCAL_SMAC_ROOT))
if str(PARENT_DIRECTORY) not in sys.path:
    sys.path.append(str(PARENT_DIRECTORY))

import smac
from smac import AlgorithmConfigurationFacade, Scenario
from smac.main.config_selector import ConfigSelector

from aclib_benchmark import (
    TIMEOUT_COST,
    CplexRegions200Benchmark,
    load_benchmark_data,
)


MODES = ("original_singleton", "fixed_batched")
SMAC_SEEDS = (0, 1)
DEPTH = 20
N_INSTANCES = 150
N_TRIALS = 3_000
RETRAIN_AFTER = 8
OUTPUT_ROOT = HERE / "results"
EXPERIMENT_VERSION = 1


def _assert_local_smac() -> Path:
    source = Path(inspect.getfile(ConfigSelector)).resolve()
    if LOCAL_SMAC_ROOT.resolve() not in source.parents:
        raise RuntimeError(
            f"Expected ConfigSelector from {LOCAL_SMAC_ROOT}, got {source}."
        )
    return source


LOCAL_SELECTOR_SOURCE = _assert_local_smac()


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
    temporary.replace(path)


class OriginalSingletonConfigSelector(ConfigSelector):
    """Reproduce SMAC 2.4.0's original row-by-row implementation."""

    def _get_x_best(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        if self._predict_x_best:
            model = self._model
            assert model is not None
            costs = [
                (
                    model.predict_marginalized(
                        x.reshape((1, -1))
                    )[0][0][0],
                    x,
                )
                for x in X
            ]
            costs.sort(key=lambda item: item[0])
            return costs[0][1], float(costs[0][0])

        raise RuntimeError("Observed-incumbent selection is not implemented.")


class RuntimeProfiler:
    """Collect the timings needed for the get_x_best comparison."""

    def __init__(self, smac_facade: AlgorithmConfigurationFacade):
        self.smac = smac_facade
        self.events: list[dict[str, Any]] = []
        self.inside_x_best = False

    def _state(self) -> tuple[int, int]:
        runhistory = self.smac.runhistory
        return int(runhistory.finished), len(runhistory._config_ids)

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        n_inputs: int | None = None,
    ) -> None:
        trials, configurations = self._state()
        event: dict[str, Any] = {
            "phase": phase,
            "seconds": float(seconds),
            "finished_trials": trials,
            "configurations": configurations,
        }
        if n_inputs is not None:
            event["n_inputs"] = int(n_inputs)
            event["expanded_inputs"] = int(n_inputs * N_INSTANCES)
        self.events.append(event)

    def wrap_x_best(self, selector: ConfigSelector) -> None:
        original = selector._get_x_best

        @functools.wraps(original)
        def measured(X: np.ndarray) -> tuple[np.ndarray, float]:
            started = time.perf_counter()
            self.inside_x_best = True
            try:
                return original(X)
            finally:
                self.inside_x_best = False
                self.record(
                    "get_x_best",
                    time.perf_counter() - started,
                    n_inputs=len(X),
                )

        selector._get_x_best = measured  # type: ignore[method-assign]

    def wrap_model_method(
        self,
        model: Any,
        method_name: str,
        phase: str,
    ) -> None:
        original = getattr(model, method_name)

        @functools.wraps(original)
        def measured(X: np.ndarray, *args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(X, *args, **kwargs)
            finally:
                measured_phase = phase
                if phase == "predict_marginalized" and self.inside_x_best:
                    measured_phase = "get_x_best_predict_marginalized"
                self.record(
                    measured_phase,
                    time.perf_counter() - started,
                    n_inputs=len(X),
                )

        setattr(model, method_name, measured)

    def summary(self) -> dict[str, dict[str, float | int]]:
        phases = sorted({event["phase"] for event in self.events})
        result: dict[str, dict[str, float | int]] = {}
        for phase in phases:
            events = [
                event for event in self.events if event["phase"] == phase
            ]
            seconds = [float(event["seconds"]) for event in events]
            result[phase] = {
                "calls": len(events),
                "total_seconds_inclusive": float(sum(seconds)),
                "mean_seconds": float(sum(seconds) / len(seconds)),
                "max_seconds": float(max(seconds)),
                "total_inputs": sum(
                    int(event.get("n_inputs", 0)) for event in events
                ),
                "total_expanded_inputs": sum(
                    int(event.get("expanded_inputs", 0))
                    for event in events
                ),
            }
        return result


def output_directory(mode: str, smac_seed: int) -> Path:
    return OUTPUT_ROOT / mode / str(smac_seed)


def _checkpoints(runhistory: Any) -> dict[str, dict[str, float | int]]:
    trials = sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )
    if not trials:
        return {}

    first_start = float(trials[0][1].starttime)
    result = {}
    for checkpoint in range(500, len(trials) + 1, 500):
        prefix = trials[:checkpoint]
        result[str(checkpoint)] = {
            "elapsed_seconds": (
                float(prefix[-1][1].endtime) - first_start
            ),
            "configurations": len(
                {trial_key.config_id for trial_key, _ in prefix}
            ),
        }
    return result


def run_comparison(
    mode: str,
    smac_seed: int,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")

    destination = output_directory(mode, smac_seed)
    summary_path = destination / "runtime_summary.json"
    if summary_path.is_file() and not overwrite:
        previous = json.loads(summary_path.read_text())
        if (
            previous.get("error") is None
            and previous.get("finished_trials") == N_TRIALS
            and previous.get("experiment_version") == EXPERIMENT_VERSION
        ):
            print(f"Complete comparison found; skipping {destination}")
            return previous

    data = load_benchmark_data()
    training_instances = data.training_instances[:N_INSTANCES]
    training_features = {
        instance: data.features[instance]
        for instance in training_instances
    }
    scenario = Scenario(
        configspace=data.configspace,
        name=mode,
        output_directory=OUTPUT_ROOT,
        deterministic=True,
        objectives="PAR10",
        crash_cost=TIMEOUT_COST,
        n_trials=N_TRIALS,
        use_default_config=True,
        instances=list(training_instances),
        instance_features=training_features,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != destination:
        raise RuntimeError(
            f"Unexpected output directory {scenario.output_directory}; "
            f"expected {destination}."
        )

    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        max_depth=DEPTH,
        pca_components=None,
    )
    if mode == "original_singleton":
        config_selector: ConfigSelector = OriginalSingletonConfigSelector(
            scenario,
            retrain_after=RETRAIN_AFTER,
        )
    else:
        config_selector = AlgorithmConfigurationFacade.get_config_selector(
            scenario,
            retrain_after=RETRAIN_AFTER,
        )

    benchmark = CplexRegions200Benchmark()
    smac_facade = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=benchmark.target,
        model=model,
        config_selector=config_selector,
        overwrite=True,
    )

    selector = smac_facade.optimizer._intensifier._config_selector
    profiler = RuntimeProfiler(smac_facade)
    profiler.wrap_model_method(model, "predict_marginalized", "predict_marginalized")
    profiler.wrap_model_method(model, "train", "model_train")
    profiler.wrap_x_best(selector)

    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "mode": mode,
        "smac_seed": smac_seed,
        "depth": DEPTH,
        "n_trials": N_TRIALS,
        "n_instances": N_INSTANCES,
        "retrain_after": RETRAIN_AFTER,
        "local_smac_root": str(LOCAL_SMAC_ROOT),
        "smac_module": str(Path(smac.__file__).resolve()),
        "config_selector_source": str(LOCAL_SELECTOR_SOURCE),
        "smac_version": importlib.metadata.version("smac"),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }
    _atomic_json(destination / "comparison_metadata.json", metadata)

    started = time.perf_counter()
    incumbent = None
    error = None
    try:
        incumbent = smac_facade.optimize()
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        walltime = time.perf_counter() - started
        summary = {
            **metadata,
            "finished_trials": int(smac_facade.runhistory.finished),
            "configurations": len(smac_facade.runhistory._config_ids),
            "walltime_seconds": float(walltime),
            "target_function_walltime_seconds": float(
                smac_facade.optimizer.used_target_function_walltime
            ),
            "target_evaluations": int(benchmark.evaluation_count),
            "incumbent_cost": (
                float(
                    smac_facade.runhistory.average_cost(
                        incumbent,
                        normalize=False,
                    )
                )
                if incumbent is not None
                else None
            ),
            "error": error,
            "phase_summary": profiler.summary(),
            "checkpoint_summary": _checkpoints(smac_facade.runhistory),
            "timings_are_inclusive": True,
        }
        _atomic_json(destination / "runtime_events.json", profiler.events)
        _atomic_json(summary_path, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--smac-seed",
        type=int,
        choices=SMAC_SEEDS,
        required=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run_comparison(
        args.mode,
        args.smac_seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

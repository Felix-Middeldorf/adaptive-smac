#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Profile the sources of SMAC overhead on the ACLib CPLEX surrogate."""

from __future__ import annotations

import argparse
import cProfile
import functools
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
PARENT_DIRECTORY = HERE.parent
if str(PARENT_DIRECTORY) not in sys.path:
    sys.path.append(str(PARENT_DIRECTORY))

import joblib
from smac import AlgorithmConfigurationFacade, Scenario
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.acquisition.maximizer.random_search import RandomSearch

from aclib_benchmark import (
    TIMEOUT_COST,
    CplexRegions200Benchmark,
    load_benchmark_data,
)


DEPTH = 20
SMAC_SEEDS = (0, 1)
N_INSTANCES = 150
N_TRIALS = 1_000
OUTPUT_ROOT = HERE / "results"
EXPERIMENT_VERSION = 1


@dataclass(frozen=True)
class Variant:
    name: str
    use_instance_features: bool = True
    challengers: int = 5_000
    local_search_iterations: int = 10
    retrain_after: int = 8
    periodic_save: bool = True
    random_search_only: bool = False


VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("baseline"),
        Variant("no_instance_features", use_instance_features=False),
        Variant("challengers_500", challengers=500),
        Variant("local_search_1", local_search_iterations=1),
        Variant("random_search_only", random_search_only=True),
        Variant("retrain_after_64", retrain_after=64),
        Variant("no_periodic_save", periodic_save=False),
    )
}


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
    temporary.replace(path)


class RuntimeProfiler:
    """Collect inclusive timings without modifying the installed SMAC package."""

    def __init__(self, smac: AlgorithmConfigurationFacade, n_instances: int):
        self.smac = smac
        self.n_instances = n_instances
        self.events: list[dict[str, Any]] = []

    def _state(self) -> tuple[int, int]:
        runhistory = self.smac.runhistory
        return int(runhistory.finished), len(runhistory._config_ids)

    def record(
        self,
        phase: str,
        seconds: float,
        *,
        n_inputs: int | None = None,
        expanded_inputs: int | None = None,
    ) -> None:
        trials, configs = self._state()
        event = {
            "phase": phase,
            "seconds": float(seconds),
            "finished_trials": trials,
            "configurations": configs,
        }
        if n_inputs is not None:
            event["n_inputs"] = int(n_inputs)
        if expanded_inputs is not None:
            event["expanded_inputs"] = int(expanded_inputs)
        self.events.append(event)

    def wrap(
        self,
        obj: Any,
        method_name: str,
        phase: str,
        *,
        array_argument: int | None = None,
        expand_instances: bool = False,
    ) -> Callable[..., Any]:
        original = getattr(obj, method_name)

        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            n_inputs = None
            if array_argument is not None and len(args) > array_argument:
                try:
                    n_inputs = len(args[array_argument])
                except TypeError:
                    n_inputs = None
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                expanded = None
                if n_inputs is not None and expand_instances:
                    expanded = n_inputs * self.n_instances
                self.record(
                    phase,
                    time.perf_counter() - started,
                    n_inputs=n_inputs,
                    expanded_inputs=expanded,
                )

        setattr(obj, method_name, measured)
        return original

    def summary(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self.events:
            grouped.setdefault(event["phase"], []).append(event)

        result = {}
        for phase, events in sorted(grouped.items()):
            durations = [float(event["seconds"]) for event in events]
            result[phase] = {
                "calls": len(events),
                "total_seconds_inclusive": float(sum(durations)),
                "mean_seconds": float(statistics.fmean(durations)),
                "median_seconds": float(statistics.median(durations)),
                "max_seconds": float(max(durations)),
                "total_inputs": sum(int(event.get("n_inputs", 0)) for event in events),
                "total_expanded_inputs": sum(
                    int(event.get("expanded_inputs", 0)) for event in events
                ),
            }
        return result


def output_directory(variant: Variant, smac_seed: int) -> Path:
    return OUTPUT_ROOT / variant.name / str(smac_seed)


def run_profile(
    variant_name: str,
    smac_seed: int,
    *,
    overwrite: bool = False,
    enable_cprofile: bool = False,
) -> dict[str, Any]:
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"SMAC seed must be one of {SMAC_SEEDS}.")

    variant = VARIANTS[variant_name]
    destination = output_directory(variant, smac_seed)
    summary_path = destination / "runtime_summary.json"
    if summary_path.is_file() and not overwrite:
        previous = json.loads(summary_path.read_text())
        if (
            previous.get("error") is None
            and previous.get("finished_trials") == N_TRIALS
        ):
            print(f"Complete profile found; skipping {destination}")
            return previous

    data = load_benchmark_data()
    training_instances = data.training_instances[:N_INSTANCES]
    all_features = {
        instance: data.features[instance]
        for instance in training_instances
    }
    scenario_features = all_features if variant.use_instance_features else None

    scenario = Scenario(
        configspace=data.configspace,
        name=variant.name,
        output_directory=OUTPUT_ROOT,
        deterministic=True,
        objectives="PAR10",
        crash_cost=TIMEOUT_COST,
        n_trials=N_TRIALS,
        use_default_config=True,
        instances=list(training_instances),
        instance_features=scenario_features,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != destination:
        raise RuntimeError(
            f"Unexpected SMAC output directory {scenario.output_directory}; "
            f"expected {destination}."
        )

    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        max_depth=DEPTH,
        pca_components=None,
    )
    if variant.random_search_only:
        acquisition_maximizer = RandomSearch(
            configspace=scenario.configspace,
            seed=scenario.seed,
        )
    else:
        acquisition_maximizer = LocalAndSortedRandomSearch(
            configspace=scenario.configspace,
            challengers=variant.challengers,
            local_search_iterations=variant.local_search_iterations,
            seed=scenario.seed,
        )
    config_selector = AlgorithmConfigurationFacade.get_config_selector(
        scenario,
        retrain_after=variant.retrain_after,
    )

    benchmark = CplexRegions200Benchmark()
    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=benchmark.target,
        model=model,
        acquisition_maximizer=acquisition_maximizer,
        config_selector=config_selector,
        overwrite=True,
    )

    profiler = RuntimeProfiler(
        smac,
        n_instances=N_INSTANCES if variant.use_instance_features else 1,
    )
    selector = smac.optimizer._intensifier._config_selector
    maximizer = selector._acquisition_maximizer
    local_search = getattr(maximizer, "_local_search", None)
    random_search = getattr(maximizer, "_random_search", None)

    profiler.wrap(selector, "_collect_data", "collect_data")
    profiler.wrap(model, "train", "model_train", array_argument=0)
    profiler.wrap(
        model,
        "predict_marginalized",
        "predict_marginalized",
        array_argument=0,
        expand_instances=variant.use_instance_features,
    )
    profiler.wrap(maximizer, "_maximize", "acquisition_maximize")
    if random_search is not None:
        profiler.wrap(
            random_search,
            "_maximize",
            "random_search",
            array_argument=None,
        )
    if local_search is not None:
        profiler.wrap(local_search, "_maximize", "local_search_maximize")
        profiler.wrap(local_search, "_search", "local_search_walk")

    optimizer = smac.optimizer
    profiler.wrap(optimizer, "ask", "ask")
    profiler.wrap(optimizer, "tell", "tell")
    profiler.wrap(smac.runhistory, "save", "runhistory_save")
    profiler.wrap(optimizer._intensifier, "save", "intensifier_save")

    original_optimizer_save = optimizer.save
    if variant.periodic_save:
        profiler.wrap(optimizer, "save", "optimizer_save")
    else:
        optimizer.save = lambda: None

    run_metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "variant": asdict(variant),
        "depth": DEPTH,
        "smac_seed": smac_seed,
        "n_instances": N_INSTANCES,
        "n_trials": N_TRIALS,
        "instance_feature_dimensions": (
            len(next(iter(all_features.values())))
            if variant.use_instance_features
            else 0
        ),
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "affinity_cpus": (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "joblib_cpu_count": joblib.cpu_count(),
            "joblib_effective_n_jobs_minus_one": joblib.effective_n_jobs(-1),
            "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    _atomic_json(destination / "run_metadata.json", run_metadata)

    python_profile = cProfile.Profile() if enable_cprofile else None
    started = time.perf_counter()
    error: str | None = None
    incumbent = None
    try:
        if python_profile is not None:
            python_profile.enable()
        incumbent = smac.optimize()
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        if python_profile is not None:
            python_profile.disable()
            python_profile.dump_stats(destination / "runtime.prof")

        if not variant.periodic_save:
            optimizer.save = original_optimizer_save
            save_started = time.perf_counter()
            original_optimizer_save()
            profiler.record(
                "final_optimizer_save",
                time.perf_counter() - save_started,
            )

        walltime = time.perf_counter() - started
        result = {
            **run_metadata,
            "finished_trials": int(smac.runhistory.finished),
            "configurations": len(smac.runhistory._config_ids),
            "walltime_seconds": float(walltime),
            "target_function_walltime_seconds": float(
                optimizer.used_target_function_walltime
            ),
            "target_evaluations": int(benchmark.evaluation_count),
            "incumbent_cost": (
                float(smac.runhistory.average_cost(incumbent, normalize=False))
                if incumbent is not None
                else None
            ),
            "error": error,
            "phase_summary": profiler.summary(),
            "timings_are_inclusive": True,
        }
        _atomic_json(destination / "runtime_events.json", profiler.events)
        _atomic_json(summary_path, result)

    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--smac-seed", type=int, choices=SMAC_SEEDS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    args = parser.parse_args()
    if args.list_variants:
        for variant in VARIANTS.values():
            print(json.dumps(asdict(variant), sort_keys=True))
        return
    if args.variant is None or args.smac_seed is None:
        parser.error("--variant and --smac-seed are required unless --list-variants is used")
    run_profile(
        args.variant,
        args.smac_seed,
        overwrite=args.overwrite,
        enable_cprofile=args.cprofile,
    )


if __name__ == "__main__":
    main()

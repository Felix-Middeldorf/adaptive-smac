#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Profile SMAC runtime mechanisms consistently across ACLib surrogates."""

from __future__ import annotations

import argparse
import cProfile
import fcntl
import functools
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


HERE = Path(__file__).resolve().parent
ACLIB_EXPERIMENT_ROOT = HERE.parents[1]
REPOSITORY_ROOT = ACLIB_EXPERIMENT_ROOT.parents[1]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"
for source in (LOCAL_SMAC_ROOT, ACLIB_EXPERIMENT_ROOT):
    source_text = str(source)
    if source_text in sys.path:
        sys.path.remove(source_text)
    sys.path.insert(0, source_text)

import joblib
import smac
from smac import AlgorithmConfigurationFacade, Scenario
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.initial_design.random_design import RandomInitialDesign
from smac.model.random_forest.random_forest import RandomForest

import smac.acquisition.maximizer.local_search as local_search_module

from fixed_depth_experiment import (
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_TREES,
    PCA_COMPONENTS,
    RANDOM_DESIGN_PROBABILITY,
    load_initial_choice,
    resolve_initial_configuration,
)
from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    BENCHMARKS,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)
from surrogate_telemetry import SurrogateTelemetryCallback, TelemetryEI


N_TRIALS = 256
DEFAULT_DEPTH = 10
DEFAULT_CHALLENGERS = 5_000
DEFAULT_LOCAL_SEARCH_ITERATIONS = 10
DEFAULT_RETRAIN_AFTER = 8
EXPERIMENT_VERSION = 1
OUTPUT_ROOT = HERE / "results"

MAIN_EXPERIMENTS = {
    "clasp_queens": ACLIB_EXPERIMENT_ROOT
    / "clasp_queens_surrogate"
    / "01_initial_cq",
    "clasp_weighted": ACLIB_EXPERIMENT_ROOT
    / "clasp_weighted-sequence_surrogate"
    / "01_initial_cw",
    "cplex_rcw": ACLIB_EXPERIMENT_ROOT
    / "cplex_rcw_surrogate"
    / "01_initial_cr",
    "lingeling_circuitfuzz": ACLIB_EXPERIMENT_ROOT
    / "lingeling_circuitfuzz_surrogate"
    / "01_initial_lc",
    "lpg_zenotravel": ACLIB_EXPERIMENT_ROOT
    / "lpg_zenotravel_surrogate"
    / "01_initial_lz",
}


@dataclass(frozen=True)
class Variant:
    name: str
    telemetry: bool = True
    use_instance_features: bool = True
    challengers: int = DEFAULT_CHALLENGERS
    local_search_iterations: int = DEFAULT_LOCAL_SEARCH_ITERATIONS
    retrain_after: int = DEFAULT_RETRAIN_AFTER
    periodic_save: bool = True
    random_search_only: bool = False


VARIANTS = {
    variant.name: variant
    for variant in (
        Variant("baseline"),
        Variant("stock_ei_no_telemetry", telemetry=False),
        Variant("no_instance_features", use_instance_features=False),
        Variant("challengers_500", challengers=500),
        Variant("local_search_1", local_search_iterations=1),
        Variant("random_search_only", random_search_only=True),
        Variant("retrain_after_64", retrain_after=64),
        Variant("no_periodic_save", periodic_save=False),
    )
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _run_lock(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    handle = (destination / ".profile.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise RuntimeError(f"Profile is already running: {destination}") from error
    return handle


def output_directory(
    benchmark_key: str,
    variant_name: str,
    depth: int,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / benchmark_key
        / f"{variant_name}_depth_{depth}"
        / str(smac_seed)
    )


class RuntimeProfiler:
    """Collect inclusive timings and input counts without patching SMAC files."""

    def __init__(
        self,
        smac_facade: AlgorithmConfigurationFacade,
        n_marginalized_instances: int,
        previous_events: list[dict[str, Any]] | None = None,
    ):
        self.smac = smac_facade
        self.n_marginalized_instances = n_marginalized_instances
        self.events = list(previous_events or [])

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
        trials, configurations = self._state()
        event: dict[str, Any] = {
            "phase": phase,
            "seconds": float(seconds),
            "finished_trials": trials,
            "configurations": configurations,
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
                    expanded = n_inputs * self.n_marginalized_instances
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
            grouped.setdefault(str(event["phase"]), []).append(event)
        summary: dict[str, dict[str, float | int]] = {}
        for phase, events in sorted(grouped.items()):
            durations = [float(event["seconds"]) for event in events]
            summary[phase] = {
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
        return summary


def _install_neighborhood_profiler(
    profiler: RuntimeProfiler,
) -> Callable[..., Iterator[Any]]:
    original = local_search_module.get_one_exchange_neighbourhood

    @functools.wraps(original)
    def measured(*args: Any, **kwargs: Any) -> Iterator[Any]:
        iterator = iter(original(*args, **kwargs))
        measured_seconds = 0.0
        count = 0
        recorded = False
        try:
            while True:
                started = time.perf_counter()
                try:
                    neighbor = next(iterator)
                except StopIteration:
                    measured_seconds += time.perf_counter() - started
                    profiler.record(
                        "neighborhood_generation",
                        measured_seconds,
                        n_inputs=count,
                    )
                    recorded = True
                    return
                measured_seconds += time.perf_counter() - started
                count += 1
                yield neighbor
        finally:
            if not recorded:
                profiler.record(
                    "neighborhood_generation",
                    measured_seconds,
                    n_inputs=count,
                )

    local_search_module.get_one_exchange_neighbourhood = measured
    return original


def _initial_design(scenario: Scenario, benchmark_key: str, data: Any):
    choice, payload = load_initial_choice(
        MAIN_EXPERIMENTS[benchmark_key] / "initial_config.json"
    )
    initial_configuration = resolve_initial_configuration(data, choice)
    if choice.kind == "default":
        design = AlgorithmConfigurationFacade.get_initial_design(scenario)
    else:
        design = RandomInitialDesign(
            scenario,
            n_configs=0,
            additional_configs=[initial_configuration],
        )
    return design, payload


def _existing_state(destination: Path) -> bool:
    return all(
        (destination / filename).exists()
        for filename in ("scenario.json", "runhistory.json", "configspace.json")
    )


def run_profile(
    benchmark_key: str,
    variant_name: str,
    depth: int,
    smac_seed: int,
    *,
    overwrite: bool = False,
    enable_cprofile: bool = False,
) -> dict[str, Any]:
    if benchmark_key not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {benchmark_key!r}.")
    if variant_name not in VARIANTS:
        raise ValueError(f"Unknown variant {variant_name!r}.")
    if depth < 1:
        raise ValueError("Depth must be positive.")
    if smac_seed < 0:
        raise ValueError("SMAC seed must be non-negative.")

    variant = VARIANTS[variant_name]
    destination = output_directory(
        benchmark_key, variant_name, depth, smac_seed
    )
    completion_path = destination / "completed.json"
    if completion_path.exists() and not overwrite:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("state") == "complete":
            print(f"Complete profile found; skipping {destination}")
            return json.loads(
                (destination / "runtime_summary.json").read_text(encoding="utf-8")
            )

    with _run_lock(destination):
        if overwrite and _existing_state(destination):
            raise RuntimeError(
                "Refusing to overwrite existing SMAC state. Use a new result "
                f"identity or remove the exact run directory manually: {destination}"
            )

        spec = get_benchmark_spec(benchmark_key)
        data = load_benchmark_data(spec)
        training_instances = data.training_instances
        all_features = {
            instance: data.features[instance]
            for instance in training_instances
        }
        scenario_features = (
            all_features if variant.use_instance_features else None
        )
        initial_choice_path = (
            MAIN_EXPERIMENTS[benchmark_key] / "initial_config.json"
        )
        initial_choice_payload = json.loads(
            initial_choice_path.read_text(encoding="utf-8")
        )
        scenario = Scenario(
            configspace=data.configspace,
            name=f"{variant_name}_depth_{depth}",
            output_directory=OUTPUT_ROOT / benchmark_key,
            deterministic=spec.deterministic,
            objectives="PAR10",
            crash_cost=spec.timeout_cost,
            n_trials=N_TRIALS,
            use_default_config=initial_choice_payload["kind"] == "default",
            instances=list(training_instances),
            instance_features=scenario_features,
            seed=smac_seed,
            n_workers=1,
        )
        if scenario.output_directory != destination:
            raise RuntimeError(
                f"Unexpected output path {scenario.output_directory}; "
                f"expected {destination}."
            )

        initial_design, initial_choice = _initial_design(
            scenario, benchmark_key, data
        )

        model = AlgorithmConfigurationFacade.get_model(
            scenario=scenario,
            n_trees=N_TREES,
            max_depth=depth,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            pca_components=PCA_COMPONENTS,
        )
        random_design = AlgorithmConfigurationFacade.get_random_design(
            scenario,
            probability=RANDOM_DESIGN_PROBABILITY,
        )
        stock_ei = AlgorithmConfigurationFacade.get_acquisition_function(scenario)
        acquisition_function = (
            TelemetryEI.from_smac_ei(stock_ei)
            if variant.telemetry
            else stock_ei
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
        benchmark = ACLibSurrogateBenchmark(spec)
        resume = _existing_state(destination) and not overwrite
        telemetry = (
            SurrogateTelemetryCallback(
                output_directory=destination,
                model=model,
                acquisition_function=acquisition_function,
                overwrite=not resume,
            )
            if variant.telemetry
            else None
        )
        callbacks = [telemetry] if telemetry is not None else []
        facade = AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=benchmark.target,
            model=model,
            acquisition_function=acquisition_function,
            acquisition_maximizer=acquisition_maximizer,
            config_selector=config_selector,
            initial_design=initial_design,
            random_design=random_design,
            callbacks=callbacks,
            overwrite=not resume,
        )

        previous_summary = {}
        previous_events: list[dict[str, Any]] = []
        events_path = destination / "runtime_events.json"
        summary_path = destination / "runtime_summary.json"
        if resume:
            if events_path.exists():
                previous_events = json.loads(events_path.read_text(encoding="utf-8"))
            if summary_path.exists():
                previous_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )

        profiler = RuntimeProfiler(
            facade,
            n_marginalized_instances=(
                len(training_instances) if variant.use_instance_features else 1
            ),
            previous_events=previous_events,
        )
        selector = facade.optimizer._intensifier._config_selector
        maximizer = selector._acquisition_maximizer
        local_search = getattr(maximizer, "_local_search", None)
        random_search = getattr(maximizer, "_random_search", None)

        profiler.wrap(selector, "_collect_data", "collect_data")
        profiler.wrap(selector, "_get_x_best", "get_x_best")
        profiler.wrap(model, "train", "model_train", array_argument=0)
        profiler.wrap(
            model,
            "predict_marginalized",
            "predict_marginalized",
            array_argument=0,
            expand_instances=variant.use_instance_features,
        )
        profiler.wrap(
            acquisition_function,
            "update",
            "acquisition_update",
        )
        profiler.wrap(
            scenario.configspace,
            "sample_configuration",
            "configspace_sample",
        )
        profiler.wrap(maximizer, "_maximize", "acquisition_maximize")
        if random_search is not None:
            profiler.wrap(random_search, "_maximize", "random_search")
        if local_search is not None:
            profiler.wrap(local_search, "_maximize", "local_search_maximize")
            profiler.wrap(local_search, "_search", "local_search_walk")
        profiler.wrap(facade.optimizer, "ask", "ask")
        profiler.wrap(facade.optimizer, "tell", "tell")
        profiler.wrap(facade.runhistory, "save", "runhistory_save")
        profiler.wrap(
            facade.optimizer._intensifier,
            "save",
            "intensifier_save",
        )
        if telemetry is not None:
            profiler.wrap(telemetry, "_snapshot", "telemetry_snapshot")
            profiler.wrap(
                telemetry,
                "_append_event",
                "telemetry_append",
            )

        original_neighborhood = _install_neighborhood_profiler(profiler)
        optimizer = facade.optimizer
        original_optimizer_save = optimizer.save
        if variant.periodic_save:
            profiler.wrap(optimizer, "save", "optimizer_save")
        else:
            optimizer.save = lambda: None

        local_random_forest_source = Path(inspect.getfile(RandomForest)).resolve()
        if LOCAL_SMAC_ROOT.resolve() not in local_random_forest_source.parents:
            raise RuntimeError(
                f"Expected local SMAC RF, found {local_random_forest_source}."
            )
        metadata = {
            "experiment_version": EXPERIMENT_VERSION,
            "benchmark": benchmark_key,
            "display_name": spec.display_name,
            "variant": asdict(variant),
            "depth": depth,
            "smac_seed": smac_seed,
            "n_trials": N_TRIALS,
            "n_training_instances": len(training_instances),
            "instance_feature_dimensions": spec.expected_features,
            "configuration_space": {
                "hyperparameters": len(data.configspace),
                "conditions": len(data.configspace.conditions),
                "forbiddens": len(data.configspace.forbidden_clauses),
            },
            "initial_configuration": initial_choice,
            "rf": {
                "n_trees": N_TREES,
                "max_depth": depth,
                "min_samples_split": MIN_SAMPLES_SPLIT,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "pca_components": PCA_COMPONENTS,
            },
            "random_design_probability": RANDOM_DESIGN_PROBABILITY,
            "assets": asset_metadata(spec),
            "local_smac_root": str(LOCAL_SMAC_ROOT),
            "local_random_forest_source": str(local_random_forest_source),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
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
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
            "timings_are_inclusive": True,
        }
        _atomic_json(destination / "run_metadata.json", metadata)

        python_profile = cProfile.Profile() if enable_cprofile else None
        segment_started = time.perf_counter()
        error: str | None = None
        incumbent = None
        try:
            if python_profile is not None:
                python_profile.enable()
            incumbent = facade.optimize()
        except BaseException as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            if python_profile is not None:
                python_profile.disable()
                python_profile.dump_stats(destination / "runtime.prof")
            local_search_module.get_one_exchange_neighbourhood = (
                original_neighborhood
            )
            if not variant.periodic_save:
                optimizer.save = original_optimizer_save
                save_started = time.perf_counter()
                original_optimizer_save()
                profiler.record(
                    "final_optimizer_save",
                    time.perf_counter() - save_started,
                )

            segment_seconds = time.perf_counter() - segment_started
            cumulative_walltime = float(
                previous_summary.get("walltime_seconds", 0.0)
            ) + segment_seconds
            finished_trials = int(facade.runhistory.finished)
            configurations = len(facade.runhistory._config_ids)
            result = {
                **metadata,
                "finished_trials": finished_trials,
                "configurations": configurations,
                "trials_per_configuration": (
                    finished_trials / configurations
                    if configurations
                    else None
                ),
                "model_train_calls": profiler.summary().get(
                    "model_train", {}
                ).get("calls", 0),
                "walltime_seconds": cumulative_walltime,
                "last_segment_seconds": segment_seconds,
                "target_function_walltime_seconds": float(
                    optimizer.used_target_function_walltime
                ),
                "target_evaluations_this_process": benchmark.evaluation_count,
                "incumbent_cost": (
                    float(
                        facade.runhistory.average_cost(
                            incumbent, normalize=False
                        )
                    )
                    if incumbent is not None
                    else None
                ),
                "error": error,
                "phase_summary": profiler.summary(),
            }
            _atomic_json(events_path, profiler.events)
            _atomic_json(summary_path, result)
            complete = finished_trials >= N_TRIALS and error is None
            _atomic_json(
                completion_path,
                {
                    "state": "complete" if complete else "incomplete",
                    "benchmark": benchmark_key,
                    "variant": variant_name,
                    "depth": depth,
                    "smac_seed": smac_seed,
                    "finished_trials": finished_trials,
                    "n_trials": N_TRIALS,
                },
            )

        print(json.dumps(result, indent=2, sort_keys=True))
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--smac-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cprofile", action="store_true")
    args = parser.parse_args()
    run_profile(
        benchmark_key=args.benchmark,
        variant_name=args.variant,
        depth=args.depth,
        smac_seed=args.smac_seed,
        overwrite=args.overwrite,
        enable_cprofile=args.cprofile,
    )


if __name__ == "__main__":
    main()

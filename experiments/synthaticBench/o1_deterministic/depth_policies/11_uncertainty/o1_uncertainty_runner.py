from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback
from smac.utils.configspace import convert_configurations_to_array

DEPTHS = (5, 10, 15, 20, 25, 30)
BENCHMARK_SEEDS = (40, 42)
SMAC_SEEDS = tuple(range(3))
INSTANCE_SEED = 0
INSTANCE_STD = 2.0
PYTHONHASHSEED = "12345"
DIMENSION = 12
N_INSTANCES = 12
N_TRIALS = 2_000
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 1
RANDOM_DESIGN_PROBABILITY = 0.0
CONFIG_SELECTOR_RETRAIN_AFTER = 8
EXPERIMENT_VERSION = 1
PROPOSAL_SCHEMA_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def fixed_depth_name(depth: int) -> str:
    return f"fixed_depth_{depth}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}.")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default)
    )
    temporary.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default))
            handle.write("\n")
    temporary.replace(path)


def make_instance_map() -> dict[str, float]:
    """Create one deterministic map reused by every run in the experiment."""
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{index}": float(offset)
        for index, offset in enumerate(rng.normal(0, INSTANCE_STD, N_INSTANCES))
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def output_directory(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"benchmark_seed_{benchmark_seed}"
        / fixed_depth_name(depth)
        / str(smac_seed)
    )


def trajectory_path(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return output_directory(benchmark_seed, smac_seed, depth) / "trajectory.json"


def proposal_json_path(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return output_directory(benchmark_seed, smac_seed, depth) / "proposal_diagnostics.json"


def proposal_jsonl_path(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return output_directory(benchmark_seed, smac_seed, depth) / "proposal_diagnostics.jsonl"


def is_acquisition_selected_origin(origin: str | None) -> bool:
    """Identify configurations returned by an acquisition-function maximizer."""
    text = "" if origin is None else str(origin)
    return (
        text.startswith("Acquisition Function Maximizer:")
        or text.startswith("Acquisition Function:")
    )


class ProposalDiagnosticsCallback(Callback):
    """Capture the exact surrogate and acquisition state at proposal time."""

    def __init__(self, live_jsonl_path: Path) -> None:
        super().__init__()
        self.live_jsonl_path = live_jsonl_path
        self.records: list[dict[str, Any]] = []
        self._configurations: list[Any] = []

    def _append_live_record(self, record: dict[str, Any]) -> None:
        self.live_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.live_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, sort_keys=True, default=_json_default)
            )
            handle.write("\n")

    def on_next_configurations_end(self, config_selector, config) -> None:
        origin = config.origin
        selected_by_acquisition = is_acquisition_selected_origin(origin)
        configuration_array = convert_configurations_to_array([config])
        record: dict[str, Any] = {
            "proposal_index": len(self.records) + 1,
            "completed_trials_before_proposal": len(config_selector._runhistory),
            "configuration": dict(config),
            "configuration_array": configuration_array[0].tolist(),
            "origin": origin,
            "selected_by_acquisition": selected_by_acquisition,
            "runhistory_config_id": None,
            "surrogate_training_observations": None,
            "model_max_depth": None,
            "acquisition_function": None,
            "acquisition_eta": None,
            "acquisition_xi": None,
            "predicted_cost_mean_marginalized": None,
            "predicted_variance_marginalized": None,
            "predicted_std_marginalized": None,
            "acquisition_value": None,
        }

        if selected_by_acquisition:
            model = config_selector._model
            acquisition_function = config_selector._acquisition_function
            if model is None or acquisition_function is None:
                raise RuntimeError(
                    "An acquisition-selected configuration has no model or "
                    "acquisition function."
                )
            try:
                mean, variance = model.predict_marginalized(configuration_array)
                acquisition_value = acquisition_function([config])
            except Exception as error:
                raise RuntimeError(
                    "Could not record proposal-time surrogate diagnostics for "
                    f"origin={origin!r}."
                ) from error

            predicted_mean = float(np.asarray(mean).reshape(-1)[0])
            predicted_variance = float(np.asarray(variance).reshape(-1)[0])
            record.update(
                {
                    "surrogate_training_observations": int(
                        config_selector._previous_entries
                    ),
                    "model_max_depth": int(model._rf_opts["max_depth"]),
                    "acquisition_function": acquisition_function.name,
                    "acquisition_eta": (
                        None
                        if getattr(acquisition_function, "_eta", None) is None
                        else float(acquisition_function._eta)
                    ),
                    "acquisition_xi": (
                        None
                        if getattr(acquisition_function, "_xi", None) is None
                        else float(acquisition_function._xi)
                    ),
                    "predicted_cost_mean_marginalized": predicted_mean,
                    "predicted_variance_marginalized": predicted_variance,
                    "predicted_std_marginalized": float(
                        np.sqrt(max(0.0, predicted_variance))
                    ),
                    "acquisition_value": float(
                        np.asarray(acquisition_value).reshape(-1)[0]
                    ),
                }
            )

        self.records.append(record)
        self._configurations.append(config)
        self._append_live_record(record)

    def export(
        self,
        runhistory: Any,
        *,
        benchmark_seed: int,
        smac_seed: int,
        depth: int,
    ) -> dict[str, Any]:
        if len(self.records) != len(self._configurations):
            raise RuntimeError("Proposal records and configurations diverged.")
        for record, config in zip(self.records, self._configurations):
            if runhistory.has_config(config):
                record["runhistory_config_id"] = int(
                    runhistory.get_config_id(config)
                )

        selected = [
            record for record in self.records
            if record["selected_by_acquisition"]
        ]
        excluded_origin_counts = Counter(
            str(record["origin"])
            for record in self.records
            if not record["selected_by_acquisition"]
        )
        return {
            "schema_version": PROPOSAL_SCHEMA_VERSION,
            "benchmark_seed": benchmark_seed,
            "smac_seed": smac_seed,
            "policy": fixed_depth_name(depth),
            "max_depth": depth,
            "definitions": {
                "prediction_timing": (
                    "on_next_configurations_end: after challenger selection and "
                    "before the configuration is yielded for evaluation"
                ),
                "prediction_scope": (
                    "predict_marginalized, matching Expected Improvement"
                ),
                "predicted_cost_mean_marginalized": (
                    "surrogate-predicted cost; lower is better"
                ),
                "predicted_variance_marginalized": (
                    "random-forest predictive variance used by Expected Improvement"
                ),
                "acquisition_value": (
                    "Expected Improvement recomputed immediately with the unchanged "
                    "model and acquisition state; higher is preferred"
                ),
                "acquisition_eta": (
                    "incumbent threshold held by Expected Improvement at selection"
                ),
            },
            "total_configuration_selector_proposals": len(self.records),
            "acquisition_selected_proposals": len(selected),
            "non_acquisition_proposals": len(self.records) - len(selected),
            "non_acquisition_origin_counts": dict(excluded_origin_counts),
            "proposals": self.records,
        }


def proposal_diagnostics_are_valid(data: dict[str, Any], depth: int) -> bool:
    proposals = data.get("proposals", ())
    selected = [
        record for record in proposals
        if record.get("selected_by_acquisition")
    ]
    required_prediction_fields = (
        "predicted_cost_mean_marginalized",
        "predicted_variance_marginalized",
        "predicted_std_marginalized",
        "acquisition_value",
        "acquisition_eta",
    )
    return (
        data.get("schema_version") == PROPOSAL_SCHEMA_VERSION
        and data.get("max_depth") == depth
        and data.get("total_configuration_selector_proposals") == len(proposals)
        and data.get("acquisition_selected_proposals") == len(selected)
        and len(selected) > 0
        and all(
            record.get("model_max_depth") == depth
            and all(record.get(field) is not None for field in required_prediction_fields)
            for record in selected
        )
    )


def trajectory_is_complete(benchmark_seed: int, smac_seed: int, depth: int) -> bool:
    directory = output_directory(benchmark_seed, smac_seed, depth)
    required = (
        directory / "trajectory.json",
        directory / "runhistory.json",
        directory / "incumbent.json",
        directory / "runtime.json",
        directory / "proposal_diagnostics.json",
        directory / "proposal_diagnostics.jsonl",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        trajectory = json.loads(required[0].read_text())
        diagnostics = json.loads(required[4].read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        trajectory.get("experiment_version") == EXPERIMENT_VERSION
        and trajectory.get("policy") == fixed_depth_name(depth)
        and trajectory.get("benchmark_seed") == benchmark_seed
        and trajectory.get("smac_seed") == smac_seed
        and trajectory.get("max_depth") == depth
        and trajectory.get("dimension") == DIMENSION
        and trajectory.get("n_instances") == N_INSTANCES
        and trajectory.get("n_trials") == N_TRIALS
        and trajectory.get("min_samples_leaf") == MIN_SAMPLES_LEAF
        and trajectory.get("min_samples_split") == MIN_SAMPLES_SPLIT
        and np.isclose(
            float(trajectory.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(trajectory.get("instance_map", ())) == N_INSTANCES
        and len(trajectory.get("best_regret", ())) == N_TRIALS
        and diagnostics.get("benchmark_seed") == benchmark_seed
        and diagnostics.get("smac_seed") == smac_seed
        and proposal_diagnostics_are_valid(diagnostics, depth)
    )


def run_fixed_depth(
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Benchmark seed must be one of {BENCHMARK_SEEDS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"SMAC seed must be one of {SMAC_SEEDS}.")
    if depth not in DEPTHS:
        raise ValueError(f"Depth must be one of {DEPTHS}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(benchmark_seed, smac_seed, depth):
        print(
            f"Skipping complete benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}, depth={depth}."
        )
        return json.loads(
            trajectory_path(benchmark_seed, smac_seed, depth).read_text()
        )

    final_proposal_path = proposal_json_path(benchmark_seed, smac_seed, depth)
    live_proposal_path = proposal_jsonl_path(benchmark_seed, smac_seed, depth)
    for stale_path in (final_proposal_path, live_proposal_path):
        if stale_path.exists():
            stale_path.unlink()

    total_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = benchmark_seed
    problem_cfg.problem.function.wrapped_bench.dim = DIMENSION
    problem_cfg.task.dimensions = DIMENSION
    problem_cfg.task.search_space_n_floats = DIMENSION
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map()
    problem.set_instances(instance_map)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    name = fixed_depth_name(depth)
    scenario = Scenario(
        name=name,
        output_directory=OUTPUT_DIRECTORY / f"benchmark_seed_{benchmark_seed}",
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    proposal_callback = ProposalDiagnosticsCallback(live_proposal_path)
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        callbacks=[proposal_callback],
        overwrite=True,
    )
    optimization_started = time.perf_counter()
    incumbent = smac.optimize()
    optimization_seconds = time.perf_counter() - optimization_started

    trials = ordered_trials(smac.runhistory)
    if len(trials) != N_TRIALS:
        raise RuntimeError(f"Expected {N_TRIALS} trials, got {len(trials)}.")
    proposal_data = proposal_callback.export(
        smac.runhistory,
        benchmark_seed=benchmark_seed,
        smac_seed=smac_seed,
        depth=depth,
    )
    if not proposal_diagnostics_are_valid(proposal_data, depth):
        raise RuntimeError("Completed run has invalid proposal diagnostics.")

    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    incumbent_data = {
        "configuration": dict(incumbent),
        "cost": float(smac.runhistory.get_cost(incumbent)),
    }
    runtime_data = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_seconds": optimization_seconds,
        "total_seconds": time.perf_counter() - total_started,
    }
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "facade": "AlgorithmConfigurationFacade",
        "policy": name,
        "policy_type": "fixed_depth",
        "max_depth": int(model._rf_opts["max_depth"]),
        "benchmark_seed": benchmark_seed,
        "problem_seed": benchmark_seed,
        "smac_seed": smac_seed,
        "instance_seed": INSTANCE_SEED,
        "instance_distribution": "normal",
        "instance_mean": 0.0,
        "instance_standard_deviation": INSTANCE_STD,
        "instance_map": instance_map,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "dimension": DIMENSION,
        "n_instances": N_INSTANCES,
        "n_trials": len(trials),
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "min_samples_split": int(model._rf_opts["min_samples_split"]),
        "config_selector_retrain_after": CONFIG_SELECTOR_RETRAIN_AFTER,
        "incumbent": incumbent_data["configuration"],
        "incumbent_cost": incumbent_data["cost"],
        "runtime": runtime_data,
        "proposal_diagnostics_file": "proposal_diagnostics.json",
        "proposal_diagnostics_jsonl_file": "proposal_diagnostics.jsonl",
        "total_configuration_selector_proposals": proposal_data[
            "total_configuration_selector_proposals"
        ],
        "acquisition_selected_proposals": proposal_data[
            "acquisition_selected_proposals"
        ],
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "f_min": f_min,
        "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": (
            np.minimum.accumulate(objective_values).astype(float).tolist()
        ),
        "trials_per_config": {
            str(config_id): count
            for config_id, count in sorted(trials_per_config.items())
        },
    }
    directory = scenario.output_directory
    smac.runhistory.save(directory / "runhistory.json")
    atomic_write_json(directory / "incumbent.json", incumbent_data)
    atomic_write_json(directory / "runtime.json", runtime_data)
    atomic_write_json(final_proposal_path, proposal_data)
    atomic_write_jsonl(live_proposal_path, proposal_data["proposals"])
    atomic_write_json(directory / "trajectory.json", result)
    print(
        f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}, depth={depth}, "
        f"acquisition_proposals={proposal_data['acquisition_selected_proposals']}, "
        f"output={directory}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--smac-seed", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    run_fixed_depth(args.benchmark_seed, args.smac_seed, args.depth)


if __name__ == "__main__":
    main()

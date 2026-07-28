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

BENCHMARK_SEEDS = (40,)
SMAC_SEEDS = tuple(range(7))
FIXED_DEPTHS = (5, 10, 15, 20, 25, 30, 40, 50)
INSTANCE_SEED = 0
INSTANCE_STD = 2.0
PYTHONHASHSEED = "12345"
DIMENSION = 15
N_INSTANCES = 20
N_TRIALS = 7_000
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 1
RANDOM_DESIGN_PROBABILITY = 0.0
CONFIG_SELECTOR_RETRAIN_AFTER = 8
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def policy_name(depth: int) -> str:
    return f"fixed_depth_{depth}"


def policy_spec(depth: int) -> dict[str, Any]:
    return {
        "name": policy_name(depth),
        "family": "fixed",
        "kind": "fixed",
        "fixed_depth": int(depth),
    }


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


def make_instance_map() -> dict[str, float]:
    """Return the same deterministic 20-instance map used by the source run."""
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{index}": float(offset)
        for index, offset in enumerate(rng.normal(0, INSTANCE_STD, N_INSTANCES))
    }


def best_average_configuration_cost(runhistory: Any) -> float | None:
    costs = [float(runhistory.get_cost(config)) for config in runhistory.get_configs()]
    finite = [cost for cost in costs if np.isfinite(cost)]
    return min(finite) if finite else None


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


class FixedDepthController:
    """Record the fixed depth and incumbent changes at every model training."""

    def __init__(self, depth: int) -> None:
        self.depth = int(depth)
        self.runhistory: Any | None = None
        self.events: list[dict[str, Any]] = []
        self._active_event: dict[str, Any] | None = None

    def attach_runhistory(self, runhistory: Any) -> None:
        self.runhistory = runhistory

    def note_configuration_suggested(self) -> None:
        if self._active_event is not None:
            self._active_event["configurations_suggested"] += 1

    def _close_event(
        self,
        completed_trials: int,
        incumbent_cost: float | None,
        *,
        closed_by: str,
    ) -> None:
        if self._active_event is None:
            return
        start_cost = self._active_event["incumbent_cost_before"]
        improvement = (
            max(0.0, float(start_cost) - float(incumbent_cost))
            if start_cost is not None and incumbent_cost is not None
            else 0.0
        )
        self.events.append(
            {
                **self._active_event,
                "completed_trials_before_next_training": completed_trials,
                "incumbent_cost_before_next_training": incumbent_cost,
                "improvement": float(improvement),
                "closed_by": closed_by,
            }
        )
        self._active_event = None

    def before_surrogate_training(
        self,
        completed_trials: int,
        incumbent_cost: float | None,
    ) -> int:
        self._close_event(
            completed_trials,
            incumbent_cost,
            closed_by="next_surrogate_training",
        )
        self._active_event = {
            "surrogate_training_event": len(self.events),
            "completed_trials_before_training": completed_trials,
            "active_depth": self.depth,
            "incumbent_cost_before": incumbent_cost,
            "configurations_suggested": 0,
            "evaluation_window": None,
        }
        return self.depth

    def finalize(self, completed_trials: int, incumbent_cost: float | None) -> None:
        self._close_event(
            completed_trials,
            incumbent_cost,
            closed_by="optimization_end",
        )

    def export(self) -> dict[str, Any]:
        return {
            "surrogate_training_events": self.events,
            "depth_transitions": [
                {
                    "completed_trials": self.events[0][
                        "completed_trials_before_training"
                    ],
                    "surrogate_training_event": 0,
                    "depth": self.depth,
                }
            ] if self.events else [],
            "evaluation_windows": [],
            "selected_depths": [],
            "policy_transitions": [
                {
                    "scheduled_boundary": 0,
                    "effective_completed_trials": 0,
                    "action": "initialize_fixed_depth",
                    "new_mode": "fixed",
                    "new_depths": [self.depth],
                }
            ],
            "final_mode": "fixed",
            "final_depth_set": [self.depth],
        }


class FixedDepthCallback(Callback):
    def __init__(self, controller: FixedDepthController) -> None:
        super().__init__()
        self.controller = controller

    def on_next_configurations_end(self, config_selector, config) -> None:
        self.controller.note_configuration_suggested()

    def on_end(self, smbo) -> None:
        self.controller.finalize(
            len(smbo.runhistory),
            best_average_configuration_cost(smbo.runhistory),
        )


def install_depth_control(model: Any, controller: FixedDepthController) -> None:
    original_train = model.train

    def controlled_train(X: np.ndarray, Y: np.ndarray) -> Any:
        if controller.runhistory is None:
            raise RuntimeError("The fixed-depth controller has no runhistory.")
        depth = controller.before_surrogate_training(
            len(controller.runhistory),
            best_average_configuration_cost(controller.runhistory),
        )
        model._rf_opts["max_depth"] = depth
        return original_train(X, Y)

    model.train = controlled_train


def run_output_directory(
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"benchmark_seed_{benchmark_seed}"
        / policy_name(depth)
        / str(smac_seed)
    )


def trajectory_path(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return run_output_directory(benchmark_seed, smac_seed, depth) / "trajectory.json"


def trajectory_is_complete(
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> bool:
    directory = run_output_directory(benchmark_seed, smac_seed, depth)
    required = (
        directory / "trajectory.json",
        directory / "runhistory.json",
        directory / "incumbent.json",
        directory / "runtime.json",
        directory / "policy_events.json",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        data = json.loads(required[0].read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("policy_spec") == policy_spec(depth)
        and data.get("benchmark_seed") == benchmark_seed
        and data.get("smac_seed") == smac_seed
        and data.get("dimension") == DIMENSION
        and data.get("n_instances") == N_INSTANCES
        and data.get("n_trials") == N_TRIALS
        and data.get("min_samples_leaf") == MIN_SAMPLES_LEAF
        and data.get("min_samples_split") == MIN_SAMPLES_SPLIT
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(data.get("best_regret", ())) == N_TRIALS
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
    if depth not in FIXED_DEPTHS:
        raise ValueError(f"Depth must be one of {FIXED_DEPTHS}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(benchmark_seed, smac_seed, depth):
        print(
            f"Skipping complete depth={depth}, benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(benchmark_seed, smac_seed, depth).read_text()
        )

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
        return float(
            problem.evaluate(
                TrialInfo(config=config, instance=instance, seed=seed)
            ).cost
        )

    name = policy_name(depth)
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
    controller = FixedDepthController(depth)
    callback = FixedDepthCallback(controller)
    install_depth_control(model, controller)
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        callbacks=[callback],
        overwrite=True,
    )
    controller.attach_runhistory(smac.runhistory)

    optimize_started = time.perf_counter()
    incumbent = smac.optimize()
    optimize_seconds = time.perf_counter() - optimize_started

    trials = ordered_trials(smac.runhistory)
    if len(trials) != N_TRIALS:
        raise RuntimeError(f"Expected {N_TRIALS} completed trials, got {len(trials)}.")
    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    policy_data = controller.export()
    output_directory = scenario.output_directory
    incumbent_data = {
        "configuration": dict(incumbent),
        "cost": float(smac.runhistory.get_cost(incumbent)),
    }
    runtime_data = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimization_seconds": optimize_seconds,
        "total_seconds": time.perf_counter() - total_started,
    }
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "facade": "AlgorithmConfigurationFacade",
        "policy": name,
        "policy_family": "fixed",
        "policy_spec": policy_spec(depth),
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
        **policy_data,
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
    smac.runhistory.save(output_directory / "runhistory.json")
    atomic_write_json(output_directory / "incumbent.json", incumbent_data)
    atomic_write_json(output_directory / "runtime.json", runtime_data)
    atomic_write_json(output_directory / "policy_events.json", policy_data)
    atomic_write_json(output_directory / "trajectory.json", result)
    print(
        f"depth={depth}, benchmark_seed={benchmark_seed}, smac_seed={smac_seed}, "
        f"output={output_directory}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-seed", type=int, default=40)
    parser.add_argument("--smac-seed", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    run_fixed_depth(args.benchmark_seed, args.smac_seed, args.depth)


if __name__ == "__main__":
    main()

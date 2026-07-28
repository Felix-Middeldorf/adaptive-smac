from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario

INSTANCE_COUNTS = (5, 10, 25, 50)
DEPTHS = (5, 10, 15, 20)
BENCHMARK_SEEDS = (40, 41)
SMAC_SEEDS = tuple(range(5))
INSTANCE_SEED = 0
INSTANCE_STD = 2.0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_TRIALS = 1000
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def instance_count_name(n_instances: int) -> str:
    return f"instances_{n_instances}"


def fixed_depth_name(depth: int) -> str:
    return f"fixed_depth_{depth}"


def make_instance_map(n_instances: int) -> dict[str, float]:
    if n_instances not in INSTANCE_COUNTS:
        raise ValueError(
            f"Expected instance count from {INSTANCE_COUNTS}, got {n_instances}."
        )
    rng = np.random.default_rng(INSTANCE_SEED)
    offsets = rng.normal(0, INSTANCE_STD, max(INSTANCE_COUNTS))
    return {
        f"i{i}": float(offsets[i])
        for i in range(n_instances)
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def trajectory_path(
    n_instances: int,
    benchmark_seed: int,
    depth: int,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / instance_count_name(n_instances)
        / f"benchmark_seed_{benchmark_seed}"
        / fixed_depth_name(depth)
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    n_instances: int,
    benchmark_seed: int,
    depth: int,
    smac_seed: int,
) -> bool:
    path = trajectory_path(n_instances, benchmark_seed, depth, smac_seed)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("n_trials") == N_TRIALS
        and data.get("benchmark_seed") == benchmark_seed
        and data.get("smac_seed") == smac_seed
        and data.get("max_depth") == depth
        and data.get("dimension") == DIMENSION
        and data.get("n_instances") == n_instances
        and np.isclose(
            float(data.get("instance_standard_deviation", -1.0)),
            INSTANCE_STD,
        )
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def run_fixed_depth(
    n_instances: int,
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> dict[str, Any]:
    if n_instances not in INSTANCE_COUNTS:
        raise ValueError(
            f"Expected instance count from {INSTANCE_COUNTS}, got {n_instances}."
        )
    if depth not in DEPTHS:
        raise ValueError(f"Expected depth from {DEPTHS}, got {depth}.")
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(
            f"Expected benchmark seed from {BENCHMARK_SEEDS}, got {benchmark_seed}."
        )
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"Expected SMAC seed from {SMAC_SEEDS}, got {smac_seed}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(n_instances, benchmark_seed, depth, smac_seed):
        print(
            f"Skipping complete n_instances={n_instances}, "
            f"benchmark_seed={benchmark_seed}, depth={depth}, smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(n_instances, benchmark_seed, depth, smac_seed).read_text()
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = benchmark_seed
    problem_cfg.problem.function.wrapped_bench.dim = DIMENSION
    problem_cfg.task.dimensions = DIMENSION
    problem_cfg.task.search_space_n_floats = DIMENSION
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances)
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        return float(problem.evaluate(trial).cost)

    name = fixed_depth_name(depth)
    scenario = Scenario(
        name=name,
        output_directory=(
            OUTPUT_DIRECTORY
            / instance_count_name(n_instances)
            / f"benchmark_seed_{benchmark_seed}"
        ),
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(scenario=scenario, max_depth=depth)
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        overwrite=True,
    )
    incumbent = smac.optimize()

    trials = ordered_trials(smac.runhistory)
    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "policy": name,
        "policy_type": "fixed_depth",
        "depth_schedule": [depth, depth, depth],
        "max_depth": depth,
        "smac_seed": smac_seed,
        "benchmark_seed": benchmark_seed,
        "problem_seed": benchmark_seed,
        "instance_seed": INSTANCE_SEED,
        "instance_distribution": "normal",
        "instance_mean": 0.0,
        "instance_standard_deviation": INSTANCE_STD,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "dimension": DIMENSION,
        "n_instances": n_instances,
        "n_trials": len(trials),
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "min_samples_split": int(model._rf_opts["min_samples_split"]),
        "instance_map": instance_map,
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "f_min": f_min,
        "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": np.minimum.accumulate(objective_values).astype(float).tolist(),
        "trials_per_config": {
            str(config_id): count
            for config_id, count in sorted(trials_per_config.items())
        },
    }
    output_path = scenario.output_directory / "trajectory.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(result, indent=2))
    temporary_path.replace(output_path)
    print(
        f"n_instances={n_instances}, benchmark_seed={benchmark_seed}, "
        f"depth={depth}, smac_seed={smac_seed}, output={output_path}"
    )
    return result


def run_depth_batch(
    n_instances: int,
    benchmark_seed: int,
    smac_seed: int,
    depths: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not depths:
        raise ValueError("A depth batch must not be empty.")
    return [
        run_fixed_depth(n_instances, benchmark_seed, smac_seed, depth)
        for depth in depths
    ]

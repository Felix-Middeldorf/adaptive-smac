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
from smac.initial_design import RandomInitialDesign

INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INITIAL_CONFIGS = 10
N_INSTANCES = 10
DEFAULT_DIMENSION = 10
DEFAULT_NUM_QUADRATIC = 3

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/C1-RelevantParameters.yaml"
)


def make_instance_map(n_instances: int = N_INSTANCES) -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, n_instances))
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_depth_policy(
    max_depth: int,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    num_quadratic: int = DEFAULT_NUM_QUADRATIC,
) -> dict[str, Any]:
    return run_c1_policy(
        policy=f"fixed_depth_{max_depth}",
        smac_seed=smac_seed,
        problem_seed=problem_seed,
        output_directory=output_directory,
        n_trials=n_trials,
        n_instances=n_instances,
        dimension=dimension,
        num_quadratic=num_quadratic,
        max_depth=max_depth,
    )


def run_leaf_policy(
    min_samples_leaf: int,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    num_quadratic: int = DEFAULT_NUM_QUADRATIC,
) -> dict[str, Any]:
    return run_c1_policy(
        policy=f"fixed_leaf_{min_samples_leaf}",
        smac_seed=smac_seed,
        problem_seed=problem_seed,
        output_directory=output_directory,
        n_trials=n_trials,
        n_instances=n_instances,
        dimension=dimension,
        num_quadratic=num_quadratic,
        min_samples_leaf=min_samples_leaf,
    )


def run_c1_policy(
    policy: str,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    num_quadratic: int = DEFAULT_NUM_QUADRATIC,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = problem_seed
    problem_cfg.problem.function.dim = dimension
    problem_cfg.problem.function.num_quadratic = num_quadratic
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances)
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        cost = np.asarray(problem.evaluate(trial).cost, dtype=float).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one C1 objective value, got {cost}")
        return float(cost[0])

    scenario = Scenario(
        name=policy,
        output_directory=output_directory,
        configspace=problem.configspace,
        deterministic=False,
        instances=list(instance_map),
        n_trials=n_trials,
        seed=smac_seed,
    )
    model_options: dict[str, int] = {}
    if max_depth is not None:
        model_options["max_depth"] = max_depth
    if min_samples_leaf is not None:
        model_options["min_samples_leaf"] = min_samples_leaf
    model = ACFacade.get_model(scenario=scenario, **model_options)
    initial_design = RandomInitialDesign(
        scenario=scenario,
        n_configs=N_INITIAL_CONFIGS,
        seed=smac_seed,
    )
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        initial_design=initial_design,
        overwrite=True,
    )
    incumbent = smac.optimize()

    trials = ordered_trials(smac.runhistory)
    costs = [float(value.cost) for _, value in trials]
    objective_values = [
        float(value.cost) - instance_map[key.instance]
        for key, value in trials
    ]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    result = {
        "benchmark": "SynthACticBench",
        "problem": "C1-RelevantParameters",
        "dimension": dimension,
        "num_quadratic": num_quadratic,
        "num_noisy": dimension - num_quadratic,
        "policy": policy,
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": n_instances,
        "instance_map": instance_map,
        "deterministic": False,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "initial_design_seed": smac_seed,
        "max_depth": model._rf_opts["max_depth"],
        "min_samples_leaf": model._rf_opts["min_samples_leaf"],
        "min_samples_split": model._rf_opts["min_samples_split"],
        "n_trials": len(trials),
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
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
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"policy={policy}, seed={smac_seed}, output={output_path}"
    )
    return result

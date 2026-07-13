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
from smac.callback import Callback
from smac.initial_design import RandomInitialDesign

INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
N_INITIAL_CONFIGS = 10
N_INSTANCES = 10
DEFAULT_DIMENSION = 10
DEFAULT_FUNCTION_NAME = "ackley"

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/C2-ParameterInteractions-ackley.yaml"
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


class StagedDepthCallback(Callback):
    def __init__(
        self,
        stage_boundaries: tuple[int, ...],
        depth_schedule: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(depth_schedule) != len(stage_boundaries) + 1:
            raise ValueError(
                "depth_schedule must contain one more value than "
                "stage_boundaries"
            )
        self.stage_boundaries = stage_boundaries
        self.depth_schedule = depth_schedule
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def depth_for_completed_trials(self, completed_trials: int) -> int:
        for boundary, depth in zip(
            self.stage_boundaries,
            self.depth_schedule,
        ):
            if completed_trials < boundary:
                return depth
        return self.depth_schedule[-1]

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = self.depth_for_completed_trials(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[StagedDepth] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


def run_depth_policy(
    max_depth: int,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    function_name: str = DEFAULT_FUNCTION_NAME,
) -> dict[str, Any]:
    return run_c2_policy(
        policy=f"fixed_depth_{max_depth}",
        smac_seed=smac_seed,
        problem_seed=problem_seed,
        output_directory=output_directory,
        n_trials=n_trials,
        n_instances=n_instances,
        dimension=dimension,
        function_name=function_name,
        max_depth=max_depth,
    )


def run_staged_depth_policy(
    policy: str,
    depth_schedule: tuple[int, ...],
    stage_boundaries: tuple[int, ...],
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    function_name: str = DEFAULT_FUNCTION_NAME,
) -> dict[str, Any]:
    return run_c2_policy(
        policy=policy,
        smac_seed=smac_seed,
        problem_seed=problem_seed,
        output_directory=output_directory,
        n_trials=n_trials,
        n_instances=n_instances,
        dimension=dimension,
        function_name=function_name,
        max_depth=depth_schedule[0],
        callback=StagedDepthCallback(stage_boundaries, depth_schedule),
        staged_depth_schedule=depth_schedule,
        staged_boundaries=stage_boundaries,
    )


def run_leaf_policy(
    min_samples_leaf: int,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    function_name: str = DEFAULT_FUNCTION_NAME,
) -> dict[str, Any]:
    return run_c2_policy(
        policy=f"fixed_leaf_{min_samples_leaf}",
        smac_seed=smac_seed,
        problem_seed=problem_seed,
        output_directory=output_directory,
        n_trials=n_trials,
        n_instances=n_instances,
        dimension=dimension,
        function_name=function_name,
        min_samples_leaf=min_samples_leaf,
    )


def run_c2_policy(
    policy: str,
    smac_seed: int,
    problem_seed: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    dimension: int = DEFAULT_DIMENSION,
    function_name: str = DEFAULT_FUNCTION_NAME,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    callback: Callback | None = None,
    staged_depth_schedule: tuple[int, ...] | None = None,
    staged_boundaries: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = problem_seed
    problem_cfg.problem.function.dim = dimension
    problem_cfg.problem.function.name = function_name
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances)
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        cost = np.asarray(problem.evaluate(trial).cost, dtype=float).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one C2 objective value, got {cost}")
        return float(cost[0])

    scenario = Scenario(
        name=policy,
        output_directory=output_directory,
        configspace=problem.configspace,
        deterministic=True,
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
        callbacks=[] if callback is None else [callback],
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
        "problem": "C2-ParameterInteractions",
        "function_name": function_name,
        "dimension": dimension,
        "policy": policy,
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": n_instances,
        "instance_map": instance_map,
        "deterministic": True,
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
    if staged_depth_schedule is not None:
        result.update(
            initial_max_depth=staged_depth_schedule[0],
            final_max_depth=model._rf_opts["max_depth"],
            stage_boundaries=list(staged_boundaries or ()),
            depth_schedule=list(staged_depth_schedule),
            transitions=getattr(callback, "transitions", []),
        )
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(f"policy={policy}, seed={smac_seed}, output={output_path}")
    return result

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
MIN_SAMPLES_SPLIT = 1

ROTATION_BLOCK_SIZE = 100
ROTATING_POLICIES: dict[str, tuple[int, ...]] = {
    "rotate_leaf_5_4_3_2_1_every_100": (5, 4, 3, 2, 1),
    "rotate_leaf_4_3_2_1_every_100": (4, 3, 2, 1),
    "rotate_leaf_3_2_1_every_100": (3, 2, 1),
}
STAGED_POLICY = "staged_leaf_3_2_1_200_200_rest"
STAGED_BOUNDARIES = (200, 400)
STAGED_SCHEDULE = (3, 2, 1)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O6-Multimodal.yaml"
)


def make_instance_map(n_instances: int = N_INSTANCES) -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, n_instances))
    }


def rotating_leaf(policy: str, completed_trials: int) -> int:
    schedule = ROTATING_POLICIES[policy]
    block = completed_trials // ROTATION_BLOCK_SIZE
    return schedule[block % len(schedule)]


def staged_leaf(completed_trials: int) -> int:
    if completed_trials < STAGED_BOUNDARIES[0]:
        return STAGED_SCHEDULE[0]
    if completed_trials < STAGED_BOUNDARIES[1]:
        return STAGED_SCHEDULE[1]
    return STAGED_SCHEDULE[2]


class LeafPolicyCallback(Callback):
    def __init__(self, policy: str) -> None:
        super().__init__()
        self.policy = policy
        self.transitions: list[tuple[int, int]] = []
        self._last_leaf: int | None = None

    def _leaf(self, completed_trials: int) -> int:
        if self.policy in ROTATING_POLICIES:
            return rotating_leaf(self.policy, completed_trials)
        if self.policy == STAGED_POLICY:
            return staged_leaf(completed_trials)
        raise ValueError(f"Unknown adaptive leaf policy: {self.policy}")

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        leaf = self._leaf(completed_trials)
        config_selector._model._rf_opts["min_samples_leaf"] = leaf
        config_selector._model._rf_opts["min_samples_split"] = MIN_SAMPLES_SPLIT
        if leaf != self._last_leaf:
            self.transitions.append((completed_trials, leaf))
            self._last_leaf = leaf
            print(
                f"[LeafPolicy] policy={self.policy}, "
                f"completed_trials={completed_trials}, "
                f"min_samples_leaf={leaf}, "
                f"min_samples_split={MIN_SAMPLES_SPLIT}"
            )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_leaf_policy(
    policy: str,
    smac_seed: int,
    problem_seed: int,
    dimension: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )

    if policy.startswith("fixed_leaf_"):
        initial_leaf = int(policy.rsplit("_", 1)[1])
        callback = None
    elif policy in ROTATING_POLICIES or policy == STAGED_POLICY:
        if policy in ROTATING_POLICIES:
            initial_leaf = ROTATING_POLICIES[policy][0]
        else:
            initial_leaf = STAGED_SCHEDULE[0]
        callback = LeafPolicyCallback(policy)
    else:
        raise ValueError(f"Unknown leaf policy: {policy}")

    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = problem_seed
    problem_cfg.problem.function.dim = dimension
    problem = make_problem(problem_cfg)
    instance_map = make_instance_map(n_instances)
    problem.set_instances(instance_map)

    def target_function(config, instance: str, seed: int = 0) -> float:
        trial = TrialInfo(config=config, instance=instance, seed=seed)
        cost = np.asarray(problem.evaluate(trial).cost, dtype=float).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one O6 objective value, got {cost}")
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
    model = ACFacade.get_model(
        scenario=scenario,
        min_samples_leaf=initial_leaf,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )
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
        "problem": "O6-Multimodal",
        "dimension": dimension,
        "policy": policy,
        "smac_seed": smac_seed,
        "problem_seed": problem_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "n_instances": n_instances,
        "instance_map": instance_map,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "initial_design_seed": smac_seed,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "initial_min_samples_leaf": initial_leaf,
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
    if callback is None:
        result["min_samples_leaf"] = initial_leaf
    elif policy in ROTATING_POLICIES:
        result.update(
            block_size=ROTATION_BLOCK_SIZE,
            leaf_schedule=list(ROTATING_POLICIES[policy]),
            transitions=callback.transitions,
        )
    else:
        result.update(
            stage_boundaries=list(STAGED_BOUNDARIES),
            leaf_schedule=list(STAGED_SCHEDULE),
            transitions=callback.transitions,
        )

    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"dimension={dimension}, policy={policy}, seed={smac_seed}, "
        f"output={output_path}"
    )
    return result

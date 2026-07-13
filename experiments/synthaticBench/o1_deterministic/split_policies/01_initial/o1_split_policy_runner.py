from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback

BENCHMARK_SEEDS = tuple(range(40, 45))
SMAC_SEEDS = tuple(range(5))
SPLIT_VALUES = (1, 2, 3)
STAGE_SWITCHES = (200, 500)
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
N_TRIALS = 1000
MAX_DEPTH = 2000
MIN_SAMPLES_LEAF = 1
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


@dataclass(frozen=True)
class SplitPolicy:
    name: str
    family: str
    schedule_type: str
    fixed_split: int | None = None
    stage_splits: tuple[int, ...] = ()

    def split(self, completed_trials: int) -> int:
        if completed_trials < 0:
            raise ValueError("completed_trials must be non-negative.")
        if self.schedule_type == "fixed":
            if self.fixed_split is None:
                raise ValueError("A fixed policy requires fixed_split.")
            value = self.fixed_split
        elif self.schedule_type == "three_stage":
            if len(self.stage_splits) != 3:
                raise ValueError("A three-stage policy requires three split values.")
            if completed_trials < STAGE_SWITCHES[0]:
                value = self.stage_splits[0]
            elif completed_trials < STAGE_SWITCHES[1]:
                value = self.stage_splits[1]
            else:
                value = self.stage_splits[2]
        else:
            raise ValueError(f"Unknown schedule type {self.schedule_type!r}.")
        if value not in SPLIT_VALUES:
            raise ValueError(f"Split value {value} is outside {SPLIT_VALUES}.")
        return value

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "schedule_type": self.schedule_type,
            "fixed_split": self.fixed_split,
            "stage_splits": list(self.stage_splits),
            "stage_switches": list(STAGE_SWITCHES),
        }

    def planned_transitions(self) -> list[list[int]]:
        transitions: list[list[int]] = []
        previous: int | None = None
        for completed_trials in range(N_TRIALS):
            value = self.split(completed_trials)
            if value != previous:
                transitions.append([completed_trials, value])
                previous = value
        return transitions


def all_policies() -> tuple[SplitPolicy, ...]:
    fixed = tuple(
        SplitPolicy(
            name=f"fixed_split_{value}",
            family="fixed",
            schedule_type="fixed",
            fixed_split=value,
        )
        for value in SPLIT_VALUES
    )
    staged = tuple(
        SplitPolicy(
            name="split_stages_" + "_".join(map(str, values)),
            family="three_stage",
            schedule_type="three_stage",
            stage_splits=values,
        )
        for values in product(SPLIT_VALUES, repeat=3)
        if len(set(values)) > 1
    )
    policies = fixed + staged
    if len(policies) != 27 or len({policy.name for policy in policies}) != 27:
        raise RuntimeError("Expected 27 unique split policies.")
    if Counter(policy.family for policy in policies) != Counter(
        {"fixed": 3, "three_stage": 24}
    ):
        raise RuntimeError("Unexpected split-policy family counts.")
    return policies


class SplitPolicyCallback(Callback):
    def __init__(self, policy: SplitPolicy) -> None:
        super().__init__()
        self.policy = policy
        self._last_split = policy.split(0)
        self.observed_transitions: list[list[int]] = [[0, self._last_split]]

    def on_next_configurations_start(self, config_selector: Any) -> None:
        completed_trials = len(config_selector._runhistory)
        value = self.policy.split(completed_trials)
        config_selector._model._rf_opts["min_samples_split"] = value
        if value != self._last_split:
            self.observed_transitions.append([completed_trials, value])
            self._last_split = value
            print(
                f"[SplitPolicy] completed_trials={completed_trials}, "
                f"min_samples_split={value}"
            )


def make_instance_map() -> dict[str, float]:
    rng = np.random.default_rng(INSTANCE_SEED)
    return {
        f"i{i}": float(offset)
        for i, offset in enumerate(rng.normal(0, 2, N_INSTANCES))
    }


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def trajectory_path(
    policy: SplitPolicy,
    benchmark_seed: int,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / policy.family
        / f"benchmark_seed_{benchmark_seed}"
        / policy.name
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    policy: SplitPolicy,
    benchmark_seed: int,
    smac_seed: int,
) -> bool:
    path = trajectory_path(policy, benchmark_seed, smac_seed)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("policy_spec") == policy.to_spec()
        and data.get("benchmark_seed") == benchmark_seed
        and data.get("smac_seed") == smac_seed
        and data.get("n_trials") == N_TRIALS
        and data.get("max_depth") == MAX_DEPTH
        and data.get("min_samples_leaf") == MIN_SAMPLES_LEAF
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def run_policy(
    benchmark_seed: int,
    smac_seed: int,
    policy: SplitPolicy,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Unexpected benchmark seed {benchmark_seed}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"Unexpected SMAC seed {smac_seed}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(policy, benchmark_seed, smac_seed):
        print(
            f"Skipping complete policy={policy.name}, "
            f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(policy, benchmark_seed, smac_seed).read_text()
        )

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

    scenario = Scenario(
        name=policy.name,
        output_directory=(
            OUTPUT_DIRECTORY / policy.family / f"benchmark_seed_{benchmark_seed}"
        ),
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=policy.split(0),
    )
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    callback = None
    if policy.schedule_type != "fixed":
        callback = SplitPolicyCallback(policy)
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
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
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(key.config_id for key, _ in trials)
    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": "SynthACticBench",
        "problem": "O1-DeterministicObjective",
        "policy": policy.name,
        "policy_family": policy.family,
        "policy_type": policy.schedule_type,
        "policy_spec": policy.to_spec(),
        "planned_transitions": policy.planned_transitions(),
        "observed_transitions": (
            None if callback is None else callback.observed_transitions
        ),
        "benchmark_seed": benchmark_seed,
        "problem_seed": benchmark_seed,
        "smac_seed": smac_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "dimension": DIMENSION,
        "n_instances": N_INSTANCES,
        "instance_map": instance_map,
        "n_trials": len(trials),
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "max_depth": int(model._rf_opts["max_depth"]),
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "initial_min_samples_split": policy.split(0),
        "final_min_samples_split": policy.split(N_TRIALS - 1),
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
        f"policy={policy.name}, benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}, output={output_path}"
    )
    return result


def run_policy_batch(
    benchmark_seed: int,
    smac_seed: int,
    policies: tuple[SplitPolicy, ...],
) -> list[dict[str, Any]]:
    if not policies:
        raise ValueError("A policy batch must not be empty.")
    print(
        f"Running {len(policies)} policies for benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}: {[policy.name for policy in policies]}"
    )
    return [run_policy(benchmark_seed, smac_seed, policy) for policy in policies]

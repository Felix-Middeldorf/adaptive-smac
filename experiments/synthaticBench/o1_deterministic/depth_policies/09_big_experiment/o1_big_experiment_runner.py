from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback

BENCHMARK_SEEDS = tuple(range(40, 47))
SMAC_SEEDS = tuple(range(10))
FIXED_DEPTHS = tuple(range(4, 21))
THREE_STAGE_DEPTHS = (5, 10, 15, 20)
THREE_STAGE_SWITCHES = (250, 660)
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
N_TRIALS = 1000
MAX_DEPTH = 20
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
SAMPLED_THREE_STAGE_POLICIES = HERE / "sampled_three_stage_policies.json"
OUTPUT_DIRECTORY = HERE / "smac_output"


@dataclass(frozen=True)
class DepthPolicy:
    name: str
    family: str
    schedule_type: str
    fixed_depth: int | None = None
    stage_depths: tuple[int, ...] = ()
    start_depth: int | None = None
    increment: int | None = None
    interval: int | None = None
    stop_after: int | None = None
    cycle_depths: tuple[int, ...] = ()

    def depth(self, completed_trials: int) -> int:
        if completed_trials < 0:
            raise ValueError("completed_trials must be non-negative.")

        if self.schedule_type == "fixed":
            assert self.fixed_depth is not None
            depth = self.fixed_depth
        elif self.schedule_type == "three_stage":
            if completed_trials < THREE_STAGE_SWITCHES[0]:
                depth = self.stage_depths[0]
            elif completed_trials < THREE_STAGE_SWITCHES[1]:
                depth = self.stage_depths[1]
            else:
                depth = self.stage_depths[2]
        elif self.schedule_type == "increment":
            assert self.start_depth is not None
            assert self.increment is not None
            assert self.interval is not None
            effective_trials = completed_trials
            if self.stop_after is not None:
                effective_trials = min(effective_trials, self.stop_after)
            depth = self.start_depth + self.increment * (
                effective_trials // self.interval
            )
        elif self.schedule_type == "alternating":
            assert self.start_depth is not None
            assert self.interval is not None
            depth = self.start_depth
            for step in range(1, completed_trials // self.interval + 1):
                if step % 2 == 1:
                    depth = min(MAX_DEPTH, depth + 4)
                else:
                    depth -= 2
        elif self.schedule_type == "cycle":
            assert self.interval is not None
            depth = self.cycle_depths[
                (completed_trials // self.interval) % len(self.cycle_depths)
            ]
        else:
            raise ValueError(f"Unknown schedule type {self.schedule_type!r}.")

        return min(MAX_DEPTH, int(depth))

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "schedule_type": self.schedule_type,
            "fixed_depth": self.fixed_depth,
            "stage_depths": list(self.stage_depths),
            "start_depth": self.start_depth,
            "increment": self.increment,
            "interval": self.interval,
            "stop_after": self.stop_after,
            "cycle_depths": list(self.cycle_depths),
            "max_depth_cap": MAX_DEPTH,
        }

    def planned_transitions(self) -> list[list[int]]:
        transitions: list[list[int]] = []
        previous_depth: int | None = None
        for completed_trials in range(N_TRIALS):
            depth = self.depth(completed_trials)
            if depth != previous_depth:
                transitions.append([completed_trials, depth])
                previous_depth = depth
        return transitions


def _load_three_stage_policies() -> tuple[DepthPolicy, ...]:
    manifest = json.loads(SAMPLED_THREE_STAGE_POLICIES.read_text())
    schedules = [tuple(map(int, values)) for values in manifest["depth_schedules"]]
    if len(schedules) != 40 or len(set(schedules)) != 40:
        raise RuntimeError("Expected 40 unique sampled three-stage schedules.")
    if any(len(values) != 3 for values in schedules):
        raise RuntimeError("Every three-stage schedule must contain three depths.")
    if any(any(depth not in THREE_STAGE_DEPTHS for depth in values) for values in schedules):
        raise RuntimeError("A sampled three-stage schedule contains an invalid depth.")
    if any(len(set(values)) == 1 for values in schedules):
        raise RuntimeError("Constant three-stage schedules duplicate fixed controls.")
    for stage in range(3):
        counts = Counter(values[stage] for values in schedules)
        if counts != Counter({depth: 10 for depth in THREE_STAGE_DEPTHS}):
            raise RuntimeError(f"Unbalanced sampled stage {stage}: {counts}.")

    return tuple(
        DepthPolicy(
            name=f"three_stage_{depths[0]}_{depths[1]}_{depths[2]}",
            family="sampled_three_stage",
            schedule_type="three_stage",
            stage_depths=depths,
        )
        for depths in schedules
    )


def all_policies() -> tuple[DepthPolicy, ...]:
    fixed = tuple(
        DepthPolicy(
            name=f"fixed_depth_{depth}",
            family="fixed",
            schedule_type="fixed",
            fixed_depth=depth,
        )
        for depth in FIXED_DEPTHS
    )
    three_stage = _load_three_stage_policies()
    increment_by_1 = tuple(
        DepthPolicy(
            name=f"increase_1_every_100_start_{start}",
            family="increase_1_every_100",
            schedule_type="increment",
            start_depth=start,
            increment=1,
            interval=100,
        )
        for start in range(4, 14)
    )
    increment_by_2 = tuple(
        DepthPolicy(
            name=f"increase_2_every_100_start_{start}",
            family="increase_2_every_100",
            schedule_type="increment",
            start_depth=start,
            increment=2,
            interval=100,
        )
        for start in range(4, 14)
    )
    increment_by_2_hold = tuple(
        DepthPolicy(
            name=f"increase_2_every_100_start_{start}_hold_after_500",
            family="increase_2_every_100_hold_after_500",
            schedule_type="increment",
            start_depth=start,
            increment=2,
            interval=100,
            stop_after=500,
        )
        for start in range(4, 14)
    )
    increment_by_3 = tuple(
        DepthPolicy(
            name=f"increase_3_every_100_start_{start}",
            family="increase_3_every_100",
            schedule_type="increment",
            start_depth=start,
            increment=3,
            interval=100,
        )
        for start in range(4, 14)
    )
    alternating = tuple(
        DepthPolicy(
            name=f"alternate_plus_4_minus_2_every_100_start_{start}",
            family="alternate_plus_4_minus_2_every_100",
            schedule_type="alternating",
            start_depth=start,
            interval=100,
        )
        for start in range(5, 13)
    )
    cycles = tuple(
        DepthPolicy(
            name="cycle_every_50_" + "_".join(map(str, depths)),
            family="cycle_every_50",
            schedule_type="cycle",
            interval=50,
            cycle_depths=depths,
        )
        for depths in (
            (5, 10, 15),
            (5, 15),
            (3, 6, 10, 15),
            (5, 8, 11, 15),
        )
    )
    policies = (
        fixed
        + three_stage
        + increment_by_1
        + increment_by_2
        + increment_by_2_hold
        + increment_by_3
        + alternating
        + cycles
    )
    expected_by_family = {
        "fixed": 17,
        "sampled_three_stage": 40,
        "increase_1_every_100": 10,
        "increase_2_every_100": 10,
        "increase_2_every_100_hold_after_500": 10,
        "increase_3_every_100": 10,
        "alternate_plus_4_minus_2_every_100": 8,
        "cycle_every_50": 4,
    }
    observed_by_family = Counter(policy.family for policy in policies)
    if len(policies) != 109 or len({policy.name for policy in policies}) != 109:
        raise RuntimeError("Expected exactly 109 uniquely named policies.")
    if observed_by_family != Counter(expected_by_family):
        raise RuntimeError(f"Unexpected policy counts: {observed_by_family}.")
    if any(
        not 1 <= policy.depth(trial) <= MAX_DEPTH
        for policy in policies
        for trial in range(N_TRIALS)
    ):
        raise RuntimeError("A policy leaves the allowed depth range [1, 20].")
    return policies


class DepthPolicyCallback(Callback):
    def __init__(self, policy: DepthPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.observed_transitions: list[list[int]] = [[0, policy.depth(0)]]
        self._last_depth = policy.depth(0)

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = self.policy.depth(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.observed_transitions.append([completed_trials, depth])
            self._last_depth = depth
            print(
                f"[DepthPolicy] completed_trials={completed_trials}, "
                f"max_depth={depth}"
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


def trajectory_path(policy: DepthPolicy, benchmark_seed: int, smac_seed: int) -> Path:
    return (
        OUTPUT_DIRECTORY
        / policy.family
        / f"benchmark_seed_{benchmark_seed}"
        / policy.name
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    policy: DepthPolicy,
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
        and data.get("dimension") == DIMENSION
        and data.get("n_instances") == N_INSTANCES
        and data.get("n_trials") == N_TRIALS
        and np.isclose(
            float(data.get("random_design_probability", -1.0)),
            RANDOM_DESIGN_PROBABILITY,
        )
        and len(data.get("best_regret", ())) == N_TRIALS
    )


def run_policy(
    benchmark_seed: int,
    smac_seed: int,
    policy: DepthPolicy,
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

    def target_function(config, instance: str, seed: int = 0) -> float:
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
    model = ACFacade.get_model(scenario=scenario, max_depth=policy.depth(0))
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    callback = None
    if policy.schedule_type != "fixed":
        callback = DepthPolicyCallback(policy)
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
        "min_samples_leaf": int(model._rf_opts["min_samples_leaf"]),
        "min_samples_split": int(model._rf_opts["min_samples_split"]),
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


def run_seed_pair(benchmark_seed: int, smac_seed: int) -> list[dict[str, Any]]:
    policies = all_policies()
    print(
        f"Running {len(policies)} policies for benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}."
    )
    return [
        run_policy(benchmark_seed, smac_seed, policy)
        for policy in policies
    ]

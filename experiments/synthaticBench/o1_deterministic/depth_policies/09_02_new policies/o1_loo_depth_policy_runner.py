from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback

BENCHMARK_SEEDS = tuple(range(40, 47))
SMAC_SEEDS = tuple(range(10))
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
N_TRIALS = 1000
MIN_DEPTH = 4
MAX_DEPTH = 20
BLOCK_SIZE = 100
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
POLICY_MANIFEST = HERE / "loo_depth_policies.json"
OUTPUT_DIRECTORY = HERE / "smac_output"


@dataclass(frozen=True)
class LooDepthPolicy:
    held_out_landscape: int
    block_depths: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"loo_block_depth_heldout_{self.held_out_landscape}"

    @property
    def training_landscapes(self) -> tuple[int, ...]:
        return tuple(
            seed for seed in BENCHMARK_SEEDS if seed != self.held_out_landscape
        )

    def depth(self, completed_trials: int) -> int:
        if completed_trials < 0:
            raise ValueError("completed_trials must be non-negative.")
        block = min(completed_trials // BLOCK_SIZE, len(self.block_depths) - 1)
        return self.block_depths[block]

    def planned_transitions(self) -> list[list[int]]:
        transitions: list[list[int]] = []
        previous_depth: int | None = None
        for block, depth in enumerate(self.block_depths):
            if depth != previous_depth:
                transitions.append([block * BLOCK_SIZE, depth])
                previous_depth = depth
        return transitions

    def to_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": "loo_best_improving_depth",
            "schedule_type": "100_trial_blocks",
            "held_out_landscape": self.held_out_landscape,
            "training_landscapes": list(self.training_landscapes),
            "block_size": BLOCK_SIZE,
            "block_depths": list(self.block_depths),
            "hold_final_depth": True,
        }


def load_policies() -> tuple[LooDepthPolicy, ...]:
    manifest = json.loads(POLICY_MANIFEST.read_text())
    policies = tuple(
        LooDepthPolicy(
            held_out_landscape=int(item["held_out_landscape"]),
            block_depths=tuple(map(int, item["block_depths"])),
        )
        for item in manifest["policies"]
    )
    if tuple(policy.held_out_landscape for policy in policies) != BENCHMARK_SEEDS:
        raise RuntimeError("The manifest must contain one ordered policy per landscape.")
    for policy in policies:
        if len(policy.block_depths) != N_TRIALS // BLOCK_SIZE:
            raise RuntimeError(f"{policy.name} must have ten block depths.")
        if any(not MIN_DEPTH <= depth <= MAX_DEPTH for depth in policy.block_depths):
            raise RuntimeError(f"{policy.name} contains an invalid depth.")
    return policies


POLICIES = load_policies()
POLICY_BY_HELD_OUT = {policy.held_out_landscape: policy for policy in POLICIES}


class LooDepthPolicyCallback(Callback):
    def __init__(self, policy: LooDepthPolicy) -> None:
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
                f"[{self.policy.name}] completed_trials={completed_trials}, "
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


def trajectory_path(
    policy: LooDepthPolicy,
    benchmark_seed: int,
    smac_seed: int,
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"heldout_{policy.held_out_landscape}"
        / f"benchmark_seed_{benchmark_seed}"
        / policy.name
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(
    policy: LooDepthPolicy,
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
    held_out_landscape: int,
    benchmark_seed: int,
    smac_seed: int,
) -> dict[str, Any]:
    if held_out_landscape not in POLICY_BY_HELD_OUT:
        raise ValueError(f"Unexpected held-out landscape {held_out_landscape}.")
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Unexpected benchmark seed {benchmark_seed}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"Unexpected SMAC seed {smac_seed}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    policy = POLICY_BY_HELD_OUT[held_out_landscape]
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
            OUTPUT_DIRECTORY
            / f"heldout_{held_out_landscape}"
            / f"benchmark_seed_{benchmark_seed}"
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
    callback = LooDepthPolicyCallback(policy)
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        callbacks=[callback],
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
        "policy_family": "loo_best_improving_depth",
        "policy_type": "100_trial_blocks",
        "policy_spec": policy.to_spec(),
        "planned_transitions": policy.planned_transitions(),
        "observed_transitions": callback.observed_transitions,
        "selection_held_out_landscape": held_out_landscape,
        "selection_training_landscapes": list(policy.training_landscapes),
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
        "best_so_far": (
            np.minimum.accumulate(objective_values).astype(float).tolist()
        ),
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
        f"policy={policy.name}, evaluation_landscape={benchmark_seed}, "
        f"smac_seed={smac_seed}, output={output_path}"
    )
    return result


def run_policy_seed_batch(
    held_out_landscape: int,
    benchmark_seed: int,
    smac_seeds: Iterable[int] = SMAC_SEEDS,
) -> list[dict[str, Any]]:
    return [
        run_policy(held_out_landscape, benchmark_seed, smac_seed)
        for smac_seed in smac_seeds
    ]

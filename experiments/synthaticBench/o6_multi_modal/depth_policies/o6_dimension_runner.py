from __future__ import annotations

import copy
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
N_INSTANCES = 10
N_INITIAL_CONFIGS = 10
STAGE_BOUNDARIES = (100, 200, 500)
STAGED_SCHEDULE = (3, 6, 9, 20)
STAGED_POLICY = "staged_depth_3_6_9_20"
LONG_STAGE_BOUNDARIES = (500, 1000, 1500)
LONG_STAGED_SCHEDULE = (5, 10, 15, 20)
LONG_STAGED_POLICY = "staged_depth_5_10_15_20_every_500"
POLICY_SEED = 2026
RANDOM_BLOCK_SIZE = 50
RANDOM_DEPTHS = (3, 9, 20)
RANDOM_POLICY = "random_depth_3_9_20_every_50"
FEEDBACK_DEPTHS = (9, 15, 20, 30)
FEEDBACK_POLICY = "feedback_rank_depth_9_15_20_30"
FEEDBACK_START = 200
FEEDBACK_INTERVAL = 100
FEEDBACK_VALIDATION_FRACTION = 0.2
FEEDBACK_RANK_TOLERANCE = 0.02
FEEDBACK_MIN_IMPROVEMENT = 0.02
FEEDBACK_MIN_DWELL = 200

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


def make_random_schedule(n_trials: int) -> tuple[int, ...]:
    rng = np.random.default_rng(POLICY_SEED)
    n_blocks = (n_trials + RANDOM_BLOCK_SIZE - 1) // RANDOM_BLOCK_SIZE
    schedule = [int(rng.choice(RANDOM_DEPTHS))]
    for _ in range(1, n_blocks):
        choices = [depth for depth in RANDOM_DEPTHS if depth != schedule[-1]]
        schedule.append(int(rng.choice(choices)))
    return tuple(schedule)


def staged_depth(completed_trials: int) -> int:
    if completed_trials < STAGE_BOUNDARIES[0]:
        return STAGED_SCHEDULE[0]
    if completed_trials < STAGE_BOUNDARIES[1]:
        return STAGED_SCHEDULE[1]
    if completed_trials < STAGE_BOUNDARIES[2]:
        return STAGED_SCHEDULE[2]
    return STAGED_SCHEDULE[3]


def long_staged_depth(completed_trials: int) -> int:
    if completed_trials < LONG_STAGE_BOUNDARIES[0]:
        return LONG_STAGED_SCHEDULE[0]
    if completed_trials < LONG_STAGE_BOUNDARIES[1]:
        return LONG_STAGED_SCHEDULE[1]
    if completed_trials < LONG_STAGE_BOUNDARIES[2]:
        return LONG_STAGED_SCHEDULE[2]
    return LONG_STAGED_SCHEDULE[3]


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def rank_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed_ranks = average_ranks(observed)
    predicted_ranks = average_ranks(predicted)
    observed_std = float(observed_ranks.std())
    predicted_std = float(predicted_ranks.std())
    if observed_std == 0 or predicted_std == 0:
        return -1.0
    return float(np.corrcoef(observed_ranks, predicted_ranks)[0, 1])


def choose_feedback_depth(
    scores: dict[int, float],
    current_depth: int,
    completed_trials: int,
    last_switch: int,
) -> tuple[int, bool]:
    best_score = max(scores.values())
    near_best = [
        depth
        for depth in FEEDBACK_DEPTHS
        if scores[depth] >= best_score - FEEDBACK_RANK_TOLERANCE
    ]
    proposed_depth = min(near_best)
    if proposed_depth == current_depth:
        return current_depth, False
    if completed_trials - last_switch < FEEDBACK_MIN_DWELL:
        return current_depth, False

    proposed_score = scores[proposed_depth]
    current_score = scores[current_depth]
    if proposed_depth < current_depth:
        should_switch = (
            proposed_score >= current_score - FEEDBACK_RANK_TOLERANCE
        )
    else:
        should_switch = (
            proposed_score >= current_score + FEEDBACK_MIN_IMPROVEMENT
        )
    return (
        (proposed_depth, True)
        if should_switch
        else (current_depth, False)
    )


class StagedDepthCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = staged_depth(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[StagedDepth] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


class LongStagedDepthCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        depth = long_staged_depth(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[LongStagedDepth] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


class RotatingDepthCallback(Callback):
    def __init__(self, schedule: tuple[int, ...]) -> None:
        super().__init__()
        self.schedule = schedule
        self.transitions: list[tuple[int, int]] = []
        self._last_depth: int | None = None

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        block = min(
            completed_trials // RANDOM_BLOCK_SIZE,
            len(self.schedule) - 1,
        )
        depth = self.schedule[block]
        config_selector._model._rf_opts["max_depth"] = depth
        if depth != self._last_depth:
            self.transitions.append((completed_trials, depth))
            self._last_depth = depth
            print(
                f"[RotatingDepth] completed_trials={completed_trials}, "
                f"max_depth={depth}"
            )


class FeedbackDepthCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.current_depth = FEEDBACK_DEPTHS[0]
        self.last_switch = 0
        self.next_evaluation = FEEDBACK_START
        self.decisions: list[dict[str, Any]] = []
        self.transitions: list[tuple[int, int]] = [(0, self.current_depth)]

    def on_next_configurations_start(self, config_selector) -> None:
        completed_trials = len(config_selector._runhistory)
        config_selector._model._rf_opts["max_depth"] = self.current_depth
        if completed_trials < self.next_evaluation:
            return
        scheduled_trial = self.next_evaluation
        while self.next_evaluation <= completed_trials:
            self.next_evaluation += FEEDBACK_INTERVAL

        encoder = config_selector._runhistory_encoder
        considered = encoder._get_considered_trials()
        ordered = sorted(
            considered.items(),
            key=lambda item: (item[1].starttime, item[1].endtime),
        )
        validation_size = max(
            20,
            int(len(ordered) * FEEDBACK_VALIDATION_FRACTION),
        )
        if len(ordered) - validation_size < 20:
            return

        train_trials = dict(ordered[:-validation_size])
        validation_trials = dict(ordered[-validation_size:])
        X_train, y_train = encoder._build_matrix(
            train_trials,
            store_statistics=True,
        )
        X_validation, y_validation = encoder._build_matrix(
            validation_trials,
            store_statistics=False,
        )

        scores: dict[int, float] = {}
        for depth in FEEDBACK_DEPTHS:
            candidate = copy.deepcopy(config_selector._model)
            candidate._rf_opts["max_depth"] = depth
            candidate.train(X_train, y_train)
            prediction, _ = candidate.predict(X_validation)
            scores[depth] = rank_correlation(y_validation, prediction)

        selected_depth, changed = choose_feedback_depth(
            scores=scores,
            current_depth=self.current_depth,
            completed_trials=completed_trials,
            last_switch=self.last_switch,
        )
        previous_depth = self.current_depth
        if changed:
            self.current_depth = selected_depth
            self.last_switch = completed_trials
            self.transitions.append((completed_trials, selected_depth))
            config_selector._model._rf_opts["max_depth"] = selected_depth
        self.decisions.append(
            {
                "scheduled_trial": scheduled_trial,
                "completed_trials": completed_trials,
                "training_size": len(X_train),
                "validation_size": len(X_validation),
                "scores": {
                    str(depth): float(scores[depth])
                    for depth in FEEDBACK_DEPTHS
                },
                "previous_depth": previous_depth,
                "selected_depth": self.current_depth,
                "changed": changed,
            }
        )
        print(
            f"[FeedbackDepth] completed_trials={completed_trials}, "
            f"scores={scores}, selected_depth={self.current_depth}, "
            f"changed={changed}"
        )


def ordered_trials(runhistory: Any) -> list[tuple[Any, Any]]:
    return sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )


def run_policy(
    policy: str,
    smac_seed: int,
    problem_seed: int,
    dimension: int,
    output_directory: Path,
    n_trials: int,
    n_instances: int = N_INSTANCES,
    min_samples_leaf: int | None = None,
    min_samples_split: int | None = None,
) -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}"
        )
    if policy.startswith("fixed_depth_"):
        initial_depth = int(policy.rsplit("_", 1)[1])
        callback = None
    elif policy == STAGED_POLICY:
        initial_depth = STAGED_SCHEDULE[0]
        callback = StagedDepthCallback()
        random_schedule = None
    elif policy == LONG_STAGED_POLICY:
        initial_depth = LONG_STAGED_SCHEDULE[0]
        callback = LongStagedDepthCallback()
        random_schedule = None
    elif policy == RANDOM_POLICY:
        random_schedule = make_random_schedule(n_trials)
        initial_depth = random_schedule[0]
        callback = RotatingDepthCallback(random_schedule)
    elif policy == FEEDBACK_POLICY:
        initial_depth = FEEDBACK_DEPTHS[0]
        callback = FeedbackDepthCallback()
        random_schedule = None
    else:
        raise ValueError(f"Unknown policy: {policy}")

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
    model_options: dict[str, int] = {"max_depth": initial_depth}
    if min_samples_leaf is not None:
        model_options["min_samples_leaf"] = min_samples_leaf
    if min_samples_split is not None:
        model_options["min_samples_split"] = min_samples_split
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
    if callback is None:
        result["max_depth"] = initial_depth
    elif policy == STAGED_POLICY:
        result.update(
            stage_boundaries=list(STAGE_BOUNDARIES),
            depth_schedule=list(STAGED_SCHEDULE),
            transitions=callback.transitions,
        )
    elif policy == LONG_STAGED_POLICY:
        result.update(
            stage_boundaries=list(LONG_STAGE_BOUNDARIES),
            depth_schedule=list(LONG_STAGED_SCHEDULE),
            transitions=callback.transitions,
        )
    elif policy == FEEDBACK_POLICY:
        result.update(
            candidate_depths=list(FEEDBACK_DEPTHS),
            feedback_start=FEEDBACK_START,
            feedback_interval=FEEDBACK_INTERVAL,
            validation_fraction=FEEDBACK_VALIDATION_FRACTION,
            rank_tolerance=FEEDBACK_RANK_TOLERANCE,
            min_improvement=FEEDBACK_MIN_IMPROVEMENT,
            min_dwell=FEEDBACK_MIN_DWELL,
            transitions=callback.transitions,
            decisions=callback.decisions,
        )
    else:
        result.update(
            block_size=RANDOM_BLOCK_SIZE,
            depth_choices=list(RANDOM_DEPTHS),
            depth_schedule=list(random_schedule),
            policy_seed=POLICY_SEED,
            transitions=callback.transitions,
        )
    output_path = scenario.output_directory / "trajectory.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(
        f"dimension={dimension}, policy={policy}, seed={smac_seed}, "
        f"output={output_path}"
    )
    return result

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

BENCHMARK_SEEDS = tuple(range(40, 47))
SMAC_SEEDS = tuple(range(10))
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
DIMENSION = 10
N_INSTANCES = 10
N_TRIALS = 1000
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

CANDIDATE_DEPTHS = (5, 10, 15, 20)
EXPLORATION_END = 250
FIRST_SELECTION_TRIAL = 300
SECOND_SELECTION_TRIAL = 500
POLICY_NAME = "selection_rotating_5_10_15_20"
POLICY_FAMILY = "selection_rotating"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


def policy_spec() -> dict[str, Any]:
    return {
        "name": POLICY_NAME,
        "family": POLICY_FAMILY,
        "candidate_depths": list(CANDIDATE_DEPTHS),
        "exploration_end": EXPLORATION_END,
        "first_selection_trial": FIRST_SELECTION_TRIAL,
        "second_selection_trial": SECOND_SELECTION_TRIAL,
        "score": "sum_of_positive_incumbent_cost_improvements",
        "tie_break": "smaller_depth_first",
        "selection_delay_policy": "hold_last_exploration_depth",
        "config_selector_retrain_after": 1,
    }


def rank_depths(
    depths: tuple[int, ...],
    improvements: dict[int, list[float]],
) -> list[int]:
    """Rank by cumulative improvement, breaking exact ties by depth."""
    return sorted(depths, key=lambda depth: (-sum(improvements[depth]), depth))


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


def best_average_configuration_cost(runhistory: Any) -> float | None:
    configs = runhistory.get_configs()
    if not configs:
        return None
    costs = [float(runhistory.get_cost(config)) for config in configs]
    finite_costs = [cost for cost in costs if np.isfinite(cost)]
    return min(finite_costs) if finite_costs else None


class SelectionRotatingCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.phase_improvements: dict[str, dict[int, list[float]]] = {
            "exploration": {depth: [] for depth in CANDIDATE_DEPTHS},
            "top_two": {depth: [] for depth in CANDIDATE_DEPTHS},
        }
        self.segments: list[dict[str, Any]] = []
        self.depth_transitions: list[list[int]] = []
        self.first_ranking: list[int] | None = None
        self.selected_two: tuple[int, int] | None = None
        self.second_ranking: list[int] | None = None
        self.selected_depth: int | None = None
        self._rotation_indices = {"exploration": 0, "top_two": 0}
        self._active_segment: dict[str, Any] | None = None
        self._last_depth: int | None = None

    def _finish_segment(
        self,
        completed_trials: int,
        best_cost: float | None,
    ) -> None:
        if self._active_segment is None:
            return
        segment = self._active_segment
        start_best_cost = segment["start_best_cost"]
        improvement = (
            max(0.0, float(start_best_cost) - best_cost)
            if start_best_cost is not None and best_cost is not None
            else 0.0
        )
        record = {
            **segment,
            "end_trial": completed_trials,
            "end_best_cost": best_cost,
            "improvement": improvement,
        }
        self.segments.append(record)
        score_phase = segment["score_phase"]
        if score_phase is not None:
            self.phase_improvements[score_phase][segment["depth"]].append(
                improvement
            )
        self._active_segment = None

    def _start_segment(
        self,
        completed_trials: int,
        best_cost: float | None,
        depth: int,
        phase: str,
        score_phase: str | None,
    ) -> None:
        self._active_segment = {
            "start_trial": completed_trials,
            "start_best_cost": best_cost,
            "depth": depth,
            "phase": phase,
            "score_phase": score_phase,
        }

    def _select_top_two(self, completed_trials: int) -> None:
        if self.selected_two is not None:
            return
        self.first_ranking = rank_depths(
            CANDIDATE_DEPTHS,
            self.phase_improvements["exploration"],
        )
        self.selected_two = tuple(self.first_ranking[:2])
        print(
            f"[SelectionRotating] trial={completed_trials}, "
            f"first_ranking={self.first_ranking}, "
            f"selected_two={self.selected_two}"
        )

    def _select_final_depth(self, completed_trials: int) -> None:
        if self.selected_depth is not None:
            return
        if self.selected_two is None:
            raise RuntimeError("The top-two depths have not been selected.")
        self.second_ranking = rank_depths(
            self.selected_two,
            self.phase_improvements["top_two"],
        )
        self.selected_depth = self.second_ranking[0]
        print(
            f"[SelectionRotating] trial={completed_trials}, "
            f"second_ranking={self.second_ranking}, "
            f"selected_depth={self.selected_depth}"
        )

    def _choose_depth(self, completed_trials: int) -> tuple[int, str, str | None]:
        if completed_trials < EXPLORATION_END:
            index = self._rotation_indices["exploration"]
            depth = CANDIDATE_DEPTHS[index % len(CANDIDATE_DEPTHS)]
            self._rotation_indices["exploration"] += 1
            return depth, "exploration", "exploration"

        if completed_trials < FIRST_SELECTION_TRIAL:
            if self._last_depth is None:
                raise RuntimeError("No exploration depth is available to hold.")
            return self._last_depth, "selection_delay", None

        if self.selected_two is None:
            self._select_top_two(completed_trials)
        assert self.selected_two is not None

        if completed_trials < SECOND_SELECTION_TRIAL:
            index = self._rotation_indices["top_two"]
            depth = self.selected_two[index % len(self.selected_two)]
            self._rotation_indices["top_two"] += 1
            return depth, "top_two", "top_two"

        if self.selected_depth is None:
            self._select_final_depth(completed_trials)
        assert self.selected_depth is not None
        return self.selected_depth, "selected", None

    def on_next_configurations_start(self, config_selector) -> None:
        runhistory = config_selector._runhistory
        completed_trials = len(runhistory)
        # The callback is also invoked once before any data exists, when SMAC
        # samples rather than trains. It is not a surrogate-training event.
        if completed_trials == 0:
            return
        best_cost = best_average_configuration_cost(runhistory)
        self._finish_segment(completed_trials, best_cost)
        depth, phase, score_phase = self._choose_depth(completed_trials)
        config_selector._model._rf_opts["max_depth"] = depth
        self._start_segment(
            completed_trials,
            best_cost,
            depth,
            phase,
            score_phase,
        )
        if depth != self._last_depth:
            self.depth_transitions.append([completed_trials, depth])
            self._last_depth = depth
        print(
            f"[SelectionRotating] completed_trials={completed_trials}, "
            f"phase={phase}, max_depth={depth}"
        )

    def on_tell_end(self, smbo, info, value) -> None:
        """Cut scoring windows exactly at 250, 300, and 500 trials.

        A surrogate-training segment can straddle a requested trial boundary.
        Closing it here attributes only the part inside the scoring window; the
        same trained depth remains active until the next surrogate retraining.
        """
        completed_trials = len(smbo.runhistory)
        if completed_trials not in {
            EXPLORATION_END,
            FIRST_SELECTION_TRIAL,
            SECOND_SELECTION_TRIAL,
        }:
            return
        best_cost = best_average_configuration_cost(smbo.runhistory)
        active = self._active_segment
        self._finish_segment(completed_trials, best_cost)
        if completed_trials == FIRST_SELECTION_TRIAL:
            self._select_top_two(completed_trials)
        elif completed_trials == SECOND_SELECTION_TRIAL:
            self._select_final_depth(completed_trials)
        if active is None or completed_trials == SECOND_SELECTION_TRIAL:
            return
        phase = (
            "selection_delay"
            if completed_trials == EXPLORATION_END
            else "awaiting_top_two_retrain"
        )
        self._start_segment(
            completed_trials,
            best_cost,
            int(active["depth"]),
            phase,
            None,
        )

    def on_end(self, smbo) -> None:
        runhistory = smbo.runhistory
        self._finish_segment(
            len(runhistory),
            best_average_configuration_cost(runhistory),
        )

    def phase_summary(self, phase: str) -> dict[str, dict[str, Any]]:
        return {
            str(depth): {
                "segment_improvements": values,
                "total_improvement": float(sum(values)),
                "average_improvement": (
                    float(np.mean(values)) if values else 0.0
                ),
                "n_segments": len(values),
            }
            for depth, values in self.phase_improvements[phase].items()
        }


def trajectory_path(benchmark_seed: int, smac_seed: int) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"benchmark_seed_{benchmark_seed}"
        / POLICY_NAME
        / str(smac_seed)
        / "trajectory.json"
    )


def trajectory_is_complete(benchmark_seed: int, smac_seed: int) -> bool:
    path = trajectory_path(benchmark_seed, smac_seed)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (
        data.get("experiment_version") == EXPERIMENT_VERSION
        and data.get("policy_spec") == policy_spec()
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


def run_selection_rotating(benchmark_seed: int, smac_seed: int) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Unexpected benchmark seed {benchmark_seed}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"Unexpected SMAC seed {smac_seed}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    if trajectory_is_complete(benchmark_seed, smac_seed):
        print(
            f"Skipping complete benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}."
        )
        return json.loads(trajectory_path(benchmark_seed, smac_seed).read_text())

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
        name=POLICY_NAME,
        output_directory=OUTPUT_DIRECTORY / f"benchmark_seed_{benchmark_seed}",
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    model = ACFacade.get_model(scenario=scenario, max_depth=CANDIDATE_DEPTHS[0])
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    # Rotate at every actual surrogate retraining rather than every eighth new
    # configuration (SMAC's default). Trial-window boundaries are additionally
    # cut in on_tell_end because intensification can allocate a variable number
    # of instance evaluations to each configuration.
    config_selector = ACFacade.get_config_selector(scenario, retrain_after=1)
    callback = SelectionRotatingCallback()
    smac = ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        random_design=random_design,
        config_selector=config_selector,
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
        "policy": POLICY_NAME,
        "policy_family": POLICY_FAMILY,
        "policy_spec": policy_spec(),
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
        "depth_transitions": callback.depth_transitions,
        "segments": callback.segments,
        "exploration_scores": callback.phase_summary("exploration"),
        "first_ranking": callback.first_ranking,
        "selected_two": list(callback.selected_two or ()),
        "top_two_scores": callback.phase_summary("top_two"),
        "second_ranking": callback.second_ranking,
        "selected_depth": callback.selected_depth,
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
        f"policy={POLICY_NAME}, benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}, output={output_path}"
    )
    return result


def run_smac_seed(smac_seed: int) -> list[dict[str, Any]]:
    """Run all seven benchmark landscapes in one of ten Slurm jobs."""
    return [
        run_selection_rotating(benchmark_seed, smac_seed)
        for benchmark_seed in BENCHMARK_SEEDS
    ]

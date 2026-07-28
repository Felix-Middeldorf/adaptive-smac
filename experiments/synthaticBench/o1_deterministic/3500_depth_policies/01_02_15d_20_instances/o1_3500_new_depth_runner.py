from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac import AlgorithmConfigurationFacade as ACFacade
from smac import Scenario
from smac.callback import Callback

BENCHMARK_SEEDS = (40, 42, 44, 46)
SMAC_SEEDS = tuple(range(6))
FIXED_DEPTHS = (18, 23, 26, 29)
EXTENDED_FIXED_DEPTHS = (32, 35, 38, 41, 45, 50)
INSTANCE_SEED = 0
INSTANCE_STD = 2.0
PYTHONHASHSEED = "12345"
DIMENSION = 15
N_INSTANCES = 20
N_TRIALS = 3_500
WINDOW_5_PERCENT = 175
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 1
RANDOM_DESIGN_PROBABILITY = 0.0
AC_DEFAULT_DEPTH = 20
EXPERIMENT_VERSION = 2

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O1-DeterministicObjective.yaml"
)
OUTPUT_DIRECTORY = HERE / "smac_output"


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str
    kind: str
    fixed_depth: int | None = None
    initial_depths: tuple[int, ...] = ()
    step: int | None = None
    strict: bool = False
    add_middle: bool = False
    has_refinement: bool = False

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "g_freeze_on_zero":
            all_zero_rule = "freeze_at_depth_used_before_the_evaluation_window"
        elif self.kind == "early_then_high":
            all_zero_rule = "use_ACFacade_default_depth_20"
        else:
            all_zero_rule = "keep_current_set"
        return {
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "fixed_depth": self.fixed_depth,
            "initial_depths": list(self.initial_depths),
            "step": self.step,
            "strict": self.strict,
            "add_middle": self.add_middle,
            "has_refinement": self.has_refinement,
            "score": "sum_positive_incumbent_improvement_per_suggested_configuration",
            "best_tie_break": "shallower_depth",
            "worst_tie_break": "deeper_depth",
            "all_zero_rule": all_zero_rule,
            "rotation": "random_start_then_next_higher_with_wraparound",
        }


def all_policy_specs() -> tuple[PolicySpec, ...]:
    fixed = tuple(
        PolicySpec(
            name=f"fixed_depth_{depth}",
            family="fixed",
            kind="fixed",
            fixed_depth=depth,
        )
        for depth in FIXED_DEPTHS
    )
    adaptive = (
        PolicySpec("g_freeze_on_zero", "adaptive_g", "g_freeze_on_zero"),
        PolicySpec("g_high_band", "adaptive_g", "g_high_band"),
        PolicySpec(
            "b2_long_refinement",
            "adaptive_b",
            "b2_long_refinement",
            add_middle=True,
            has_refinement=True,
        ),
        PolicySpec("early_then_high", "adaptive_hybrid", "early_then_high"),
    )
    policies = fixed + adaptive
    if len(policies) != 8 or len({policy.name for policy in policies}) != 8:
        raise RuntimeError("The explicit specification contains 8 unique policies.")
    return policies


POLICIES = all_policy_specs()
EXTENDED_FIXED_POLICIES = tuple(
    PolicySpec(
        name=f"fixed_depth_{depth}",
        family="fixed",
        kind="fixed",
        fixed_depth=depth,
    )
    for depth in EXTENDED_FIXED_DEPTHS
)
RUNNABLE_POLICIES = POLICIES + EXTENDED_FIXED_POLICIES
if len({policy.name for policy in RUNNABLE_POLICIES}) != len(RUNNABLE_POLICIES):
    raise RuntimeError("Runnable policy names must be unique.")
POLICY_BY_NAME = {policy.name: policy for policy in RUNNABLE_POLICIES}


def rounded_up_midpoint(depth_1: int, depth_2: int) -> int:
    return (int(depth_1) + int(depth_2) + 1) // 2


def best_ranking(scores: dict[int, float]) -> list[int]:
    """Higher score wins; exact best-score ties prefer shallower depths."""
    return sorted(scores, key=lambda depth: (-scores[depth], depth))


def worst_ranking(scores: dict[int, float]) -> list[int]:
    """Lower score loses; exact worst-score ties remove the deeper depth."""
    return sorted(scores, key=lambda depth: (scores[depth], -depth))


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
    """One deterministic map reused by every policy and seed combination."""
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


def rotation_seed(policy_name: str, benchmark_seed: int, smac_seed: int) -> int:
    payload = f"{policy_name}:{benchmark_seed}:{smac_seed}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class DepthPolicyController:
    """State machine driven by actual random-forest training calls."""

    def __init__(
        self,
        policy: PolicySpec,
        benchmark_seed: int,
        smac_seed: int,
    ) -> None:
        self.policy = policy
        self.rng = np.random.default_rng(
            rotation_seed(policy.name, benchmark_seed, smac_seed)
        )
        self.runhistory: Any | None = None
        self.mode = "default"
        self.depths: tuple[int, ...] = (AC_DEFAULT_DEPTH,)
        self.rotation_index = 0
        self.evaluation_window: str | None = None
        self.window_depths: tuple[int, ...] = ()
        self.window_scheduled_start = 0
        self.window_effective_start = 0
        self.events: list[dict[str, Any]] = []
        self.window_summaries: list[dict[str, Any]] = []
        self.selections: list[dict[str, Any]] = []
        self.policy_transitions: list[dict[str, Any]] = []
        self.depth_transitions: list[dict[str, int]] = []
        self._active_event: dict[str, Any] | None = None
        self._last_depth: int | None = None
        self._stable_depth = AC_DEFAULT_DEPTH
        self._next_expanding_boundary = WINDOW_5_PERCENT
        self._expanding_window_index = 0

        if policy.kind == "fixed":
            assert policy.fixed_depth is not None
            self._set_fixed(policy.fixed_depth)
        elif policy.kind == "expanding":
            self._set_rotation(
                policy.initial_depths,
                evaluation_window="expanding_0",
                scheduled_start=0,
                effective_start=0,
            )
        self._record_transition(
            scheduled_boundary=0,
            effective_completed_trials=0,
            action="initialize",
            previous_mode=None,
            previous_depths=(),
        )

    def attach_runhistory(self, runhistory: Any) -> None:
        self.runhistory = runhistory

    def _set_default(self) -> None:
        self.mode = "default"
        self.depths = (AC_DEFAULT_DEPTH,)
        self.evaluation_window = None
        self.window_depths = ()

    def _set_fixed(self, depth: int) -> None:
        if depth < 1:
            raise ValueError("Random-forest depth must be at least one.")
        self.mode = "fixed"
        self.depths = (int(depth),)
        self._stable_depth = int(depth)
        self.evaluation_window = None
        self.window_depths = ()

    def _set_rotation(
        self,
        depths: tuple[int, ...] | list[int],
        *,
        evaluation_window: str | None,
        scheduled_start: int,
        effective_start: int,
    ) -> None:
        unique = tuple(sorted({max(1, int(depth)) for depth in depths}))
        if not unique:
            raise ValueError("A rotation requires at least one depth.")
        self.mode = "rotate"
        self.depths = unique
        self.rotation_index = int(self.rng.integers(len(unique)))
        self.evaluation_window = evaluation_window
        self.window_depths = unique if evaluation_window is not None else ()
        self.window_scheduled_start = scheduled_start
        self.window_effective_start = effective_start

    def _record_transition(
        self,
        *,
        scheduled_boundary: int,
        effective_completed_trials: int,
        action: str,
        previous_mode: str | None,
        previous_depths: tuple[int, ...],
    ) -> None:
        next_depth = (
            self.depths[self.rotation_index]
            if self.mode == "rotate"
            else self.depths[0]
        )
        self.policy_transitions.append(
            {
                "scheduled_boundary": scheduled_boundary,
                "effective_completed_trials": effective_completed_trials,
                "action": action,
                "previous_mode": previous_mode,
                "previous_depths": list(previous_depths),
                "new_mode": self.mode,
                "new_depths": list(self.depths),
                "rotation_start_depth": int(next_depth),
            }
        )

    def _transition(
        self,
        *,
        scheduled_boundary: int,
        effective_completed_trials: int,
        action: str,
        setter: Callable[[], None],
    ) -> None:
        previous_mode = self.mode
        previous_depths = self.depths
        setter()
        self._record_transition(
            scheduled_boundary=scheduled_boundary,
            effective_completed_trials=effective_completed_trials,
            action=action,
            previous_mode=previous_mode,
            previous_depths=previous_depths,
        )

    def _next_depth(self) -> int:
        if self.mode != "rotate":
            return self.depths[0]
        depth = self.depths[self.rotation_index]
        self.rotation_index = (self.rotation_index + 1) % len(self.depths)
        return depth

    def note_configuration_suggested(self) -> None:
        if self._active_event is not None:
            self._active_event["configurations_suggested"] += 1

    def _close_active_event(
        self,
        completed_trials: int,
        incumbent_cost: float | None,
        *,
        closed_by: str,
    ) -> None:
        if self._active_event is None:
            return
        start = self._active_event["incumbent_cost_before"]
        improvement = (
            max(0.0, float(start) - float(incumbent_cost))
            if start is not None and incumbent_cost is not None
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

    def _window_scores(self, label: str, depths: tuple[int, ...]) -> dict[int, dict[str, Any]]:
        output: dict[int, dict[str, Any]] = {}
        for depth in depths:
            events = [
                event for event in self.events
                if event["evaluation_window"] == label
                and event["active_depth"] == depth
            ]
            improvement = float(sum(event["improvement"] for event in events))
            suggestions = int(
                sum(event["configurations_suggested"] for event in events)
            )
            output[depth] = {
                "improvement_sum": improvement,
                "configurations_suggested": suggestions,
                "score": improvement / suggestions if suggestions else 0.0,
                "surrogate_training_events": len(events),
            }
        return output

    def _summarize_window(
        self,
        scheduled_end: int,
        effective_end: int,
    ) -> dict[str, Any] | None:
        if self.evaluation_window is None:
            return None
        label = self.evaluation_window
        by_depth = self._window_scores(label, self.window_depths)
        scores = {depth: values["score"] for depth, values in by_depth.items()}
        summary = {
            "name": label,
            "scheduled_start_trial": self.window_scheduled_start,
            "scheduled_end_trial": scheduled_end,
            "effective_start_trial": self.window_effective_start,
            "effective_end_trial": effective_end,
            "depths": list(self.window_depths),
            "scores": {str(depth): values for depth, values in by_depth.items()},
            "best_ranking": best_ranking(scores),
            "worst_ranking": worst_ranking(scores),
            "all_scores_zero": all(value == 0.0 for value in scores.values()),
        }
        self.window_summaries.append(summary)
        self.evaluation_window = None
        self.window_depths = ()
        return summary

    @staticmethod
    def _summary_scores(summary: dict[str, Any]) -> dict[int, float]:
        return {
            int(depth): float(values["score"])
            for depth, values in summary["scores"].items()
        }

    def _select_best(
        self,
        summary: dict[str, Any],
        count: int,
        *,
        add_middle: bool = False,
    ) -> tuple[int, ...] | None:
        if summary["all_scores_zero"]:
            return None
        ranking = best_ranking(self._summary_scores(summary))
        selected = ranking[:count]
        if add_middle and len(selected) == 2:
            selected.append(rounded_up_midpoint(selected[0], selected[1]))
        return tuple(sorted(set(selected)))

    def _record_selection(
        self,
        boundary: int,
        completed_trials: int,
        summary: dict[str, Any],
        selected: tuple[int, ...] | None,
        action: str,
    ) -> None:
        self.selections.append(
            {
                "scheduled_boundary": boundary,
                "effective_completed_trials": completed_trials,
                "window": summary["name"],
                "action": action,
                "best_ranking": summary["best_ranking"],
                "worst_ranking": summary["worst_ranking"],
                "selected_depths": list(selected or self.depths),
                "kept_set_because_all_scores_zero": selected is None,
            }
        )

    def _handle_staged_boundary(self, boundary: int, completed_trials: int) -> None:
        base_depths = (5, 10, 15, 20)
        if boundary == 175:
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_primary_rotation",
                setter=lambda: self._set_rotation(
                    base_depths,
                    evaluation_window="primary_5_to_15_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return

        if boundary == 525:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(
                summary, 2, add_middle=self.policy.add_middle
            )
            next_depths = self.depths if selected is None else selected
            self._record_selection(
                boundary, completed_trials, summary, selected, "select_primary_set"
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="keep_primary_set_all_zero" if selected is None else "rotate_selected_primary_set",
                setter=lambda: self._set_rotation(
                    next_depths,
                    evaluation_window="primary_15_to_25_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return

        if boundary == 875:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(summary, 1)
            self._record_selection(
                boundary, completed_trials, summary, selected, "select_primary_depth"
            )
            if selected is None:
                next_depths = self.depths
                setter = lambda: self._set_rotation(
                    next_depths,
                    evaluation_window=None,
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                )
                action = "keep_primary_set_all_zero"
            else:
                setter = lambda: self._set_fixed(selected[0])
                action = "use_selected_primary_depth"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return

        if boundary == 1750:
            current_depth = self._last_depth or self.depths[0]
            neighborhood = (
                max(1, current_depth - 3), current_depth,
                current_depth + 3, current_depth + 6,
            )
            self.selections.append(
                {
                    "scheduled_boundary": boundary,
                    "effective_completed_trials": completed_trials,
                    "action": "construct_refinement_neighborhood",
                    "currD": current_depth,
                    "selected_depths": list(sorted(set(neighborhood))),
                }
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_refinement_rotation",
                setter=lambda: self._set_rotation(
                    neighborhood,
                    evaluation_window="refinement_50_to_60_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return

        if boundary == 2100:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(
                summary, 2, add_middle=self.policy.add_middle
            )
            next_depths = self.depths if selected is None else selected
            self._record_selection(
                boundary, completed_trials, summary, selected, "select_refinement_set"
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="keep_refinement_set_all_zero" if selected is None else "rotate_selected_refinement_set",
                setter=lambda: self._set_rotation(
                    next_depths,
                    evaluation_window="refinement_60_to_70_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return

        if boundary == 2450:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(summary, 1)
            self._record_selection(
                boundary, completed_trials, summary, selected, "select_refinement_depth"
            )
            if selected is None:
                next_depths = self.depths
                setter = lambda: self._set_rotation(
                    next_depths,
                    evaluation_window=None,
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                )
                action = "keep_refinement_set_all_zero"
            else:
                setter = lambda: self._set_fixed(selected[0])
                action = "use_selected_refinement_depth"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return
        raise RuntimeError(f"Unexpected staged boundary {boundary}.")

    def _handle_g_boundary(self, boundary: int, completed_trials: int) -> None:
        if boundary == 700:
            coarse_depths = (
                (20, 23, 26, 29)
                if self.policy.kind == "g_high_band"
                else (10, 14, 18, 22, 25)
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_g_coarse_rotation",
                setter=lambda: self._set_rotation(
                    coarse_depths,
                    evaluation_window="g_20_to_40_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return
        if boundary == 1400:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected_two = self._select_best(summary, 2)
            self._record_selection(
                boundary, completed_trials, summary, selected_two, "select_g_midpoint"
            )
            if selected_two is None:
                if self.policy.kind == "g_freeze_on_zero":
                    fallback = self._stable_depth
                    setter = lambda: self._set_fixed(fallback)
                    action = "freeze_g_at_prior_depth_all_zero"
                else:
                    next_depths = self.depths
                    setter = lambda: self._set_rotation(
                        next_depths,
                        evaluation_window=None,
                        scheduled_start=boundary,
                        effective_start=completed_trials,
                    )
                    action = "keep_g_coarse_set_all_zero"
            else:
                midpoint = rounded_up_midpoint(*selected_two[:2])
                setter = lambda: self._set_fixed(midpoint)
                action = "use_g_coarse_midpoint"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return
        if boundary == 2100:
            current_depth = self._stable_depth
            local_step = 3 if self.policy.kind == "g_high_band" else 4
            neighborhood = (
                max(1, current_depth - local_step),
                current_depth,
                current_depth + local_step,
            )
            self.selections.append(
                {
                    "scheduled_boundary": boundary,
                    "effective_completed_trials": completed_trials,
                    "action": "construct_g_refinement_neighborhood",
                    "currD": current_depth,
                    "selected_depths": list(sorted(set(neighborhood))),
                }
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_g_refinement_rotation",
                setter=lambda: self._set_rotation(
                    neighborhood,
                    evaluation_window="g_60_to_75_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return
        if boundary == 2625:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected_two = self._select_best(summary, 2)
            self._record_selection(
                boundary, completed_trials, summary, selected_two,
                "select_g_final_midpoint",
            )
            if selected_two is None:
                if self.policy.kind == "g_freeze_on_zero":
                    fallback = self._stable_depth
                    setter = lambda: self._set_fixed(fallback)
                    action = "freeze_g_at_prior_depth_all_zero"
                else:
                    next_depths = self.depths
                    setter = lambda: self._set_rotation(
                        next_depths,
                        evaluation_window=None,
                        scheduled_start=boundary,
                        effective_start=completed_trials,
                    )
                    action = "keep_g_refinement_set_all_zero"
            else:
                midpoint = rounded_up_midpoint(*selected_two[:2])
                setter = lambda: self._set_fixed(midpoint)
                action = "use_g_final_midpoint"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return
        raise RuntimeError(f"Unexpected g boundary {boundary}.")

    def _handle_b2_long_boundary(
        self, boundary: int, completed_trials: int
    ) -> None:
        if boundary in (175, 525, 875):
            self._handle_staged_boundary(boundary, completed_trials)
            return
        if boundary == 1400:
            current_depth = self._stable_depth
            neighborhood = (
                max(1, current_depth - 3),
                current_depth,
                current_depth + 3,
                current_depth + 6,
            )
            self.selections.append(
                {
                    "scheduled_boundary": boundary,
                    "effective_completed_trials": completed_trials,
                    "action": "construct_long_refinement_neighborhood",
                    "currD": current_depth,
                    "selected_depths": list(sorted(set(neighborhood))),
                }
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_long_refinement_rotation",
                setter=lambda: self._set_rotation(
                    neighborhood,
                    evaluation_window="long_refinement_40_to_55_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return
        if boundary == 1925:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(summary, 2, add_middle=True)
            next_depths = self.depths if selected is None else selected
            self._record_selection(
                boundary,
                completed_trials,
                summary,
                selected,
                "select_long_refinement_set",
            )
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=(
                    "keep_long_refinement_set_all_zero"
                    if selected is None
                    else "rotate_selected_long_refinement_set"
                ),
                setter=lambda: self._set_rotation(
                    next_depths,
                    evaluation_window="long_refinement_55_to_70_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return
        if boundary == 2450:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected = self._select_best(summary, 1)
            self._record_selection(
                boundary,
                completed_trials,
                summary,
                selected,
                "select_long_refinement_depth",
            )
            if selected is None:
                next_depths = self.depths
                setter = lambda: self._set_rotation(
                    next_depths,
                    evaluation_window=None,
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                )
                action = "keep_long_refinement_set_all_zero"
            else:
                setter = lambda: self._set_fixed(selected[0])
                action = "use_selected_long_refinement_depth"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return
        raise RuntimeError(f"Unexpected b2-long boundary {boundary}.")

    def _handle_early_then_high_boundary(
        self, boundary: int, completed_trials: int
    ) -> None:
        if boundary == 700:
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="start_early_coarse_rotation",
                setter=lambda: self._set_rotation(
                    (10, 14, 18, 22, 25),
                    evaluation_window="early_20_to_40_percent",
                    scheduled_start=boundary,
                    effective_start=completed_trials,
                ),
            )
            return
        if boundary == 1400:
            summary = self._summarize_window(boundary, completed_trials)
            assert summary is not None
            selected_two = self._select_best(summary, 2)
            self._record_selection(
                boundary,
                completed_trials,
                summary,
                selected_two,
                "select_early_midpoint",
            )
            if selected_two is None:
                setter = lambda: self._set_fixed(AC_DEFAULT_DEPTH)
                action = "use_default_depth_early_window_all_zero"
            else:
                midpoint = rounded_up_midpoint(*selected_two[:2])
                setter = lambda: self._set_fixed(midpoint)
                action = "use_early_midpoint"
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action=action,
                setter=setter,
            )
            return
        if boundary == 2100:
            self._transition(
                scheduled_boundary=boundary,
                effective_completed_trials=completed_trials,
                action="force_stable_high_depth_20",
                setter=lambda: self._set_fixed(20),
            )
            return
        raise RuntimeError(f"Unexpected early-then-high boundary {boundary}.")

    def _handle_expanding_boundary(self, boundary: int, completed_trials: int) -> None:
        summary = self._summarize_window(boundary, completed_trials)
        assert summary is not None
        scores = self._summary_scores(summary)
        current = tuple(sorted(self.depths))
        selected: tuple[int, ...] | None = None
        action = "keep_set_all_zero"
        best = best_ranking(scores)[0]
        worst = worst_ranking(scores)[0]

        if not summary["all_scores_zero"]:
            direction: str | None = None
            if self.policy.strict:
                if worst == current[0] or best == current[-1]:
                    direction = "up"
                elif worst == current[-1] or best == current[0]:
                    direction = "down"
            else:
                median = float(np.median(current))
                if worst < median:
                    direction = "up"
                elif worst > median:
                    direction = "down"

            if direction == "up":
                assert self.policy.step is not None
                candidate = current[-1] + self.policy.step
                selected = tuple(sorted((set(current) - {worst}) | {candidate}))
                action = "expand_upward"
            elif direction == "down":
                assert self.policy.step is not None
                candidate = max(1, current[0] - self.policy.step)
                candidate_set = (set(current) - {worst}) | {candidate}
                if len(candidate_set) == len(current):
                    selected = tuple(sorted(candidate_set))
                    action = "expand_downward"
                else:
                    selected = current
                    action = "keep_set_at_depth_floor"
            else:
                selected = current
                action = "keep_set_update_condition_not_met"

        next_depths = current if selected is None else selected
        self._record_selection(
            boundary, completed_trials, summary, selected, action
        )
        self._expanding_window_index += 1
        self._transition(
            scheduled_boundary=boundary,
            effective_completed_trials=completed_trials,
            action=action,
            setter=lambda: self._set_rotation(
                next_depths,
                evaluation_window=f"expanding_{self._expanding_window_index}",
                scheduled_start=boundary,
                effective_start=completed_trials,
            ),
        )

    def _apply_due_boundaries(self, completed_trials: int) -> None:
        if self.policy.kind == "staged":
            boundaries = [175, 525, 875]
            if self.policy.has_refinement:
                boundaries += [1750, 2100, 2450]
            handled = {
                transition["scheduled_boundary"]
                for transition in self.policy_transitions
            }
            for boundary in boundaries:
                if boundary <= completed_trials and boundary not in handled:
                    self._handle_staged_boundary(boundary, completed_trials)
        elif self.policy.kind in ("g_freeze_on_zero", "g_high_band"):
            handled = {
                transition["scheduled_boundary"]
                for transition in self.policy_transitions
            }
            for boundary in (700, 1400, 2100, 2625):
                if boundary <= completed_trials and boundary not in handled:
                    self._handle_g_boundary(boundary, completed_trials)
        elif self.policy.kind == "b2_long_refinement":
            handled = {
                transition["scheduled_boundary"]
                for transition in self.policy_transitions
            }
            for boundary in (175, 525, 875, 1400, 1925, 2450):
                if boundary <= completed_trials and boundary not in handled:
                    self._handle_b2_long_boundary(boundary, completed_trials)
        elif self.policy.kind == "early_then_high":
            handled = {
                transition["scheduled_boundary"]
                for transition in self.policy_transitions
            }
            for boundary in (700, 1400, 2100):
                if boundary <= completed_trials and boundary not in handled:
                    self._handle_early_then_high_boundary(
                        boundary, completed_trials
                    )
        elif self.policy.kind == "expanding":
            while (
                self._next_expanding_boundary < N_TRIALS
                and self._next_expanding_boundary <= completed_trials
            ):
                boundary = self._next_expanding_boundary
                self._handle_expanding_boundary(boundary, completed_trials)
                self._next_expanding_boundary += WINDOW_5_PERCENT

    def before_surrogate_training(
        self,
        completed_trials: int,
        incumbent_cost: float | None,
    ) -> int:
        self._close_active_event(
            completed_trials, incumbent_cost, closed_by="next_surrogate_training"
        )
        self._apply_due_boundaries(completed_trials)
        depth = self._next_depth()
        event_index = len(self.events)
        if depth != self._last_depth:
            self.depth_transitions.append(
                {
                    "completed_trials": completed_trials,
                    "surrogate_training_event": event_index,
                    "depth": depth,
                }
            )
        self._last_depth = depth
        self._active_event = {
            "surrogate_training_event": event_index,
            "completed_trials_before_training": completed_trials,
            "active_depth": depth,
            "incumbent_cost_before": incumbent_cost,
            "configurations_suggested": 0,
            "evaluation_window": self.evaluation_window,
        }
        return depth

    def finalize(self, completed_trials: int, incumbent_cost: float | None) -> None:
        self._close_active_event(
            completed_trials, incumbent_cost, closed_by="optimization_end"
        )
        if self.evaluation_window is not None:
            self._summarize_window(N_TRIALS, completed_trials)

    def export(self) -> dict[str, Any]:
        return {
            "surrogate_training_events": self.events,
            "depth_transitions": self.depth_transitions,
            "evaluation_windows": self.window_summaries,
            "selected_depths": self.selections,
            "policy_transitions": self.policy_transitions,
            "final_mode": self.mode,
            "final_depth_set": list(self.depths),
        }


class PolicyCallback(Callback):
    def __init__(self, controller: DepthPolicyController) -> None:
        super().__init__()
        self.controller = controller

    def on_next_configurations_end(self, config_selector, config) -> None:
        self.controller.note_configuration_suggested()

    def on_end(self, smbo) -> None:
        self.controller.finalize(
            len(smbo.runhistory),
            best_average_configuration_cost(smbo.runhistory),
        )


def install_depth_control(model: Any, controller: DepthPolicyController) -> None:
    """Change depth immediately before each real surrogate-model training call."""
    original_train = model.train

    def controlled_train(X: np.ndarray, Y: np.ndarray) -> Any:
        if controller.runhistory is None:
            raise RuntimeError("The policy controller has no runhistory.")
        depth = controller.before_surrogate_training(
            len(controller.runhistory),
            best_average_configuration_cost(controller.runhistory),
        )
        model._rf_opts["max_depth"] = depth
        return original_train(X, Y)

    model.train = controlled_train


def policy_output_directory(
    policy_name: str, benchmark_seed: int, smac_seed: int
) -> Path:
    return (
        OUTPUT_DIRECTORY
        / f"benchmark_seed_{benchmark_seed}"
        / policy_name
        / str(smac_seed)
    )


def trajectory_path(policy_name: str, benchmark_seed: int, smac_seed: int) -> Path:
    return policy_output_directory(policy_name, benchmark_seed, smac_seed) / "trajectory.json"


def trajectory_is_complete(
    policy: PolicySpec, benchmark_seed: int, smac_seed: int
) -> bool:
    directory = policy_output_directory(policy.name, benchmark_seed, smac_seed)
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
        and data.get("policy_spec") == policy.to_dict()
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


def run_policy(
    benchmark_seed: int,
    smac_seed: int,
    policy_name: str,
) -> dict[str, Any]:
    if benchmark_seed not in BENCHMARK_SEEDS:
        raise ValueError(f"Benchmark seed must be one of {BENCHMARK_SEEDS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"SMAC seed must be one of {SMAC_SEEDS}.")
    if policy_name not in POLICY_BY_NAME:
        raise ValueError(f"Unknown policy {policy_name!r}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(
            f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, got "
            f"{os.environ.get('PYTHONHASHSEED')!r}."
        )
    policy = POLICY_BY_NAME[policy_name]
    if trajectory_is_complete(policy, benchmark_seed, smac_seed):
        print(
            f"Skipping complete policy={policy_name}, "
            f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}."
        )
        return json.loads(
            trajectory_path(policy_name, benchmark_seed, smac_seed).read_text()
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

    scenario = Scenario(
        name=policy.name,
        output_directory=OUTPUT_DIRECTORY / f"benchmark_seed_{benchmark_seed}",
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
    )
    initial_depth = policy.fixed_depth or AC_DEFAULT_DEPTH
    model = ACFacade.get_model(
        scenario=scenario,
        max_depth=initial_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        min_samples_split=MIN_SAMPLES_SPLIT,
    )
    random_design = ACFacade.get_random_design(
        scenario=scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    controller = DepthPolicyController(policy, benchmark_seed, smac_seed)
    callback = PolicyCallback(controller)
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
        "policy": policy.name,
        "policy_family": policy.family,
        "policy_spec": policy.to_dict(),
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
        "config_selector_retrain_after": 8,
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
        f"policy={policy.name}, benchmark_seed={benchmark_seed}, "
        f"smac_seed={smac_seed}, output={output_directory}"
    )
    return result


def run_policy_pack(
    benchmark_seed: int,
    smac_seed: int,
    policy_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    summaries = []
    for policy_name in policy_names:
        result = run_policy(benchmark_seed, smac_seed, policy_name)
        summaries.append(
            {
                "policy": policy_name,
                "benchmark_seed": benchmark_seed,
                "smac_seed": smac_seed,
                "n_trials": result["n_trials"],
                "incumbent_cost": result["incumbent_cost"],
                "trajectory": str(
                    trajectory_path(policy_name, benchmark_seed, smac_seed)
                ),
            }
        )
    return summaries

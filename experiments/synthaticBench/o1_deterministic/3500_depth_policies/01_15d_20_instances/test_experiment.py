from __future__ import annotations

from o1_3500_adaptive_depth_runner import (
    BENCHMARK_SEEDS,
    N_TRIALS,
    POLICIES,
    SMAC_SEEDS,
    DepthPolicyController,
    best_ranking,
    rounded_up_midpoint,
    worst_ranking,
)
from submit_experiment import experiment_packs


def policy(name: str):
    return next(item for item in POLICIES if item.name == name)


def event(window: str, depth: int, improvement: float, suggestions: int):
    return {
        "evaluation_window": window,
        "active_depth": depth,
        "improvement": improvement,
        "configurations_suggested": suggestions,
    }


def test_explicit_policy_list_and_job_packing() -> None:
    assert [item.name for item in POLICIES] == [
        "fixed_depth_5", "fixed_depth_10", "fixed_depth_15", "fixed_depth_20",
        "a1", "a2", "b1", "b2", "c1", "c2", "c3",
        "e1", "e2", "e3", "g",
    ]
    packs = experiment_packs()
    assert len(packs) == 72
    assert all(len(pack.policy_names) == 5 for pack in packs)
    assert sum(len(pack.policy_names) for pack in packs) == (
        len(POLICIES) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    ) == 360


def test_score_is_improvement_sum_per_suggested_configuration() -> None:
    controller = DepthPolicyController(policy("a1"), 40, 0)
    controller.events = [
        event("test", 5, 0.5, 4),
        event("test", 5, 0.25, 2),
        event("test", 10, 1.0, 10),
    ]
    scores = controller._window_scores("test", (5, 10, 15))
    assert scores[5]["improvement_sum"] == 0.75
    assert scores[5]["configurations_suggested"] == 6
    assert scores[5]["score"] == 0.125
    assert scores[10]["score"] == 0.1
    assert scores[15]["score"] == 0.0


def test_tie_breaks_and_midpoint() -> None:
    scores = {5: 1.0, 10: 1.0, 15: 0.0, 20: 0.0}
    assert best_ranking(scores)[:2] == [5, 10]
    assert worst_ranking(scores)[:2] == [20, 15]
    assert rounded_up_midpoint(5, 10) == 8


def test_boundary_is_applied_at_first_training_after_boundary() -> None:
    controller = DepthPolicyController(policy("a1"), 40, 0)
    assert controller.before_surrogate_training(100, 10.0) == 20
    for _ in range(8):
        controller.note_configuration_suggested()
    first_rotating_depth = controller.before_surrogate_training(180, 9.0)
    assert first_rotating_depth in {5, 10, 15, 20}
    transition = controller.policy_transitions[-1]
    assert transition["scheduled_boundary"] == 175
    assert transition["effective_completed_trials"] == 180

    second_rotating_depth = controller.before_surrogate_training(190, 8.5)
    ordered = (5, 10, 15, 20)
    expected = ordered[(ordered.index(first_rotating_depth) + 1) % len(ordered)]
    assert second_rotating_depth == expected


def test_all_zero_scores_keep_the_current_set() -> None:
    controller = DepthPolicyController(policy("a1"), 40, 0)
    controller.before_surrogate_training(180, 10.0)
    controller.before_surrogate_training(530, 10.0)
    assert controller.depths == (5, 10, 15, 20)
    assert controller.policy_transitions[-1]["action"] == "keep_primary_set_all_zero"
    assert controller.selections[-1]["kept_set_because_all_scores_zero"] is True


def test_c_policy_expands_away_from_a_worst_low_depth() -> None:
    controller = DepthPolicyController(policy("c1"), 40, 0)
    controller.events = [
        event("expanding_0", 4, 0.0, 1),
        event("expanding_0", 6, 1.0, 1),
        event("expanding_0", 8, 1.5, 1),
        event("expanding_0", 10, 2.0, 1),
    ]
    controller._handle_expanding_boundary(175, 180)
    assert controller.depths == (6, 8, 10, 12)
    assert controller.policy_transitions[-1]["action"] == "expand_upward"


def test_e_policy_keeps_set_when_strict_condition_is_not_met() -> None:
    controller = DepthPolicyController(policy("e1"), 40, 0)
    controller.events = [
        event("expanding_0", 4, 2.0, 1),
        event("expanding_0", 6, 0.0, 1),
        event("expanding_0", 8, 3.0, 1),
        event("expanding_0", 10, 1.0, 1),
    ]
    controller._handle_expanding_boundary(175, 180)
    assert controller.depths == (4, 6, 8, 10)
    assert (
        controller.policy_transitions[-1]["action"]
        == "keep_set_update_condition_not_met"
    )


def test_budget_boundaries_match_3500_completed_trials() -> None:
    assert N_TRIALS == 3500
    assert [int(N_TRIALS * fraction) for fraction in (0.05, 0.15, 0.25)] == [
        175, 525, 875
    ]
    assert [int(N_TRIALS * fraction) for fraction in (0.5, 0.6, 0.7)] == [
        1750, 2100, 2450
    ]
    assert [int(N_TRIALS * fraction) for fraction in (0.2, 0.4, 0.75)] == [
        700, 1400, 2625
    ]


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")

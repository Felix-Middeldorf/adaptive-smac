from __future__ import annotations

from o1_3500_new_depth_runner import (
    BENCHMARK_SEEDS,
    EXTENDED_FIXED_DEPTHS,
    EXTENDED_FIXED_POLICIES,
    N_TRIALS,
    POLICIES,
    SMAC_SEEDS,
    DepthPolicyController,
    best_ranking,
    rounded_up_midpoint,
    worst_ranking,
)
from submit_experiment import experiment_packs, experiment_runs
from submit_extended_fixed_depths import (
    experiment_packs as extended_experiment_packs,
    experiment_runs as extended_experiment_runs,
)


def policy(name: str):
    return next(item for item in POLICIES if item.name == name)


def event(window: str, depth: int, improvement: float, suggestions: int):
    return {
        "evaluation_window": window,
        "active_depth": depth,
        "improvement": improvement,
        "configurations_suggested": suggestions,
    }


def test_policy_matrix_and_job_packing() -> None:
    assert [item.name for item in POLICIES] == [
        "fixed_depth_18",
        "fixed_depth_23",
        "fixed_depth_26",
        "fixed_depth_29",
        "g_freeze_on_zero",
        "g_high_band",
        "b2_long_refinement",
        "early_then_high",
    ]
    runs = experiment_runs()
    packs = experiment_packs()
    assert len(runs) == 8 * 4 * 6 == 192
    assert len(set(runs)) == 192
    assert len(packs) == 64
    assert all(len(pack.runs) == 3 for pack in packs)
    assert len(packs) <= 80
    assert {run.benchmark_seed for run in runs} == set(BENCHMARK_SEEDS)
    assert {run.smac_seed for run in runs} == set(SMAC_SEEDS)


def test_extended_fixed_depth_matrix_and_job_packing() -> None:
    assert EXTENDED_FIXED_DEPTHS == (32, 35, 38, 41, 45, 50)
    assert [policy.name for policy in EXTENDED_FIXED_POLICIES] == [
        "fixed_depth_32",
        "fixed_depth_35",
        "fixed_depth_38",
        "fixed_depth_41",
        "fixed_depth_45",
        "fixed_depth_50",
    ]
    runs = extended_experiment_runs()
    packs = extended_experiment_packs()
    assert len(runs) == 6 * 4 * 6 == 144
    assert len(set(runs)) == 144
    assert len(packs) == 72
    assert all(len(pack.runs) == 2 for pack in packs)
    assert len(packs) <= 80


def test_score_and_tie_breaks() -> None:
    controller = DepthPolicyController(policy("b2_long_refinement"), 40, 0)
    controller.events = [
        event("test", 20, 0.5, 4),
        event("test", 20, 0.25, 2),
        event("test", 23, 1.0, 10),
    ]
    scores = controller._window_scores("test", (20, 23, 26))
    assert scores[20]["score"] == 0.125
    assert scores[23]["score"] == 0.1
    assert scores[26]["score"] == 0.0
    tied = {5: 1.0, 10: 1.0, 15: 0.0, 20: 0.0}
    assert best_ranking(tied)[:2] == [5, 10]
    assert worst_ranking(tied)[:2] == [20, 15]
    assert rounded_up_midpoint(20, 23) == 22


def test_g_freezes_at_prior_depth_when_window_is_all_zero() -> None:
    controller = DepthPolicyController(policy("g_freeze_on_zero"), 40, 0)
    assert controller.before_surrogate_training(100, 10.0) == 20
    assert controller.before_surrogate_training(705, 10.0) in {10, 14, 18, 22, 25}
    assert controller.before_surrogate_training(1405, 10.0) == 20
    assert controller.mode == "fixed"
    assert controller.policy_transitions[-1]["action"] == "freeze_g_at_prior_depth_all_zero"


def test_g_high_band_and_local_step() -> None:
    controller = DepthPolicyController(policy("g_high_band"), 40, 0)
    assert controller.before_surrogate_training(705, 10.0) in {20, 23, 26, 29}
    controller.events.extend([
        event("g_20_to_40_percent", 20, 1.0, 1),
        event("g_20_to_40_percent", 23, 2.0, 1),
        event("g_20_to_40_percent", 26, 3.0, 1),
        event("g_20_to_40_percent", 29, 4.0, 1),
    ])
    controller.before_surrogate_training(1405, 9.0)
    assert controller.depths == (28,)
    controller.before_surrogate_training(2105, 9.0)
    assert controller.depths == (25, 28, 31)


def test_b2_long_refinement_has_two_15_percent_windows() -> None:
    controller = DepthPolicyController(policy("b2_long_refinement"), 40, 0)
    controller.before_surrogate_training(180, 10.0)
    controller.before_surrogate_training(530, 10.0)
    controller.before_surrogate_training(880, 10.0)
    controller.before_surrogate_training(1405, 10.0)
    assert controller.evaluation_window == "long_refinement_40_to_55_percent"
    assert controller.depths == (17, 20, 23, 26)
    controller.before_surrogate_training(1930, 10.0)
    assert controller.evaluation_window == "long_refinement_55_to_70_percent"
    assert controller.policy_transitions[-1]["scheduled_boundary"] == 1925


def test_early_then_high_forces_depth_20_at_60_percent() -> None:
    controller = DepthPolicyController(policy("early_then_high"), 40, 0)
    controller.before_surrogate_training(705, 10.0)
    controller.before_surrogate_training(1405, 10.0)
    assert controller.depths == (20,)
    assert controller.before_surrogate_training(2105, 10.0) == 20
    assert controller.policy_transitions[-1]["action"] == "force_stable_high_depth_20"


def test_budget_boundaries() -> None:
    assert N_TRIALS == 3500
    assert [int(N_TRIALS * fraction) for fraction in (0.05, 0.15, 0.25)] == [
        175, 525, 875
    ]
    assert [int(N_TRIALS * fraction) for fraction in (0.4, 0.55, 0.7)] == [
        1400, 1925, 2450
    ]
    assert [int(N_TRIALS * fraction) for fraction in (0.2, 0.4, 0.6, 0.75)] == [
        700, 1400, 2100, 2625
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

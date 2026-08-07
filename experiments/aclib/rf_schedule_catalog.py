"""Deterministic catalog of diverse three-phase random-forest schedules."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any


DEPTHS = (5, 10, 15, 20, 30)
SPLIT_SIZES = (2, 4, 8)
FEATURE_RATIOS = (0.3, 0.5, 5.0 / 6.0, 1.0)
CHECKPOINTS = (500, 2_000)
N_SCHEDULES = 50


@dataclass(frozen=True, order=True)
class RFSettings:
    depth: int
    min_samples_split: int
    feature_ratio: float

    def __post_init__(self) -> None:
        if self.depth not in DEPTHS:
            raise ValueError(f"Unsupported depth {self.depth}.")
        if self.min_samples_split not in SPLIT_SIZES:
            raise ValueError(
                f"Unsupported split size {self.min_samples_split}."
            )
        if self.feature_ratio not in FEATURE_RATIOS:
            raise ValueError(
                f"Unsupported feature ratio {self.feature_ratio}."
            )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "depth": self.depth,
            "min_samples_split": self.min_samples_split,
            "feature_ratio": self.feature_ratio,
        }


@dataclass(frozen=True)
class RFSchedule:
    index: int
    phases: tuple[RFSettings, RFSettings, RFSettings]

    @property
    def name(self) -> str:
        return f"schedule_{self.index:02d}"

    def phase_index(self, completed_trials: int) -> int:
        if completed_trials < 0:
            raise ValueError("completed_trials must not be negative.")
        if completed_trials < CHECKPOINTS[0]:
            return 0
        if completed_trials < CHECKPOINTS[1]:
            return 1
        return 2

    def settings_for_trial(self, completed_trials: int) -> RFSettings:
        return self.phases[self.phase_index(completed_trials)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "checkpoints": list(CHECKPOINTS),
            "phases": [
                {
                    "start_trial": start,
                    "end_trial_exclusive": end,
                    **settings.to_dict(),
                }
                for start, end, settings in zip(
                    (0, *CHECKPOINTS),
                    (*CHECKPOINTS, 5_000),
                    self.phases,
                )
            ],
        }


def _setting(
    depth: int,
    split_size: int,
    feature_ratio: float,
) -> RFSettings:
    return RFSettings(depth, split_size, feature_ratio)


def _anchor_schedules() -> list[tuple[RFSettings, RFSettings, RFSettings]]:
    """Structured controls, monotone ramps, and deliberately non-monotone paths."""
    return [
        # Constant controls spanning low to high model flexibility.
        (_setting(5, 8, 0.3),) * 3,
        (_setting(10, 4, 5.0 / 6.0),) * 3,
        (_setting(15, 4, 0.5),) * 3,
        (_setting(30, 2, 1.0),) * 3,
        # Joint complexity ramps in both directions.
        (
            _setting(5, 8, 0.3),
            _setting(15, 4, 0.5),
            _setting(30, 2, 1.0),
        ),
        (
            _setting(30, 2, 1.0),
            _setting(15, 4, 0.5),
            _setting(5, 8, 0.3),
        ),
        # Isolate depth, split-size, and feature-ratio trends.
        (
            _setting(5, 4, 5.0 / 6.0),
            _setting(15, 4, 5.0 / 6.0),
            _setting(30, 4, 5.0 / 6.0),
        ),
        (
            _setting(30, 4, 5.0 / 6.0),
            _setting(15, 4, 5.0 / 6.0),
            _setting(5, 4, 5.0 / 6.0),
        ),
        (
            _setting(15, 8, 5.0 / 6.0),
            _setting(15, 4, 5.0 / 6.0),
            _setting(15, 2, 5.0 / 6.0),
        ),
        (
            _setting(15, 2, 5.0 / 6.0),
            _setting(15, 4, 5.0 / 6.0),
            _setting(15, 8, 5.0 / 6.0),
        ),
        (
            _setting(15, 4, 0.3),
            _setting(15, 4, 0.5),
            _setting(15, 4, 1.0),
        ),
        (
            _setting(15, 4, 1.0),
            _setting(15, 4, 0.5),
            _setting(15, 4, 0.3),
        ),
        # Non-monotone paths test whether temporary flexibility matters.
        (
            _setting(5, 8, 0.3),
            _setting(30, 2, 1.0),
            _setting(10, 4, 0.5),
        ),
        (
            _setting(30, 2, 1.0),
            _setting(5, 8, 0.3),
            _setting(20, 4, 5.0 / 6.0),
        ),
    ]


def _build_schedules() -> tuple[RFSchedule, ...]:
    settings = tuple(
        RFSettings(depth, split_size, feature_ratio)
        for depth, split_size, feature_ratio in product(
            DEPTHS,
            SPLIT_SIZES,
            FEATURE_RATIOS,
        )
    )
    candidates = list(_anchor_schedules())

    # Three coprime modular permutations give every phase broad, balanced
    # coverage while avoiding a simple one-factor-at-a-time correlation.
    for index in range(len(settings) * 3):
        candidates.append(
            (
                settings[(37 * index + 3) % len(settings)],
                settings[(23 * index + 17) % len(settings)],
                settings[(47 * index + 41) % len(settings)],
            )
        )

    selected: list[tuple[RFSettings, RFSettings, RFSettings]] = []
    seen: set[tuple[RFSettings, RFSettings, RFSettings]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        selected.append(candidate)
        if len(selected) == N_SCHEDULES:
            break

    if len(selected) != N_SCHEDULES:
        raise RuntimeError("Could not construct 50 unique RF schedules.")
    schedules = tuple(
        RFSchedule(index=index, phases=phases)
        for index, phases in enumerate(selected)
    )
    for phase in range(3):
        phase_settings = [schedule.phases[phase] for schedule in schedules]
        if {item.depth for item in phase_settings} != set(DEPTHS):
            raise RuntimeError(f"Depth coverage is incomplete in phase {phase}.")
        if {
            item.min_samples_split for item in phase_settings
        } != set(SPLIT_SIZES):
            raise RuntimeError(
                f"Split-size coverage is incomplete in phase {phase}."
            )
        if {
            item.feature_ratio for item in phase_settings
        } != set(FEATURE_RATIOS):
            raise RuntimeError(
                f"Feature-ratio coverage is incomplete in phase {phase}."
            )
    return schedules


SCHEDULES = _build_schedules()


def get_schedule(index: int) -> RFSchedule:
    if not 0 <= index < len(SCHEDULES):
        raise ValueError(f"schedule index must be in [0, {len(SCHEDULES) - 1}].")
    return SCHEDULES[index]

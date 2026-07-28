#!/usr/bin/env python3
"""Summarize speed and trajectory equivalence for the paired comparison."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUT = HERE / "comparison.csv"
MODES = ("original_singleton", "fixed_batched")
SEEDS = (0, 1)


def _phase(summary: dict[str, Any], name: str, key: str) -> float | int:
    return summary.get("phase_summary", {}).get(name, {}).get(key, 0)


def _trial_signature(entry: dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry["config_id"],
        entry["instance"],
        entry["seed"],
        entry["budget"],
        entry["cost"],
        entry["status"],
    )


def _trajectories_match(seed: int) -> bool | None:
    paths = [
        RESULTS / mode / str(seed) / "runhistory.json"
        for mode in MODES
    ]
    if not all(path.is_file() for path in paths):
        return None
    histories = [json.loads(path.read_text()) for path in paths]
    return (
        histories[0]["configs"] == histories[1]["configs"]
        and [
            _trial_signature(entry) for entry in histories[0]["data"]
        ]
        == [
            _trial_signature(entry) for entry in histories[1]["data"]
        ]
    )


def main() -> None:
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    rows = []
    for mode in MODES:
        for seed in SEEDS:
            path = RESULTS / mode / str(seed) / "runtime_summary.json"
            if not path.is_file():
                continue
            summary = json.loads(path.read_text())
            summaries[(mode, seed)] = summary
            rows.append(
                {
                    "mode": mode,
                    "smac_seed": seed,
                    "finished_trials": summary["finished_trials"],
                    "configurations": summary["configurations"],
                    "walltime_seconds": summary["walltime_seconds"],
                    "target_seconds": summary[
                        "target_function_walltime_seconds"
                    ],
                    "get_x_best_seconds": _phase(
                        summary,
                        "get_x_best",
                        "total_seconds_inclusive",
                    ),
                    "get_x_best_calls": _phase(
                        summary,
                        "get_x_best",
                        "calls",
                    ),
                    "xbest_prediction_calls": _phase(
                        summary,
                        "get_x_best_predict_marginalized",
                        "calls",
                    ),
                    "xbest_prediction_inputs": _phase(
                        summary,
                        "get_x_best_predict_marginalized",
                        "total_inputs",
                    ),
                    "model_train_seconds": _phase(
                        summary,
                        "model_train",
                        "total_seconds_inclusive",
                    ),
                }
            )

    if not rows:
        print(f"No completed summaries found below {RESULTS}")
        return

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'mode':20} {'seed':>4} {'trials':>7} {'configs':>7} "
        f"{'wall h':>8} {'xbest h':>8} {'xbest calls':>11} "
        f"{'pred calls':>11} {'pred inputs':>12}"
    )
    for row in rows:
        print(
            f"{row['mode']:20} {row['smac_seed']:4d} "
            f"{row['finished_trials']:7d} {row['configurations']:7d} "
            f"{row['walltime_seconds'] / 3600:8.2f} "
            f"{row['get_x_best_seconds'] / 3600:8.2f} "
            f"{row['get_x_best_calls']:11d} "
            f"{row['xbest_prediction_calls']:11d} "
            f"{row['xbest_prediction_inputs']:12d}"
        )

    print("\nPaired results:")
    for seed in SEEDS:
        original = summaries.get(("original_singleton", seed))
        fixed = summaries.get(("fixed_batched", seed))
        if original is None or fixed is None:
            print(f"seed={seed}: incomplete pair")
            continue
        wall_speedup = (
            original["walltime_seconds"] / fixed["walltime_seconds"]
        )
        original_xbest = float(
            _phase(
                original,
                "get_x_best",
                "total_seconds_inclusive",
            )
        )
        fixed_xbest = float(
            _phase(
                fixed,
                "get_x_best",
                "total_seconds_inclusive",
            )
        )
        xbest_speedup = (
            original_xbest / fixed_xbest
            if fixed_xbest > 0
            else float("inf")
        )
        print(
            f"seed={seed}: wall_speedup={wall_speedup:.2f}x, "
            f"xbest_speedup={xbest_speedup:.2f}x, "
            f"trajectory_identical={_trajectories_match(seed)}"
        )

        original_checkpoints = original.get("checkpoint_summary", {})
        fixed_checkpoints = fixed.get("checkpoint_summary", {})
        for checkpoint in sorted(
            set(original_checkpoints) & set(fixed_checkpoints),
            key=int,
        ):
            speedup = (
                original_checkpoints[checkpoint]["elapsed_seconds"]
                / fixed_checkpoints[checkpoint]["elapsed_seconds"]
            )
            print(f"  trials={checkpoint}: {speedup:.2f}x")

    print(f"\nWrote {OUTPUT}")
    print("Phase timings are inclusive and must not be added together.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Print a compact comparison of completed runtime profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUT = HERE / "runtime_comparison.csv"
PHASES = (
    "acquisition_maximize",
    "local_search_maximize",
    "predict_marginalized",
    "model_train",
    "collect_data",
    "optimizer_save",
    "runhistory_save",
)


def main() -> None:
    rows = []
    for path in sorted(RESULTS.glob("*/*/runtime_summary.json")):
        data = json.loads(path.read_text())
        phases = data["phase_summary"]
        walltime = float(data["walltime_seconds"])
        row = {
            "variant": data["variant"]["name"],
            "smac_seed": data["smac_seed"],
            "finished_trials": data["finished_trials"],
            "configurations": data["configurations"],
            "walltime_seconds": walltime,
            "seconds_per_trial": walltime / max(data["finished_trials"], 1),
            "target_seconds": data["target_function_walltime_seconds"],
        }
        for phase in PHASES:
            timing = phases.get(phase, {})
            row[f"{phase}_seconds"] = timing.get(
                "total_seconds_inclusive", 0.0
            )
            row[f"{phase}_calls"] = timing.get("calls", 0)
        rows.append(row)

    if not rows:
        print(f"No summaries found below {RESULTS}")
        return

    fieldnames = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{'variant':24} {'seed':>4} {'trials':>7} {'configs':>7} "
        f"{'wall h':>8} {'s/trial':>9} {'acq h':>8} {'pred h':>8} "
        f"{'train h':>8} {'save h':>8}"
    )
    for row in rows:
        print(
            f"{row['variant']:24} {row['smac_seed']:4d} "
            f"{row['finished_trials']:7d} {row['configurations']:7d} "
            f"{row['walltime_seconds'] / 3600:8.2f} "
            f"{row['seconds_per_trial']:9.3f} "
            f"{row['acquisition_maximize_seconds'] / 3600:8.2f} "
            f"{row['predict_marginalized_seconds'] / 3600:8.2f} "
            f"{row['model_train_seconds'] / 3600:8.2f} "
            f"{row['optimizer_save_seconds'] / 3600:8.2f}"
        )
    print(f"\nWrote {OUTPUT}")
    print("Phase timings are inclusive and must not be added together.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create compact CSV summaries of runtime-mechanism profiles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUN_CSV = HERE / "runtime_profiles.csv"
PHASE_CSV = HERE / "runtime_phases.csv"
ABLATION_CSV = HERE / "mechanism_ablation_ratios.csv"

PHASES = (
    "ask",
    "tell",
    "collect_data",
    "model_train",
    "get_x_best",
    "acquisition_update",
    "acquisition_maximize",
    "configspace_sample",
    "random_search",
    "local_search_maximize",
    "local_search_walk",
    "neighborhood_generation",
    "predict_marginalized",
    "telemetry_snapshot",
    "telemetry_append",
    "optimizer_save",
    "runhistory_save",
    "intensifier_save",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runs: list[dict[str, Any]] = []
    phases: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*/*/*/runtime_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        walltime = float(data["walltime_seconds"])
        finished = int(data["finished_trials"])
        identity = {
            "benchmark": data["benchmark"],
            "display_name": data["display_name"],
            "variant": data["variant"]["name"],
            "depth": int(data["depth"]),
            "smac_seed": int(data["smac_seed"]),
        }
        row = {
            **identity,
            "state": "complete" if finished >= int(data["n_trials"]) else "partial",
            "finished_trials": finished,
            "configurations": int(data["configurations"]),
            "model_train_calls": int(data.get("model_train_calls", 0)),
            "walltime_seconds": walltime,
            "seconds_per_trial": walltime / max(finished, 1),
            "seconds_per_configuration": walltime
            / max(int(data["configurations"]), 1),
            "seconds_per_model_train": walltime
            / max(int(data.get("model_train_calls", 0)), 1),
            "target_seconds": float(data["target_function_walltime_seconds"]),
            "target_share_percent": 100
            * float(data["target_function_walltime_seconds"])
            / max(walltime, 1e-12),
            "hyperparameters": int(
                data["configuration_space"]["hyperparameters"]
            ),
            "conditions": int(data["configuration_space"]["conditions"]),
            "training_instances": int(data["n_training_instances"]),
            "instance_features": int(data["instance_feature_dimensions"]),
        }
        for phase in PHASES:
            timing = data["phase_summary"].get(phase, {})
            row[f"{phase}_seconds"] = float(
                timing.get("total_seconds_inclusive", 0.0)
            )
            row[f"{phase}_calls"] = int(timing.get("calls", 0))
        runs.append(row)

        for phase, timing in sorted(data["phase_summary"].items()):
            phases.append(
                {
                    **identity,
                    "finished_trials": finished,
                    "phase": phase,
                    "calls": int(timing["calls"]),
                    "total_seconds_inclusive": float(
                        timing["total_seconds_inclusive"]
                    ),
                    "mean_seconds": float(timing["mean_seconds"]),
                    "median_seconds": float(timing["median_seconds"]),
                    "max_seconds": float(timing["max_seconds"]),
                    "total_inputs": int(timing.get("total_inputs", 0)),
                    "total_expanded_inputs": int(
                        timing.get("total_expanded_inputs", 0)
                    ),
                    "walltime_share_inclusive": float(
                        timing["total_seconds_inclusive"]
                    )
                    / max(walltime, 1e-12),
                }
            )
    return runs, phases


def ablation_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity = {
        (
            row["benchmark"],
            row["variant"],
            row["depth"],
            row["smac_seed"],
        ): row
        for row in runs
    }
    comparisons: list[dict[str, Any]] = []
    for row in runs:
        if row["variant"] == "baseline" or row["depth"] != 10:
            continue
        baseline = by_identity.get(
            (row["benchmark"], "baseline", 10, row["smac_seed"])
        )
        if baseline is None:
            continue
        comparison = {
            "benchmark": row["benchmark"],
            "variant": row["variant"],
            "depth": row["depth"],
            "smac_seed": row["smac_seed"],
            "baseline_finished_trials": baseline["finished_trials"],
            "variant_finished_trials": row["finished_trials"],
            "seconds_per_trial_ratio": row["seconds_per_trial"]
            / max(baseline["seconds_per_trial"], 1e-12),
            "seconds_per_configuration_ratio": row[
                "seconds_per_configuration"
            ]
            / max(baseline["seconds_per_configuration"], 1e-12),
            "seconds_per_model_train_ratio": row["seconds_per_model_train"]
            / max(baseline["seconds_per_model_train"], 1e-12),
        }
        for phase in PHASES:
            key = f"{phase}_seconds"
            comparison[f"{phase}_ratio"] = row[key] / max(
                baseline[key], 1e-12
            )
        comparisons.append(comparison)
    return comparisons


def main() -> None:
    runs, phases = load_rows()
    if not runs:
        print(f"No runtime summaries found below {RESULTS}")
        return
    ablations = ablation_rows(runs)
    _write_csv(RUN_CSV, runs)
    _write_csv(PHASE_CSV, phases)
    _write_csv(ABLATION_CSV, ablations)

    print(
        f"{'benchmark':24} {'variant':23} {'d':>3} {'s':>2} "
        f"{'trials':>6} {'configs':>7} {'wall h':>8} {'s/trial':>9} "
        f"{'acq h':>8} {'pred h':>8} {'train h':>8} {'telem h':>8}"
    )
    for row in sorted(
        runs,
        key=lambda item: (
            item["benchmark"],
            item["variant"],
            item["depth"],
            item["smac_seed"],
        ),
    ):
        telemetry_seconds = (
            row["telemetry_snapshot_seconds"]
            + row["telemetry_append_seconds"]
        )
        print(
            f"{row['benchmark']:24} {row['variant']:23} "
            f"{row['depth']:3d} {row['smac_seed']:2d} "
            f"{row['finished_trials']:6d} {row['configurations']:7d} "
            f"{row['walltime_seconds'] / 3600:8.2f} "
            f"{row['seconds_per_trial']:9.2f} "
            f"{row['acquisition_maximize_seconds'] / 3600:8.2f} "
            f"{row['predict_marginalized_seconds'] / 3600:8.2f} "
            f"{row['model_train_seconds'] / 3600:8.2f} "
            f"{telemetry_seconds / 3600:8.2f}"
        )
    print("\nPhase timings are inclusive and must not be added together.")
    print(f"Wrote {RUN_CSV}")
    print(f"Wrote {PHASE_CSV}")
    if ablations:
        print(f"Wrote {ABLATION_CSV}")


if __name__ == "__main__":
    main()

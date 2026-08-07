#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Analyze both completed 50-schedule Clasp experiments."""

from pathlib import Path

from rf_schedule_analytics import (
    analyze_benchmark,
    cross_benchmark_summary,
)


HERE = Path(__file__).resolve().parent
DEFINITIONS = (
    (
        "queens",
        HERE / "clasp_queens_surrogate" / "05_bruteforce_rf_schedules",
        "clasp_queens",
        "Clasp Queens",
    ),
    (
        "weighted",
        HERE
        / "clasp_weighted-sequence_surrogate"
        / "05_bruteforce_rf_schedules",
        "clasp_weighted",
        "Clasp weighted-sequence",
    ),
)


def main() -> None:
    summaries = {}
    for label, directory, benchmark_key, display_name in DEFINITIONS:
        _, summary, _ = analyze_benchmark(
            directory,
            benchmark_key,
            display_name,
        )
        summaries[label] = summary
        print(f"\n{display_name}: top 10 schedules")
        print(
            summary[
                [
                    "schedule",
                    "par10_500_mean",
                    "par10_2000_mean",
                    "par10_5000_mean",
                    "par10_5000_std",
                    "walltime_seconds_mean",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    cross = cross_benchmark_summary(
        summaries,
        HERE / "rf_schedule_analytics",
    )
    print("\nBest cross-benchmark schedules")
    print(cross.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

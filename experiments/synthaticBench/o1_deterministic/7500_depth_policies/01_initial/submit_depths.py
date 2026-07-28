from __future__ import annotations

import argparse
from itertools import product

import submitit

from o1_7500_depth_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    DIMENSION,
    HERE,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_INSTANCES,
    N_TRIALS,
    PYTHONHASHSEED,
    RANDOM_DESIGN_PROBABILITY,
    SMAC_SEEDS,
    run_fixed_depth,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 50
TIMEOUT_MIN = 24 * 60
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def experiment_runs() -> tuple[tuple[int, int, int], ...]:
    runs = tuple(product(BENCHMARK_SEEDS, SMAC_SEEDS, DEPTHS))
    expected = len(BENCHMARK_SEEDS) * len(SMAC_SEEDS) * len(DEPTHS)
    if len(runs) != expected or len(set(runs)) != expected:
        raise RuntimeError("Experiment runs are missing or duplicated.")
    return runs


def print_experiment_summary() -> None:
    runs = experiment_runs()
    print(f"Fixed depths: {DEPTHS}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Trials per run: {N_TRIALS}")
    print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"Slurm jobs: {len(runs)}")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    print(f"Total SMAC trials: {len(runs) * N_TRIALS}")


def submit_jobs() -> None:
    print_experiment_summary()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_7500_D50_I100_Depths_RD10",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for benchmark_seed, smac_seed, depth in experiment_runs():
            job = executor.submit(
                run_fixed_depth,
                benchmark_seed,
                smac_seed,
                depth,
            )
            jobs.append((benchmark_seed, smac_seed, depth, job))
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for benchmark_seed, smac_seed, depth, job in jobs:
        print(
            f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}, "
            f"depth={depth}: {job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the experiment without submitting jobs.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

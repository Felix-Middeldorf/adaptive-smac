from __future__ import annotations

import argparse

import submitit

from o1_selection_rotating_runner import (
    BENCHMARK_SEEDS,
    DIMENSION,
    HERE,
    N_INSTANCES,
    N_TRIALS,
    PYTHONHASHSEED,
    RANDOM_DESIGN_PROBABILITY,
    SMAC_SEEDS,
    policy_spec,
    run_smac_seed,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 10
TIMEOUT_MIN = 720
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def print_experiment_summary() -> None:
    jobs = len(SMAC_SEEDS)
    runs = jobs * len(BENCHMARK_SEEDS)
    print("Experiment: O1 rotating depth selection")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds / Slurm jobs: {SMAC_SEEDS}")
    print(f"Benchmark runs per job: {len(BENCHMARK_SEEDS)}")
    print(f"Total SMAC runs: {runs}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Trials per run: {N_TRIALS}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"Policy: {policy_spec()}")


def submit_jobs() -> None:
    print_experiment_summary()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_SelectionRotating",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
    )
    jobs = []
    with executor.batch():
        for seed in SMAC_SEEDS:
            jobs.append(executor.submit(run_smac_seed, seed))
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for smac_seed, job in zip(SMAC_SEEDS, jobs, strict=True):
        print(f"smac_seed={smac_seed}: {job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the ten-job layout without submitting.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

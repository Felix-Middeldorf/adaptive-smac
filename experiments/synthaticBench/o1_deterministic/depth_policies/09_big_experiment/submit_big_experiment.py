from __future__ import annotations

import argparse
from collections import Counter

import submitit

from o1_big_experiment_runner import (
    BENCHMARK_SEEDS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    all_policies,
    run_seed_pair,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 70
TIMEOUT_MIN = 720
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def print_experiment_summary() -> None:
    policies = all_policies()
    counts = Counter(policy.family for policy in policies)
    jobs = len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Policies per job: {len(policies)}")
    print(f"Policy counts: {dict(counts)}")
    print(f"Slurm jobs: {jobs}")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    print(f"Total SMAC runs: {jobs * len(policies)}")


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
        slurm_job_name="SynthACtic_O1_BigDepthStudy",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for benchmark_seed in BENCHMARK_SEEDS:
            for smac_seed in SMAC_SEEDS:
                job = executor.submit(run_seed_pair, benchmark_seed, smac_seed)
                jobs.append((benchmark_seed, smac_seed, job))
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for benchmark_seed, smac_seed, job in jobs:
        print(
            f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}: "
            f"{job.job_id}"
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


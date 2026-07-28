#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit five default-RF runs using one-hot instance features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

from o1_one_hot_instance_features_runner import (
    BENCHMARK_SEED,
    DIMENSION,
    HERE,
    N_INSTANCES,
    N_TRIALS,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_experiment,
)


SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 90
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


@dataclass(frozen=True)
class RunJob:
    smac_seed: int

    def __call__(self):
        result = run_experiment(self.smac_seed)
        return {
            "benchmark_seed": result["benchmark_seed"],
            "smac_seed": result["smac_seed"],
            "n_trials": result["n_trials"],
            "incumbent_cost": result["incumbent_cost"],
            "pca_applied": result["pca_applied_after_optimization"],
        }

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[RunJob, ...]:
    jobs = tuple(RunJob(smac_seed) for smac_seed in SMAC_SEEDS)
    if len(jobs) != 5 or len(set(jobs)) != 5:
        raise RuntimeError("Expected exactly five unique SMAC runs.")
    return jobs


def print_summary(*, list_jobs: bool = False) -> None:
    jobs = experiment_jobs()
    print("Benchmark: SynthACticBench O1 deterministic")
    print(f"Benchmark seed: {BENCHMARK_SEED}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print("Instance features: 10-dimensional one-hot encoding")
    print("Surrogate: unmodified AlgorithmConfigurationFacade default RF")
    print("Expected PCA: 10 instance features -> 4 components")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"Slurm tasks: {len(jobs)} (one SMAC run per task)")
    print(f"Time limit per task: {TIMEOUT_MIN} minutes")
    print(f"Memory per task: {MEM_GB} GB")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(f"job={index} smac_seed={job.smac_seed}")


def submit_jobs() -> None:
    jobs = experiment_jobs()
    print_summary()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(jobs),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="O1_one_hot_instance_features",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    submitted = []
    with executor.batch():
        for job in jobs:
            submitted.append((job, executor.submit(job)))
    print(f"Submitted {len(submitted)} Slurm tasks.")
    for run_job, submitted_job in submitted:
        print(f"smac_seed={run_job.smac_seed}: {submitted_job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without submitting jobs.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List all runs without submitting jobs.",
    )
    args = parser.parse_args()
    if args.dry_run or args.list_jobs:
        print_summary(list_jobs=args.list_jobs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

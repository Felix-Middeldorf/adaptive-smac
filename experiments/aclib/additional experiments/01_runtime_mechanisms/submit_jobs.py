#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit controlled cross-benchmark ACLib runtime profiling jobs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

from profile_runtime import (
    ACLIB_EXPERIMENT_ROOT,
    BENCHMARKS,
    DEFAULT_DEPTH,
    HERE,
    LOCAL_SMAC_ROOT,
    N_TRIALS,
    REPOSITORY_ROOT,
    VARIANTS,
    run_profile,
)


SMAC_SEEDS = (0, 1, 2)
MECHANISM_SEED = 0
DEPTH_VARIANTS = (5, 30)
MAX_SUBMITTED_JOBS = 80
SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "lect0190"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
CPUS_PER_TASK = 1
ARRAY_PARALLELISM = 40
PYTHONHASHSEED = "0"
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"


@dataclass(frozen=True)
class ProfileJob:
    benchmark_key: str
    variant_name: str
    depth: int
    smac_seed: int

    def __call__(self):
        actual_hash_seed = os.environ.get("PYTHONHASHSEED")
        if actual_hash_seed != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, "
                f"found {actual_hash_seed!r}."
            )
        return run_profile(
            benchmark_key=self.benchmark_key,
            variant_name=self.variant_name,
            depth=self.depth,
            smac_seed=self.smac_seed,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)

    @property
    def label(self) -> str:
        return (
            f"{self.benchmark_key}/{self.variant_name}/"
            f"depth_{self.depth}/seed_{self.smac_seed}"
        )


def baseline_jobs() -> tuple[ProfileJob, ...]:
    return tuple(
        ProfileJob(key, "baseline", DEFAULT_DEPTH, seed)
        for key in BENCHMARKS
        for seed in SMAC_SEEDS
    )


def mechanism_jobs() -> tuple[ProfileJob, ...]:
    return tuple(
        ProfileJob(key, variant, DEFAULT_DEPTH, MECHANISM_SEED)
        for key in BENCHMARKS
        for variant in VARIANTS
        if variant != "baseline"
    )


def depth_jobs() -> tuple[ProfileJob, ...]:
    return tuple(
        ProfileJob(key, "baseline", depth, seed)
        for key in BENCHMARKS
        for depth in DEPTH_VARIANTS
        for seed in SMAC_SEEDS
    )


def jobs_for_suite(suite: str) -> tuple[ProfileJob, ...]:
    suites = {
        "baseline": baseline_jobs(),
        "mechanisms": mechanism_jobs(),
        "depth": depth_jobs(),
    }
    if suite == "all":
        jobs = baseline_jobs() + mechanism_jobs() + depth_jobs()
    else:
        jobs = suites[suite]
    if len(jobs) != len(set(jobs)):
        raise RuntimeError(f"Suite {suite!r} contains duplicate jobs.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Suite {suite!r} has {len(jobs)} jobs; limit is "
            f"{MAX_SUBMITTED_JOBS}."
        )
    return jobs


def slurm_parameters(suite: str) -> dict[str, Any]:
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{EPM_SOURCE}:{ACLIB_EXPERIMENT_ROOT}:"
        f"{HERE}:${{PYTHONPATH:-}}"
    )
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": min(
            ARRAY_PARALLELISM, len(jobs_for_suite(suite))
        ),
        "cpus_per_task": CPUS_PER_TASK,
        "mem_gb": MEM_GB,
        "slurm_job_name": f"ACLib_runtime_mechanisms_{suite}",
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(suite: str, list_jobs: bool = False) -> None:
    jobs = jobs_for_suite(suite)
    print(f"Suite: {suite}")
    print(f"Jobs: {len(jobs)}")
    print(f"Benchmarks: {tuple(BENCHMARKS)}")
    print(f"Trials per job: {N_TRIALS}")
    print(f"Local SMAC: {LOCAL_SMAC_ROOT}")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Slurm job name: {slurm_parameters(suite)['slurm_job_name']}")
    print(f"Array parallelism: {slurm_parameters(suite)['slurm_array_parallelism']}")
    print(f"Memory: {MEM_GB} GB; time limit: {TIMEOUT_MIN} minutes")
    print(f"Worker PYTHONHASHSEED: {PYTHONHASHSEED}")
    print(f"Results: {HERE / 'results'}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(f"job={index:02d} {job.label}")


def submit(suite: str) -> None:
    jobs = jobs_for_suite(suite)
    print_summary(suite)
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / suite),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(**slurm_parameters(suite))
    submitted = []
    with executor.batch():
        for job_spec in jobs:
            submitted.append((job_spec, executor.submit(job_spec)))
    print(f"Submitted {len(submitted)} profiling jobs.")
    for job_spec, job in submitted:
        print(f"{job_spec.label}: {job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("baseline", "mechanisms", "depth", "all"),
        default="baseline",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()
    if args.dry_run or args.list_jobs:
        print_summary(args.suite, list_jobs=args.list_jobs)
    else:
        submit(args.suite)


if __name__ == "__main__":
    main()

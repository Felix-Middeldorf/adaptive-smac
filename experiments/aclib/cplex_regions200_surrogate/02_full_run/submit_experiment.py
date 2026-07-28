#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the 35 full ACLib fixed-depth runs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT_DIRECTORY = HERE.parent
if str(PARENT_DIRECTORY) not in sys.path:
    sys.path.append(str(PARENT_DIRECTORY))

import submitit

from aclib_benchmark import REPOSITORY_ROOT
from run_fixed_depth import (
    DEPTHS,
    N_INSTANCES,
    N_TRIALS,
    OUTPUT_ROOT,
    PCA_COMPONENTS,
    SMAC_SEEDS,
    run_fixed_depth,
)


MAX_SUBMITTED_JOBS = 80
SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 48 * 60
MEM_GB = 6
LOG_DIRECTORY = HERE / "submitit_logs"
SMAC_SITE_PACKAGES = Path(
    "/home/io632776/work/py-envs/py3.12-smac/lib/python3.9/site-packages"
)
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"


@dataclass(frozen=True)
class RunJob:
    depth: int
    smac_seed: int

    def __call__(self):
        return run_fixed_depth(depth=self.depth, smac_seed=self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[RunJob, ...]:
    jobs = tuple(
        RunJob(depth=depth, smac_seed=smac_seed)
        for depth in DEPTHS
        for smac_seed in SMAC_SEEDS
    )
    expected = len(DEPTHS) * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("Job matrix is incomplete or contains duplicates.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(jobs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return jobs


def print_summary(*, list_jobs: bool = False) -> None:
    jobs = experiment_jobs()
    print("Benchmark: ACLib cplex_regions200 surrogate")
    print(f"SMAC surrogate depths: {DEPTHS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Training instances: {N_INSTANCES}")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"PCA components: {PCA_COMPONENTS} (disabled)")
    print(f"Total SMAC runs / Slurm jobs: {len(jobs)}")
    print("Runs per Slurm task: 1")
    print(f"Concurrent-job limit: {MAX_SUBMITTED_JOBS}")
    print(f"Time limit per job: {TIMEOUT_MIN} minutes")
    print(f"Memory per job: {MEM_GB} GB")
    print(f"Output root: {OUTPUT_ROOT}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(f"job={index:02d} depth={job.depth} smac_seed={job.smac_seed}")


def submit_jobs() -> None:
    jobs = experiment_jobs()
    print_summary()
    pythonpath = (
        f"{SMAC_SITE_PACKAGES}:{EPM_SOURCE}:{PARENT_DIRECTORY}:{HERE}:"
        "${PYTHONPATH:-}"
    )
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=10,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(jobs),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="ACLib_CPLEX_R200_full10k",
        slurm_setup=[
            "export PYTHONHASHSEED=0",
            f"export PYTHONPATH={pythonpath}",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    submitted = []
    with executor.batch():
        for job_spec in jobs:
            submitted.append((job_spec, executor.submit(job_spec)))
    print(f"Submitted {len(submitted)} jobs.")
    for job_spec, job in submitted:
        print(
            f"depth={job_spec.depth}, smac_seed={job_spec.smac_seed}: "
            f"{job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()
    if args.dry_run or args.list_jobs:
        print_summary(list_jobs=args.list_jobs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

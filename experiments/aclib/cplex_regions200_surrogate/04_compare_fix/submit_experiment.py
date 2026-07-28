#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the paired original-versus-batched comparison."""

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
from run_compare import (
    LOCAL_SMAC_ROOT,
    MODES,
    N_TRIALS,
    SMAC_SEEDS,
    run_comparison,
)


SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 48 * 60
MEM_GB = 6
LOG_DIRECTORY = HERE / "submitit_logs"
SMAC_DEPENDENCY_SITE_PACKAGES = Path(
    "/home/io632776/work/py-envs/py3.12-smac/lib/python3.9/site-packages"
)
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"


@dataclass(frozen=True)
class ComparisonJob:
    mode: str
    smac_seed: int

    def __call__(self):
        return run_comparison(self.mode, self.smac_seed)


def experiment_jobs() -> tuple[ComparisonJob, ...]:
    return tuple(
        ComparisonJob(mode=mode, smac_seed=seed)
        for mode in MODES
        for seed in SMAC_SEEDS
    )


def print_summary(*, list_jobs: bool = False) -> None:
    jobs = experiment_jobs()
    print(f"Modes: {MODES}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Trials per run: {N_TRIALS}")
    print(f"Local SMAC checkout: {LOCAL_SMAC_ROOT}")
    print(f"Total jobs: {len(jobs)}")
    print(f"Time limit: {TIMEOUT_MIN} minutes")
    print(f"Memory: {MEM_GB} GB")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(
                f"job={index:02d} mode={job.mode} "
                f"smac_seed={job.smac_seed}"
            )


def submit_jobs() -> None:
    jobs = experiment_jobs()
    print_summary()
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{SMAC_DEPENDENCY_SITE_PACKAGES}:"
        f"{EPM_SOURCE}:{PARENT_DIRECTORY}:{HERE}:${{PYTHONPATH:-}}"
    )
    executor = submitit.AutoExecutor(folder=str(LOG_DIRECTORY), cluster="slurm")
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(jobs),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="ACLib_CPLEX_compare_xbest",
        slurm_setup=[
            "export PYTHONHASHSEED=0",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
    )
    submitted = []
    with executor.batch():
        for job_spec in jobs:
            submitted.append((job_spec, executor.submit(job_spec)))
    for job_spec, job in submitted:
        print(
            f"mode={job_spec.mode}, smac_seed={job_spec.smac_seed}: "
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

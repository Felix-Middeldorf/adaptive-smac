from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import submitit

from aclib_benchmark import HERE, REPOSITORY_ROOT
from run_smac import (
    DEFAULT_N_INSTANCES,
    DEFAULT_N_TRIALS,
    DEFAULT_OUTPUT_ROOT,
    run_experiment,
)


DEFAULT_SMAC_SEEDS = tuple(range(10))
MAX_SUBMITTED_JOBS = 80
SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 12 * 60
MEM_GB = 6
LOG_DIRECTORY = HERE / "submitit_logs"
ACLIB_PYTHON = Path("/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python")
SMAC_SITE_PACKAGES = Path(
    "/home/io632776/work/py-envs/py3.12-smac/lib/python3.9/site-packages"
)
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"


@dataclass(frozen=True)
class RunJob:
    smac_seed: int
    n_trials: int
    n_instances: int
    output_root: str
    validate_test: bool

    def __call__(self):
        return run_experiment(
            smac_seed=self.smac_seed,
            n_trials=self.n_trials,
            n_instances=self.n_instances,
            output_root=Path(self.output_root),
            validate_test=self.validate_test,
            overwrite=False,
        )

    def checkpoint(self):
        # SMAC 2.4 reloads matching scenario/runhistory/intensifier state, so a
        # timed-out or requeued task continues instead of overwriting the run.
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs(
    *,
    smac_seeds: tuple[int, ...],
    n_trials: int,
    n_instances: int,
    output_root: Path,
    validate_test: bool,
) -> tuple[RunJob, ...]:
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")
    if not 1 <= n_instances <= DEFAULT_N_INSTANCES:
        raise ValueError(
            f"n_instances must be between 1 and {DEFAULT_N_INSTANCES}."
        )
    if len(smac_seeds) != len(set(smac_seeds)):
        raise ValueError("SMAC seeds must be unique.")
    jobs = tuple(
        RunJob(
            smac_seed=seed,
            n_trials=n_trials,
            n_instances=n_instances,
            output_root=str(output_root.resolve()),
            validate_test=validate_test,
        )
        for seed in smac_seeds
    )
    if not jobs:
        raise ValueError("At least one SMAC seed is required.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise ValueError(
            f"Requested {len(jobs)} jobs, above the limit {MAX_SUBMITTED_JOBS}."
        )
    return jobs


def print_summary(jobs: tuple[RunJob, ...], *, list_jobs: bool = False) -> None:
    first = jobs[0]
    print("Benchmark: ACLib cplex_regions200 surrogate")
    print("Facade: AlgorithmConfigurationFacade")
    print(f"SMAC seeds: {tuple(job.smac_seed for job in jobs)}")
    print(f"Training instances: {first.n_instances}")
    print(f"Completed-trial budget per run: {first.n_trials}")
    print(f"Held-out test validation: {first.validate_test}")
    print(f"SMAC runs / Slurm tasks: {len(jobs)}")
    print(f"Concurrent-job limit: {MAX_SUBMITTED_JOBS}")
    print(f"Time limit per task: {TIMEOUT_MIN} minutes")
    print(f"Memory per task: {MEM_GB} GB")
    print(f"Output root: {first.output_root}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(f"job={index:02d} smac_seed={job.smac_seed}")


def submit_jobs(jobs: tuple[RunJob, ...]) -> None:
    print_summary(jobs)
    pythonpath = f"{SMAC_SITE_PACKAGES}:{EPM_SOURCE}:{HERE}:${{PYTHONPATH:-}}"
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=min(len(jobs), MAX_SUBMITTED_JOBS),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="ACLib_cplex_regions200_SMAC",
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
        print(f"smac_seed={job_spec.smac_seed}: {job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smac-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SMAC_SEEDS,
    )
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--n-instances", type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--validate-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()

    jobs = experiment_jobs(
        smac_seeds=tuple(args.smac_seeds),
        n_trials=args.n_trials,
        n_instances=args.n_instances,
        output_root=args.output_root,
        validate_test=args.validate_test,
    )
    if args.dry_run or args.list_jobs:
        print_summary(jobs, list_jobs=args.list_jobs)
    else:
        if Path(__import__("sys").executable).resolve() != ACLIB_PYTHON.resolve():
            raise RuntimeError(
                "Submit with ./run_in_env.sh submit_experiment.py so Slurm "
                "uses the ACLib Python 3.9 runtime."
            )
        submit_jobs(jobs)


if __name__ == "__main__":
    main()

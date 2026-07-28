from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

from o1_7000_fixed_depth_runner import (
    BENCHMARK_SEEDS,
    DIMENSION,
    FIXED_DEPTHS,
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
MAX_SUBMITTED_JOBS = 70
TIMEOUT_MIN = 6 * 60
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs_7000_fixed"


@dataclass(frozen=True)
class RunJob:
    benchmark_seed: int
    smac_seed: int
    depth: int

    def __call__(self):
        result = run_fixed_depth(
            benchmark_seed=self.benchmark_seed,
            smac_seed=self.smac_seed,
            depth=self.depth,
        )
        return {
            "policy": result["policy"],
            "benchmark_seed": result["benchmark_seed"],
            "smac_seed": result["smac_seed"],
            "n_trials": result["n_trials"],
            "incumbent_cost": result["incumbent_cost"],
        }

    def checkpoint(self):
        # A requeued task skips an already complete run; an incomplete run restarts.
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[RunJob, ...]:
    jobs = tuple(
        RunJob(benchmark_seed, smac_seed, depth)
        for depth in FIXED_DEPTHS
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
    )
    expected = len(FIXED_DEPTHS) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("Experiment jobs are missing or duplicated.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(jobs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return jobs


def print_experiment_summary(*, list_jobs: bool = False) -> None:
    jobs = experiment_jobs()
    print(f"Fixed depths ({len(FIXED_DEPTHS)}): {FIXED_DEPTHS}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"Total SMAC runs: {len(jobs)}")
    print(f"Runs per Slurm task: 1")
    print(f"Slurm array tasks: {len(jobs)} (limit: {MAX_SUBMITTED_JOBS})")
    print(f"Maximum simultaneous tasks: {len(jobs)}")
    print(f"Time limit per task: {TIMEOUT_MIN} minutes")
    print(f"Memory per task: {MEM_GB} GB")
    print(f"Total completed-trial budget: {len(jobs) * N_TRIALS:,}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(
                f"job={index:02d} benchmark_seed={job.benchmark_seed} "
                f"smac_seed={job.smac_seed} depth={job.depth}"
            )


def submit_jobs() -> None:
    run_jobs = experiment_jobs()
    print_experiment_summary()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(run_jobs),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_D15_I20_Fixed7000",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    submitted = []
    with executor.batch():
        for run_job in run_jobs:
            submitted.append((run_job, executor.submit(run_job)))
    print(f"Submitted {len(submitted)} Slurm array tasks.")
    for run_job, job in submitted:
        print(
            f"depth={run_job.depth}, benchmark_seed={run_job.benchmark_seed}, "
            f"smac_seed={run_job.smac_seed}: {job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the experiment without submitting.",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print all 56 tasks (implies --dry-run).",
    )
    args = parser.parse_args()
    if args.dry_run or args.list_jobs:
        print_experiment_summary(list_jobs=args.list_jobs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

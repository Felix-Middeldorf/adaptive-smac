from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

from o1_3500_new_depth_runner import (
    BENCHMARK_SEEDS,
    DIMENSION,
    HERE,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_INSTANCES,
    N_TRIALS,
    POLICIES,
    PYTHONHASHSEED,
    RANDOM_DESIGN_PROBABILITY,
    SMAC_SEEDS,
    run_policy,
)

SLURM_PARTITION = "c23ms"
MAX_SUBMITTED_JOBS = 80
RUNS_PER_JOB = 3
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


@dataclass(frozen=True)
class RunSpec:
    benchmark_seed: int
    smac_seed: int
    policy_name: str


@dataclass(frozen=True)
class RunPack:
    runs: tuple[RunSpec, ...]

    def __call__(self):
        summaries = []
        for run in self.runs:
            result = run_policy(
                run.benchmark_seed,
                run.smac_seed,
                run.policy_name,
            )
            summaries.append(
                {
                    "policy": run.policy_name,
                    "benchmark_seed": run.benchmark_seed,
                    "smac_seed": run.smac_seed,
                    "n_trials": result["n_trials"],
                    "incumbent_cost": result["incumbent_cost"],
                }
            )
        return summaries

    def checkpoint(self):
        # Completed runs are validated and skipped when a requeued pack restarts.
        return submitit.helpers.DelayedSubmission(self)


def experiment_runs() -> tuple[RunSpec, ...]:
    return tuple(
        RunSpec(benchmark_seed, smac_seed, policy.name)
        for policy in POLICIES
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
    )


def experiment_packs() -> tuple[RunPack, ...]:
    runs = experiment_runs()
    packs = tuple(
        RunPack(runs=runs[start:start + RUNS_PER_JOB])
        for start in range(0, len(runs), RUNS_PER_JOB)
    )
    flattened = [run for pack in packs for run in pack.runs]
    if flattened != list(runs) or len(set(flattened)) != len(runs):
        raise RuntimeError("Packed experiment runs are missing, reordered, or duplicated.")
    if len(packs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(packs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return packs


def print_experiment_summary(*, list_packs: bool = False) -> None:
    packs = experiment_packs()
    total_runs = len(experiment_runs())
    print(f"New policies ({len(POLICIES)}): {[policy.name for policy in POLICIES]}")
    print("Reused policies (not submitted): ['fixed_depth_20', 'g']")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"New SMAC runs: {total_runs}")
    print(f"Runs per Slurm task: {RUNS_PER_JOB}")
    print(f"Slurm array tasks: {len(packs)} (limit: {MAX_SUBMITTED_JOBS})")
    print(f"Maximum simultaneous tasks: {len(packs)}")
    print(f"Time limit per task: {TIMEOUT_MIN} minutes")
    print(f"Total completed-trial budget: {total_runs * N_TRIALS:,}")
    if list_packs:
        for index, pack in enumerate(packs):
            descriptions = [
                f"{run.policy_name}/b{run.benchmark_seed}/s{run.smac_seed}"
                for run in pack.runs
            ]
            print(f"pack={index:02d} runs={descriptions}")


def submit_jobs() -> None:
    packs = experiment_packs()
    print_experiment_summary()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(packs),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_D15_I20_NewDepth",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for pack in packs:
            jobs.append((pack, executor.submit(pack)))
    print(f"Submitted {len(jobs)} Slurm array tasks.")
    for pack, job in jobs:
        print(f"runs={pack.runs}: {job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the experiment without submitting.",
    )
    parser.add_argument(
        "--list-packs",
        action="store_true",
        help="Print every packed Slurm task (implies --dry-run).",
    )
    args = parser.parse_args()
    if args.dry_run or args.list_packs:
        print_experiment_summary(list_packs=args.list_packs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

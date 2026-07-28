from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

from o1_3500_uncertainty_runner import (
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
TIMEOUT_MIN = 4 * 60
MAX_SUBMITTED_JOBS = 80
RUNS_PER_JOB = 2
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


@dataclass(frozen=True)
class RunSpec:
    benchmark_seed: int
    smac_seed: int
    depth: int


@dataclass(frozen=True)
class RunPack:
    runs: tuple[RunSpec, ...]

    def __call__(self):
        summaries = []
        for run in self.runs:
            result = run_fixed_depth(
                benchmark_seed=run.benchmark_seed,
                smac_seed=run.smac_seed,
                depth=run.depth,
            )
            summaries.append(
                {
                    "policy": result["policy"],
                    "benchmark_seed": result["benchmark_seed"],
                    "smac_seed": result["smac_seed"],
                    "n_trials": result["n_trials"],
                    "acquisition_selected_proposals": result[
                        "acquisition_selected_proposals"
                    ],
                    "incumbent_cost": result["incumbent_cost"],
                }
            )
        return summaries

    def checkpoint(self):
        # Complete runs in a requeued pack are validated and skipped.
        return submitit.helpers.DelayedSubmission(self)


def experiment_runs() -> tuple[RunSpec, ...]:
    runs = tuple(
        RunSpec(benchmark_seed, smac_seed, depth)
        for depth in DEPTHS
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
    )
    expected = len(DEPTHS) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    if len(runs) != expected or len(set(runs)) != expected:
        raise RuntimeError("Experiment runs are missing or duplicated.")
    return runs


def experiment_packs() -> tuple[RunPack, ...]:
    runs = experiment_runs()
    packs = tuple(
        RunPack(runs=runs[start:start + RUNS_PER_JOB])
        for start in range(0, len(runs), RUNS_PER_JOB)
    )
    flattened = tuple(run for pack in packs for run in pack.runs)
    if flattened != runs or len(set(flattened)) != len(runs):
        raise RuntimeError("Packed runs are missing, reordered, or duplicated.")
    if len(packs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(packs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return packs


def print_experiment_summary(*, list_packs: bool = False) -> None:
    runs = experiment_runs()
    packs = experiment_packs()
    print(f"Fixed depths: {DEPTHS}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"Total SMAC runs: {len(runs)}")
    print(f"Runs per Slurm task: {RUNS_PER_JOB}")
    print(f"Slurm array tasks: {len(packs)} (limit: {MAX_SUBMITTED_JOBS})")
    print(f"Maximum simultaneous tasks: {len(packs)}")
    print(f"Time limit per task: {TIMEOUT_MIN} minutes")
    print(f"Memory per task: {MEM_GB} GB")
    print(f"Total completed-trial budget: {len(runs) * N_TRIALS:,}")
    if list_packs:
        for index, pack in enumerate(packs):
            descriptions = [
                f"b{run.benchmark_seed}/s{run.smac_seed}/d{run.depth}"
                for run in pack.runs
            ]
            print(f"job={index:02d} runs={descriptions}")


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
        slurm_job_name="SynthACtic_O1_D15_I15_Uncertainty",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    submitted = []
    with executor.batch():
        for pack in packs:
            submitted.append((pack, executor.submit(pack)))
    print(f"Submitted {len(submitted)} Slurm array tasks.")
    for pack, job in submitted:
        print(f"runs={pack.runs}: {job.job_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without submitting.",
    )
    parser.add_argument(
        "--list-packs",
        action="store_true",
        help="Print all packed tasks (implies --dry-run).",
    )
    args = parser.parse_args()
    if args.dry_run or args.list_packs:
        print_experiment_summary(list_packs=args.list_packs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

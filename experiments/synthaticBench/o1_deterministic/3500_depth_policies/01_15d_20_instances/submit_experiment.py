from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

from o1_3500_adaptive_depth_runner import (
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
    run_policy_pack,
)

SLURM_PARTITION = "c23ms"
MAX_SUBMITTED_JOBS = 80
POLICIES_PER_JOB = 5
MAX_PARALLEL_JOBS = 72
TIMEOUT_MIN = 12 * 60
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


@dataclass(frozen=True)
class RunPack:
    benchmark_seed: int
    smac_seed: int
    policy_names: tuple[str, ...]

    def __call__(self):
        return run_policy_pack(
            self.benchmark_seed,
            self.smac_seed,
            self.policy_names,
        )

    def checkpoint(self):
        # Completed runs in the pack are detected and skipped after requeue.
        return submitit.helpers.DelayedSubmission(self)


def experiment_packs() -> tuple[RunPack, ...]:
    names = tuple(policy.name for policy in POLICIES)
    packs = tuple(
        RunPack(
            benchmark_seed=benchmark_seed,
            smac_seed=smac_seed,
            policy_names=names[start:start + POLICIES_PER_JOB],
        )
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
        for start in range(0, len(names), POLICIES_PER_JOB)
    )
    flattened = [
        (pack.benchmark_seed, pack.smac_seed, policy_name)
        for pack in packs
        for policy_name in pack.policy_names
    ]
    expected_runs = len(POLICIES) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    if len(flattened) != expected_runs or len(set(flattened)) != expected_runs:
        raise RuntimeError("Packed experiment runs are missing or duplicated.")
    if len(packs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(packs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return packs


def print_experiment_summary(*, list_packs: bool = False) -> None:
    packs = experiment_packs()
    total_runs = len(POLICIES) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)
    print(f"Policies ({len(POLICIES)}): {[policy.name for policy in POLICIES]}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimension: {DIMENSION}")
    print(f"Instances: {N_INSTANCES}")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
    print(f"min_samples_split: {MIN_SAMPLES_SPLIT}")
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print(f"Total SMAC runs: {total_runs}")
    print(f"Policies per Slurm job: {POLICIES_PER_JOB}")
    print(f"Slurm array tasks: {len(packs)} (limit: {MAX_SUBMITTED_JOBS})")
    print(f"Maximum simultaneous tasks: {MAX_PARALLEL_JOBS}")
    print(f"Total completed-trial budget: {total_runs * N_TRIALS:,}")
    if list_packs:
        for index, pack in enumerate(packs):
            print(
                f"pack={index:02d} benchmark_seed={pack.benchmark_seed} "
                f"smac_seed={pack.smac_seed} policies={pack.policy_names}"
            )


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
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_D15_I20_AdaptiveDepth",
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
        print(
            f"benchmark_seed={pack.benchmark_seed}, smac_seed={pack.smac_seed}, "
            f"policies={pack.policy_names}: {job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the complete experiment without submitting.",
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

from __future__ import annotations

import argparse
from collections import Counter

import submitit

from o1_split_policy_runner import (
    BENCHMARK_SEEDS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    all_policies,
    run_policy_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 75
POLICY_SHARDS_PER_SEED_PAIR = 3
TIMEOUT_MIN = 720
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def policy_shards() -> tuple[tuple, ...]:
    policies = all_policies()
    shards = tuple(
        tuple(policies[index::POLICY_SHARDS_PER_SEED_PAIR])
        for index in range(POLICY_SHARDS_PER_SEED_PAIR)
    )
    if sum(map(len, shards)) != len(policies) or any(not shard for shard in shards):
        raise RuntimeError("Invalid policy sharding.")
    if {policy.name for shard in shards for policy in shard} != {
        policy.name for policy in policies
    }:
        raise RuntimeError("Policy sharding lost or duplicated a policy.")
    return shards


def print_experiment_summary() -> None:
    policies = all_policies()
    counts = Counter(policy.family for policy in policies)
    shards = policy_shards()
    jobs = len(BENCHMARK_SEEDS) * len(SMAC_SEEDS) * len(shards)
    runs = len(BENCHMARK_SEEDS) * len(SMAC_SEEDS) * len(policies)
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Policies: {len(policies)}")
    print(f"Policy counts: {dict(counts)}")
    print(f"Policy shards per benchmark/SMAC seed pair: {len(shards)}")
    print(f"SMAC runs per job: {[len(shard) for shard in shards]}")
    print(f"Slurm jobs: {jobs}")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    print(f"Total SMAC runs: {runs}")


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
        slurm_job_name="SynthACtic_O1_SplitPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    shards = policy_shards()
    with executor.batch():
        for benchmark_seed in BENCHMARK_SEEDS:
            for smac_seed in SMAC_SEEDS:
                for shard_index, policies in enumerate(shards):
                    job = executor.submit(
                        run_policy_batch,
                        benchmark_seed,
                        smac_seed,
                        policies,
                    )
                    jobs.append((benchmark_seed, smac_seed, shard_index, policies, job))
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for benchmark_seed, smac_seed, shard_index, policies, job in jobs:
        print(
            f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}, "
            f"shard={shard_index}, policies={len(policies)}: {job.job_id}"
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

from __future__ import annotations

import submitit

from o1_instance_value_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    HERE,
    INSTANCE_STDS,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_depth_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 60
DEPTH_SHARDS = 2
TIMEOUT_MIN = 720
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def depth_shards() -> tuple[tuple[int, ...], ...]:
    shards = tuple(
        tuple(DEPTHS[index::DEPTH_SHARDS])
        for index in range(DEPTH_SHARDS)
    )
    if sorted(depth for shard in shards for depth in shard) != sorted(DEPTHS):
        raise RuntimeError("Depth sharding lost or duplicated a depth.")
    return shards


def print_experiment_summary() -> None:
    shards = depth_shards()
    jobs = (
        len(INSTANCE_STDS)
        * len(BENCHMARK_SEEDS)
        * len(SMAC_SEEDS)
        * len(shards)
    )
    if jobs > 80:
        raise RuntimeError(f"Experiment would submit {jobs} jobs; limit is 80.")
    print(f"Instance standard deviations: {INSTANCE_STDS}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Fixed depths: {DEPTHS}")
    print(f"Depth shards: {shards}")
    print(f"Slurm jobs: {jobs}")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    print(f"Total SMAC runs: {jobs * len(shards[0])}")


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
        slurm_job_name="SynthACtic_O1_InstanceValues_RD0",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for instance_std in INSTANCE_STDS:
            for benchmark_seed in BENCHMARK_SEEDS:
                for smac_seed in SMAC_SEEDS:
                    for shard_index, depths in enumerate(depth_shards()):
                        job = executor.submit(
                            run_depth_batch,
                            instance_std,
                            benchmark_seed,
                            smac_seed,
                            depths,
                        )
                        jobs.append(
                            (instance_std, benchmark_seed, smac_seed, shard_index, job)
                        )
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for instance_std, benchmark_seed, smac_seed, shard_index, job in jobs:
        print(
            f"std={instance_std}, benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}, shard={shard_index}: {job.job_id}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()

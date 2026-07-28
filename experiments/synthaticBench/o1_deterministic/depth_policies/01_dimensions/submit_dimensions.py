from __future__ import annotations

import argparse

import submitit

from o1_dimension_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    DIMENSIONS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_depth_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 80
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
        len(DIMENSIONS)
        * len(BENCHMARK_SEEDS)
        * len(SMAC_SEEDS)
        * len(shards)
    )
    runs = (
        len(DIMENSIONS)
        * len(BENCHMARK_SEEDS)
        * len(SMAC_SEEDS)
        * len(DEPTHS)
    )
    if jobs > 80:
        raise RuntimeError(f"Experiment would submit {jobs} jobs; limit is 80.")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Fixed depths: {DEPTHS}")
    print(f"Depth shards: {shards}")
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
        slurm_job_name="SynthACtic_O1_Dimensions_FixedDepths_RD0",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for dimension in DIMENSIONS:
            for benchmark_seed in BENCHMARK_SEEDS:
                for smac_seed in SMAC_SEEDS:
                    for shard_index, depths in enumerate(depth_shards()):
                        job = executor.submit(
                            run_depth_batch,
                            dimension,
                            benchmark_seed,
                            smac_seed,
                            depths,
                        )
                        jobs.append(
                            (dimension, benchmark_seed, smac_seed, shard_index, job)
                        )
    print(f"Submitted {len(jobs)} Slurm jobs.")
    for dimension, benchmark_seed, smac_seed, shard_index, job in jobs:
        print(
            f"dimension={dimension}, benchmark_seed={benchmark_seed}, "
            f"smac_seed={smac_seed}, shard={shard_index}: {job.job_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()

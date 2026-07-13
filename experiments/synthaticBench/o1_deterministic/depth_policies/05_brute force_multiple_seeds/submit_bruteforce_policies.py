from __future__ import annotations

import itertools

import submitit

from o1_multiseed_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    policy_name,
    run_policy_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 64
TIMEOUT_MIN = 240
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs" / "policies"


def all_schedules() -> list[tuple[int, int, int]]:
    return list(itertools.product(DEPTHS, repeat=3))


def submit_jobs() -> None:
    schedules = all_schedules()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_MultiSeedDepthPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for schedule in schedules:
            job = executor.submit(
                run_policy_batch, schedule, BENCHMARK_SEEDS, SMAC_SEEDS
            )
            jobs.append((schedule, job))
    print(
        f"Submitted {len(jobs)} jobs for "
        f"{len(jobs) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)} SMAC runs."
    )
    for schedule, job in jobs:
        print(f"{policy_name(schedule)}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

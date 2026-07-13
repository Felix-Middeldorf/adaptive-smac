from __future__ import annotations

import itertools

import submitit

from o1_depth_runner import (
    DEPTHS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    policy_name,
    run_policy_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 64
TIMEOUT_MIN = 180
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs" / "policies"


def all_schedules() -> list[tuple[int, int, int]]:
    return list(itertools.product(DEPTHS, repeat=3))


def submit_jobs() -> None:
    schedules = all_schedules()
    if len(schedules) != 64:
        raise RuntimeError(f"Expected 64 schedules, got {len(schedules)}.")
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_20D_NoSizeDepthPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH={HERE}:$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for schedule in schedules:
            jobs.append(
                (schedule, executor.submit(run_policy_batch, schedule, SMAC_SEEDS))
            )
    print(f"Submitted {len(jobs)} jobs for {len(jobs) * len(SMAC_SEEDS)} runs.")
    for schedule, job in jobs:
        print(f"{policy_name(schedule)}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

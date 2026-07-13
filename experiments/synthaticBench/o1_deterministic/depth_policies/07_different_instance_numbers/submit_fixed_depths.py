from __future__ import annotations

import submitit

from o1_instance_count_runner import (
    DEPTHS,
    HERE,
    INSTANCE_COUNTS,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_fixed_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = len(INSTANCE_COUNTS) * len(DEPTHS)
TIMEOUT_MIN = 600
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def submit_jobs() -> None:
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
        slurm_job_name="SynthACtic_O1_InstanceCounts_RD0",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for n_instances in INSTANCE_COUNTS:
            for depth in DEPTHS:
                job = executor.submit(
                    run_fixed_batch,
                    n_instances,
                    depth,
                    SMAC_SEEDS,
                )
                jobs.append((n_instances, depth, job))
    print(
        f"Submitted {len(jobs)} jobs for "
        f"{len(jobs) * len(SMAC_SEEDS)} SMAC runs."
    )
    for n_instances, depth, job in jobs:
        print(f"instances={n_instances}, depth={depth}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

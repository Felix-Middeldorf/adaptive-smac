from __future__ import annotations

import submitit

from o1_increment_policy_runner import (
    BENCHMARK_SEEDS,
    FIXED_DEPTHS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_fixed_batch,
)

SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 240
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs" / "fixed"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(FIXED_DEPTHS),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_IncrementPolicyFixedDepths",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for depth in FIXED_DEPTHS:
            job = executor.submit(
                run_fixed_batch, depth, BENCHMARK_SEEDS, SMAC_SEEDS
            )
            jobs.append((depth, job))
    print(
        f"Submitted {len(jobs)} jobs for "
        f"{len(jobs) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)} SMAC runs."
    )
    for depth, job in jobs:
        print(f"fixed_depth_{depth}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

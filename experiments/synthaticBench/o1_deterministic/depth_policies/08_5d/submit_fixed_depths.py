from __future__ import annotations

import submitit

from o1_5d_fixed_depth_runner import (
    DEPTHS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_fixed_depth,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = len(DEPTHS) * len(SMAC_SEEDS)
TIMEOUT_MIN = 360
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
        slurm_job_name="SynthACtic_O1_5D_FixedDepths_RD0",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for depth in DEPTHS:
            for smac_seed in SMAC_SEEDS:
                job = executor.submit(run_fixed_depth, smac_seed, depth)
                jobs.append((depth, smac_seed, job))
    print(f"Submitted {len(jobs)} jobs for {len(jobs)} SMAC runs.")
    for depth, smac_seed, job in jobs:
        print(f"depth={depth}, smac_seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

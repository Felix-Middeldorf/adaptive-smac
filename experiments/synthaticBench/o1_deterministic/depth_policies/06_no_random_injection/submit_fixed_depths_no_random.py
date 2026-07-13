from __future__ import annotations

import submitit

from o1_fixed_depth_random_design_runner import (
    DEPTHS,
    HERE,
    NO_RANDOM_DESIGN_PROBABILITY,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    random_design_label,
    run_fixed_depth,
)

SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 240
MEM_GB = 4
RANDOM_DESIGN_PROBABILITY = NO_RANDOM_DESIGN_PROBABILITY
LOG_DIRECTORY = HERE / "submitit_logs" / random_design_label(
    RANDOM_DESIGN_PROBABILITY
)


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=len(DEPTHS) * len(SMAC_SEEDS),
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_FixedDepths_RD0",
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
                job = executor.submit(
                    run_fixed_depth,
                    smac_seed,
                    depth,
                    RANDOM_DESIGN_PROBABILITY,
                )
                jobs.append((depth, smac_seed, job))

    print(
        f"Submitted {len(jobs)} jobs for {len(jobs)} SMAC runs with "
        f"random_design_probability={RANDOM_DESIGN_PROBABILITY}."
    )
    for depth, smac_seed, job in jobs:
        print(f"fixed_depth_{depth}, seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

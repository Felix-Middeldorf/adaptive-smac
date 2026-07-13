from __future__ import annotations

import submitit

from o1_increment_policy_runner import (
    BENCHMARK_SEEDS,
    HERE,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    all_policies,
    run_policy_batch,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 47
TIMEOUT_MIN = 240
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs" / "policies"


def submit_jobs() -> None:
    policies = all_policies()
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY), cluster="slurm", slurm_max_num_timeout=1000
    )
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=MAX_PARALLEL_JOBS,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_IncrementDepthPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for policy in policies:
            job = executor.submit(
                run_policy_batch, policy, BENCHMARK_SEEDS, SMAC_SEEDS
            )
            jobs.append((policy, job))
    print(
        f"Submitted {len(jobs)} jobs for "
        f"{len(jobs) * len(BENCHMARK_SEEDS) * len(SMAC_SEEDS)} SMAC runs."
    )
    for policy, job in jobs:
        print(f"{policy.name}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

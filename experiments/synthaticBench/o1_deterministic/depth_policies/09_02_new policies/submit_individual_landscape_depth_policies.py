from __future__ import annotations

import argparse

import submitit

from o1_individual_landscape_depth_policy_runner import (
    HERE,
    POLICIES,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_individual_landscape_policy,
    trajectory_is_complete,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 70
TIMEOUT_MIN = 240
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs_individual_landscape"


def pending_jobs() -> list[tuple[int, int]]:
    return [
        (policy.landscape_seed, smac_seed)
        for policy in POLICIES
        for smac_seed in SMAC_SEEDS
        if not trajectory_is_complete(policy.landscape_seed, smac_seed)
    ]


def print_experiment_summary() -> list[tuple[int, int]]:
    jobs = len(POLICIES) * len(SMAC_SEEDS)
    pending = pending_jobs()
    print(f"Landscape-specific policies: {len(POLICIES)}")
    print(f"SMAC seeds per policy: {len(SMAC_SEEDS)}")
    print(f"Full experiment: {jobs} Slurm jobs, {jobs} SMAC runs")
    print(f"Pending: {len(pending)} Slurm jobs, {len(pending)} SMAC runs")
    print("SMAC runs per job: 1")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    print("Selection and evaluation use the same landscape (in-sample).")
    for policy in POLICIES:
        print(
            f"landscape={policy.landscape_seed}: "
            f"depths={policy.block_depths}"
        )
    if jobs != 70 or MAX_PARALLEL_JOBS > 80:
        raise RuntimeError("Expected exactly 70 jobs within the 80-job limit.")
    return pending


def submit_jobs() -> None:
    pending = print_experiment_summary()
    if not pending:
        print("All 70 trajectories are already complete; nothing to submit.")
        return

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
        slurm_job_name="SynthACtic_O1_InSampleDepthPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for landscape_seed, smac_seed in pending:
            job = executor.submit(
                run_individual_landscape_policy,
                landscape_seed,
                smac_seed,
            )
            jobs.append((landscape_seed, smac_seed, job))

    print(f"Submitted {len(jobs)} Slurm jobs as one batch.")
    for landscape_seed, smac_seed, job in jobs:
        print(
            f"landscape_seed={landscape_seed}, smac_seed={smac_seed}: "
            f"{job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the completion-aware 70-job layout.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import submitit

from o1_loo_depth_policy_runner import (
    BENCHMARK_SEEDS,
    HERE,
    POLICIES,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    run_policy_seed_batch,
    trajectory_is_complete,
)

SLURM_PARTITION = "c23ms"
MAX_PARALLEL_JOBS = 49
TIMEOUT_MIN = 240
MEM_GB = 4
LOG_DIRECTORY = HERE / "submitit_logs"


def pending_jobs() -> list[tuple[int, int, tuple[int, ...]]]:
    pending: list[tuple[int, int, tuple[int, ...]]] = []
    for policy in POLICIES:
        for benchmark_seed in BENCHMARK_SEEDS:
            missing_seeds = tuple(
                smac_seed
                for smac_seed in SMAC_SEEDS
                if not trajectory_is_complete(policy, benchmark_seed, smac_seed)
            )
            if missing_seeds:
                pending.append(
                    (policy.held_out_landscape, benchmark_seed, missing_seeds)
                )
    return pending


def print_experiment_summary() -> list[tuple[int, int, tuple[int, ...]]]:
    jobs = len(POLICIES) * len(BENCHMARK_SEEDS)
    runs = jobs * len(SMAC_SEEDS)
    pending = pending_jobs()
    pending_runs = sum(len(seeds) for _, _, seeds in pending)
    print(f"Held-out policies: {len(POLICIES)}")
    print(f"Evaluation landscapes per policy: {len(BENCHMARK_SEEDS)}")
    print(f"SMAC seeds per policy/landscape job: {len(SMAC_SEEDS)}")
    print(f"Full experiment: {jobs} Slurm jobs, {runs} SMAC runs")
    print(f"Pending: {len(pending)} Slurm jobs, {pending_runs} SMAC runs")
    print(f"Maximum simultaneous jobs: {MAX_PARALLEL_JOBS}")
    for policy in POLICIES:
        print(
            f"heldout={policy.held_out_landscape}: "
            f"depths={policy.block_depths}"
        )
    if jobs > 80 or MAX_PARALLEL_JOBS > 80:
        raise RuntimeError("The experiment must remain within the 80-job limit.")
    return pending


def submit_jobs() -> None:
    pending = print_experiment_summary()
    if not pending:
        print("All 490 trajectories are already complete; nothing to submit.")
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
        slurm_job_name="SynthACtic_O1_LOODepthPolicies",
        slurm_setup=[
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            f"export PYTHONPATH='{HERE}':$PYTHONPATH",
        ],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for held_out, benchmark_seed, smac_seeds in pending:
            job = executor.submit(
                run_policy_seed_batch,
                held_out,
                benchmark_seed,
                smac_seeds,
            )
            jobs.append((held_out, benchmark_seed, smac_seeds, job))

    print(f"Submitted {len(jobs)} Slurm jobs as one batch.")
    for held_out, benchmark_seed, smac_seeds, job in jobs:
        print(
            f"heldout={held_out}, evaluation_landscape={benchmark_seed}, "
            f"smac_seeds={smac_seeds}: {job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the completion-aware 49-job layout.",
    )
    args = parser.parse_args()
    if args.dry_run:
        print_experiment_summary()
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

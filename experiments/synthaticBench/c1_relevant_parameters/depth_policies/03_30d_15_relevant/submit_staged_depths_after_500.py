from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.c1_relevant_parameters.c1_runner import (
    PYTHONHASHSEED,
    run_staged_depth_policy,
)

DEPTH_SCHEDULES = (
    ("staged_depth_3_20_after_500", (3, 20)),
    ("staged_depth_3_15_after_500", (3, 15)),
    ("staged_depth_3_12_after_500", (3, 12)),
)
STAGE_BOUNDARIES = (500,)
SMAC_SEEDS = range(5)
PROBLEM_SEED = 52
N_INSTANCES = 10
N_TRIALS = 1000
DIMENSION = 30
NUM_QUADRATIC = 15
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / "staged_depths_after_500"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=30,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_C1_D30_R15_Staged3After500",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for policy, depth_schedule in DEPTH_SCHEDULES:
            for seed in SMAC_SEEDS:
                job = executor.submit(
                    run_staged_depth_policy,
                    policy,
                    depth_schedule,
                    STAGE_BOUNDARIES,
                    seed,
                    PROBLEM_SEED,
                    OUTPUT_DIRECTORY,
                    N_TRIALS,
                    N_INSTANCES,
                    DIMENSION,
                    NUM_QUADRATIC,
                )
                jobs.append((policy, seed, job))
    for policy, seed, job in jobs:
        print(f"policy={policy}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

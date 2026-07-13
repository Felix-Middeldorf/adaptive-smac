from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.c2_parameter_interactions.c2_runner import (
    PYTHONHASHSEED,
    run_staged_depth_policy,
)

POLICY = "staged_depth_9_15_after_500"
DEPTH_SCHEDULE = (9, 15)
STAGE_BOUNDARIES = (500,)
SMAC_SEEDS = range(10)
PROBLEM_SEED = 52
N_INSTANCES = 10
N_TRIALS = 1000
DIMENSION = 10
FUNCTION_NAME = "griewank"
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / POLICY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=10,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_C2_Griewank_9_15",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for seed in SMAC_SEEDS:
            job = executor.submit(
                run_staged_depth_policy,
                POLICY,
                DEPTH_SCHEDULE,
                STAGE_BOUNDARIES,
                seed,
                PROBLEM_SEED,
                OUTPUT_DIRECTORY,
                N_TRIALS,
                N_INSTANCES,
                DIMENSION,
                FUNCTION_NAME,
            )
            jobs.append((seed, job))
    for seed, job in jobs:
        print(f"seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

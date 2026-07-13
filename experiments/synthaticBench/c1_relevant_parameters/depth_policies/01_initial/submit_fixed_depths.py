from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.c1_relevant_parameters.c1_runner import (
    run_depth_policy,
)

DEPTHS = (3, 6, 9, 12, 15, 20)
SMAC_SEEDS = range(5)
PROBLEM_SEED = 52
N_INSTANCES = 10
N_TRIALS = 1000
PYTHONHASHSEED = "12345"
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / "fixed_depths"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=30,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_C1_Depths",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for depth in DEPTHS:
            for seed in SMAC_SEEDS:
                job = executor.submit(
                    run_depth_policy,
                    depth,
                    seed,
                    PROBLEM_SEED,
                    OUTPUT_DIRECTORY,
                    N_TRIALS,
                    N_INSTANCES,
                )
                jobs.append((depth, seed, job))
    for depth, seed, job in jobs:
        print(f"depth={depth}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

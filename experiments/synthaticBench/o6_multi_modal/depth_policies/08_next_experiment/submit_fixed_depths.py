from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.o6_multi_modal.depth_policies.o6_dimension_runner import (
    run_policy,
)

DEPTHS = (9, 15, 20, 30)
SMAC_SEEDS = range(5)
DIMENSION = 20
N_INSTANCES = 15
PROBLEM_SEED = 52
N_TRIALS = 1500
PYTHONHASHSEED = "12345"
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / "fixed"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O6_D20_I15_Feedback_Fixed",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for depth in DEPTHS:
            for seed in SMAC_SEEDS:
                job = executor.submit(
                    run_policy,
                    f"fixed_depth_{depth}",
                    seed,
                    PROBLEM_SEED,
                    DIMENSION,
                    OUTPUT_DIRECTORY,
                    N_TRIALS,
                    N_INSTANCES,
                )
                jobs.append((depth, seed, job))
    for depth, seed, job in jobs:
        print(f"depth={depth}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

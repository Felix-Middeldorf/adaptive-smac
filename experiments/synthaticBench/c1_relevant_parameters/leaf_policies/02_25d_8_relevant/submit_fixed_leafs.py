from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.c1_relevant_parameters.c1_runner import (
    run_leaf_policy,
)

LEAF_SIZES = (1, 2, 3, 4, 5)
SMAC_SEEDS = range(5)
PROBLEM_SEED = 52
N_INSTANCES = 10
N_TRIALS = 1000
DIMENSION = 25
NUM_QUADRATIC = 8
PYTHONHASHSEED = "12345"
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs" / "fixed_leafs"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=20,
        slurm_partition="c23ms",
        slurm_array_parallelism=25,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_C1_D25_R8_Leafs",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for leaf_size in LEAF_SIZES:
            for seed in SMAC_SEEDS:
                job = executor.submit(
                    run_leaf_policy,
                    leaf_size,
                    seed,
                    PROBLEM_SEED,
                    OUTPUT_DIRECTORY,
                    N_TRIALS,
                    N_INSTANCES,
                    DIMENSION,
                    NUM_QUADRATIC,
                )
                jobs.append((leaf_size, seed, job))
    for leaf_size, seed, job in jobs:
        print(f"leaf={leaf_size}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

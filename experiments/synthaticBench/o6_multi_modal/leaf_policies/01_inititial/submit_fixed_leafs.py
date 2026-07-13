from pathlib import Path
import sys

import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.synthaticBench.o6_multi_modal.leaf_policies.o6_leaf_runner import (
    run_leaf_policy,
)

LEAF_SIZES = (1, 2, 3, 4, 5)
SMAC_SEEDS = range(5)
DIMENSION = 10
N_INSTANCES = 10
PROBLEM_SEED = 52
N_TRIALS = 1000
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
        slurm_array_parallelism=25,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O6_Leaf_D10_Fixed",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for leaf_size in LEAF_SIZES:
            for seed in SMAC_SEEDS:
                job = executor.submit(
                    run_leaf_policy,
                    f"fixed_leaf_{leaf_size}",
                    seed,
                    PROBLEM_SEED,
                    DIMENSION,
                    OUTPUT_DIRECTORY,
                    N_TRIALS,
                    N_INSTANCES,
                )
                jobs.append((leaf_size, seed, job))
    for leaf_size, seed, job in jobs:
        print(f"leaf={leaf_size}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

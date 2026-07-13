from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import submitit

SMAC_SEEDS = range(5)
LEAF_SIZES = (1, 2, 3)
PROBLEM_SEED = 53
MIN_SAMPLES_SPLIT = 1
N_TRIALS = 1500
PYTHONHASHSEED = "12345"
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs" / "fixed"

sys.path.insert(0, str(REPOSITORY_ROOT))
BASE = importlib.import_module(
    "experiments.synthaticBench.o1_deterministic.leaf_policies."
    "02_leafs_seeds.submit_fixed_policies"
)


def run_fixed_policy(
    smac_seed: int,
    min_samples_leaf: int,
) -> dict[str, Any]:
    BASE.PROBLEM_SEED = PROBLEM_SEED
    BASE.MIN_SAMPLES_SPLIT = MIN_SAMPLES_SPLIT
    BASE.N_TRIALS = N_TRIALS
    BASE.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    return BASE.run_fixed_policy(smac_seed, min_samples_leaf)


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=15,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O1_1500_Fixed",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for leaf_size in LEAF_SIZES:
            for smac_seed in SMAC_SEEDS:
                jobs.append(
                    (
                        leaf_size,
                        smac_seed,
                        executor.submit(
                            run_fixed_policy,
                            smac_seed,
                            leaf_size,
                        ),
                    )
                )
    print(f"Submitted {len(jobs)} fixed-policy jobs:")
    for leaf_size, smac_seed, job in jobs:
        print(f"leaf={leaf_size}, seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

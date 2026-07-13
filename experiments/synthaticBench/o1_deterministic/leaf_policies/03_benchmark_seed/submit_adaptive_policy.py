from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import submitit

SMAC_SEEDS = range(5)
PROBLEM_SEED = 53
PYTHONHASHSEED = "12345"
SLURM_PARTITION = "c23ms"

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
OUTPUT_DIRECTORY = HERE / "smac_output"
LOG_DIRECTORY = HERE / "submitit_logs" / "adaptive"

sys.path.insert(0, str(REPOSITORY_ROOT))
BASE = importlib.import_module(
    "experiments.synthaticBench.o1_deterministic.leaf_policies."
    "02_leafs_seeds.submit_adaptive_policy"
)


def run_adaptive_policy(smac_seed: int) -> dict[str, Any]:
    # Reuse the tested experiment implementation, changing only the
    # benchmark/problem seed and output location.
    BASE.PROBLEM_SEED = PROBLEM_SEED
    BASE.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    return BASE.run_adaptive_policy(smac_seed)


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder=str(LOG_DIRECTORY),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=5,
        cpus_per_task=1,
        mem_gb=4,
        slurm_job_name="SynthACtic_O1_Seed53_AdaptiveLeaf",
        slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"],
        slurm_additional_parameters={"requeue": True},
    )
    jobs = []
    with executor.batch():
        for smac_seed in SMAC_SEEDS:
            jobs.append(
                (
                    smac_seed,
                    executor.submit(run_adaptive_policy, smac_seed),
                )
            )

    print(f"Submitted {len(jobs)} adaptive-policy jobs for problem seed 53:")
    for smac_seed, job in jobs:
        print(f"seed={smac_seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

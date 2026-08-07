#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit the dimension study's compact-LLM and fixed-depth jobs."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

import experiment


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
ARRAY_PARALLELISM = 90
API_KEY_FILE = Path("/home/io632776/.config/openai/smac_api_key")
LOCAL_SMAC = experiment.base.LOCAL_SMAC_ROOT
PYTHON_ENV = Path(
    "/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python"
)


@dataclass(frozen=True)
class FixedDepthJob:
    dimension: int
    depth: int
    smac_seed: int

    def __call__(self):
        return experiment.run_fixed_depth(self.dimension, self.depth, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


@dataclass(frozen=True)
class CompactLLMJob:
    dimension: int
    smac_seed: int

    def __call__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is unavailable in the compact-policy job.")
        return experiment.run_compact(self.dimension, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def fixed_jobs(
    dimensions: tuple[int, ...] = experiment.DIMENSIONS,
) -> tuple[FixedDepthJob, ...]:
    return tuple(FixedDepthJob(*args) for args in experiment.fixed_jobs(dimensions))


def compact_jobs(
    dimensions: tuple[int, ...] = experiment.DIMENSIONS,
) -> tuple[CompactLLMJob, ...]:
    return tuple(CompactLLMJob(*args) for args in experiment.compact_jobs(dimensions))


def validate_api_key_file() -> None:
    if not API_KEY_FILE.is_file():
        raise RuntimeError(f"API key file does not exist: {API_KEY_FILE}.")
    mode = API_KEY_FILE.stat().st_mode & 0o777
    if mode != 0o600:
        raise RuntimeError(f"API key permissions are {mode:o}; expected 600.")
    if not API_KEY_FILE.read_text(encoding="utf-8").strip().startswith("sk-"):
        raise RuntimeError("API key file has an unexpected prefix.")


def common_parameters(job_name: str, parallelism: int) -> dict[str, Any]:
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": parallelism,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": job_name,
        "slurm_setup": [
            f"export PYTHONHASHSEED={experiment.PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'export PYTHONPATH="{LOCAL_SMAC}:{experiment.HERE}:${{PYTHONPATH:-}}"',
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(dimensions: tuple[int, ...], list_jobs: bool = False) -> None:
    print(f"Benchmark seed: {experiment.BENCHMARK_SEED}")
    print(f"Selected dimensions: {dimensions}; instances: {experiment.N_INSTANCES}")
    print(f"SMAC seeds: {experiment.SMAC_SEEDS}; trials/job: {experiment.N_TRIALS}")
    print(f"Compact LLM jobs: {len(compact_jobs(dimensions))}")
    print(f"Fixed-depth jobs: {len(fixed_jobs(dimensions))} at depths {experiment.FIXED_DEPTHS}")
    print(f"Total jobs: {len(compact_jobs(dimensions)) + len(fixed_jobs(dimensions))}")
    print(f"Slurm: account={SLURM_ACCOUNT}, partition={SLURM_PARTITION}, time=4h, memory={MEM_GB}GB")
    print(f"Python: {PYTHON_ENV}")
    print(f"Local SMAC: {LOCAL_SMAC}")
    if list_jobs:
        for index, job in enumerate(compact_jobs(dimensions)):
            print(f"compact[{index:02d}] dimension={job.dimension} smac_seed={job.smac_seed}")
        for index, job in enumerate(fixed_jobs(dimensions)):
            print(
                f"fixed[{index:02d}] dimension={job.dimension} depth={job.depth} "
                f"smac_seed={job.smac_seed}"
            )


def smoke_check() -> None:
    assert experiment.DIMENSIONS == (2, 5, 25, 50, 100)
    assert experiment.BENCHMARK_SEED == 40
    assert experiment.SMAC_SEEDS == tuple(range(5))
    assert experiment.FIXED_DEPTHS == (5, 10, 15, 20, 30)
    assert experiment.N_INSTANCES == 10 and experiment.N_TRIALS == 1_000
    assert len(compact_jobs()) == 25 and len(set(compact_jobs())) == 25
    assert len(fixed_jobs()) == 125 and len(set(fixed_jobs())) == 125
    assert len(compact_jobs((50, 100))) == 10
    assert len(fixed_jobs((50, 100))) == 50
    assert SLURM_ACCOUNT == "thes2388"
    assert str(LOCAL_SMAC.resolve()) in experiment.base.local_smac_metadata()["module"]
    validate_api_key_file()
    print("Smoke check passed: 25 compact + 125 fixed jobs, secure key, local SMAC.")


def submit(
    dimensions: tuple[int, ...],
) -> tuple[list[submitit.Job], list[submitit.Job]]:
    validate_api_key_file()
    experiment.HERE.joinpath("submitit_logs_fixed").mkdir(parents=True, exist_ok=True)
    experiment.HERE.joinpath("submitit_logs_compact").mkdir(parents=True, exist_ok=True)

    fixed_executor = submitit.AutoExecutor(folder=experiment.HERE / "submitit_logs_fixed")
    fixed_executor.update_parameters(
        **common_parameters("SynthACtic_O1_Dim_Fixed", len(fixed_jobs(dimensions)))
    )

    compact_executor = submitit.AutoExecutor(folder=experiment.HERE / "submitit_logs_compact")
    compact_parameters = common_parameters(
        "SynthACtic_O1_Dim_CompactLLM", len(compact_jobs(dimensions))
    )
    compact_parameters["slurm_setup"] += [
        f'test -r "{API_KEY_FILE}"',
        f'export OPENAI_API_KEY="$(< "{API_KEY_FILE}")"',
    ]
    compact_executor.update_parameters(**compact_parameters)

    with fixed_executor.batch():
        fixed_submissions = [
            fixed_executor.submit(job) for job in fixed_jobs(dimensions)
        ]
    with compact_executor.batch():
        compact_submissions = [
            compact_executor.submit(job) for job in compact_jobs(dimensions)
        ]
    print(f"Submitted fixed array: {fixed_submissions[0].job_id.rsplit('_', 1)[0]}")
    print(f"Submitted compact array: {compact_submissions[0].job_id.rsplit('_', 1)[0]}")
    return fixed_submissions, compact_submissions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        choices=experiment.DIMENSIONS,
        default=list(experiment.DIMENSIONS),
        help="Submit/list only these objective dimensions.",
    )
    args = parser.parse_args()
    dimensions = tuple(dict.fromkeys(args.dimensions))
    print_summary(dimensions, args.list_jobs)
    if args.smoke_check:
        smoke_check()
    if args.submit:
        submit(dimensions)


if __name__ == "__main__":
    main()

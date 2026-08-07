#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit Mistral fixed-100-tree O1 policy jobs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import submitit

import experiment


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
# All tasks may start immediately. Each worker handles a RWTHGPT 429 with the
# policy's 15-second retry loop, rather than waiting behind a Slurm array cap.
ARRAY_PARALLELISM = 40


@dataclass(frozen=True)
class MistralJob:
    kind: str
    dimension: int
    smac_seed: int

    def __call__(self):
        runner = (
            experiment.run_mistral_compact
            if self.kind == "compact"
            else experiment.run_mistral_every_second
        )
        return runner(self.dimension, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def all_jobs() -> tuple[MistralJob, ...]:
    return tuple(MistralJob(*args) for args in experiment.jobs())


def smoke_check() -> None:
    assert experiment.DIMENSIONS == (10, 25, 50, 100)
    assert experiment.SMAC_SEEDS == tuple(range(5))
    assert len(all_jobs()) == 40
    assert experiment.INITIAL_SETTINGS.n_trees == 100
    assert experiment.RWTHGPT_API_KEY_FILE.is_file()
    assert experiment.RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
    print("Smoke check passed: 40 Mistral jobs, 100 fixed trees, no Slurm API throttle.")


def submit() -> list[submitit.Job]:
    folder = experiment.HERE / "submitit_logs"
    folder.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(
        timeout_min=TIMEOUT_MIN,
        slurm_partition=SLURM_PARTITION,
        slurm_account=SLURM_ACCOUNT,
        slurm_array_parallelism=ARRAY_PARALLELISM,
        cpus_per_task=1,
        mem_gb=MEM_GB,
        slurm_job_name="SynthACtic_O1_Mistral100",
        slurm_setup=[
            f"export PYTHONHASHSEED={experiment.PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'export PYTHONPATH="{experiment.base.LOCAL_SMAC_ROOT}:{experiment.HERE}:{experiment.SHARED_POLICY_CODE}:${{PYTHONPATH:-}}"',
        ],
        slurm_additional_parameters={"requeue": True},
    )
    with executor.batch():
        submitted = [executor.submit(job) for job in all_jobs()]
    print(f"Submitted Mistral array: {submitted[0].job_id.rsplit('_', 1)[0]} (40 tasks)")
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print("Dimensions=(10, 25, 50, 100); seeds=(0, 1, 2, 3, 4); trials/job=1000")
    print("40 jobs: 20 compact-summary + 20 every-second-trial; n_trees fixed at 100")
    if args.list_jobs:
        for index, job in enumerate(all_jobs()):
            print(f"{index:02d}: {job}")
    if args.smoke_check:
        smoke_check()
    if args.submit or not (args.smoke_check or args.list_jobs):
        submit()


if __name__ == "__main__":
    main()

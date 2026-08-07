#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit the 5,000-trial O1 deterministic depth-policy study."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

import experiment


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 12 * 60
MEM_GB = 8
# No Slurm-side API throttle: every Mistral worker itself retries a denied
# RWTHGPT request after 15 seconds.  The remaining workers make no API calls.
ARRAY_PARALLELISM = 280


@dataclass(frozen=True)
class StudyJob:
    kind: str
    dimension: int
    smac_seed: int
    depth: int | None = None

    def __call__(self):
        if self.kind == "fixed":
            assert self.depth is not None
            return experiment.run_fixed(self.dimension, self.smac_seed, self.depth)
        if self.kind == "mistral":
            return experiment.run_mistral_compact(self.dimension, self.smac_seed)
        if self.kind == "holdout":
            return experiment.run_holdout(self.dimension, self.smac_seed)
        raise ValueError(f"Unknown job type {self.kind!r}")

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def all_jobs() -> tuple[StudyJob, ...]:
    return tuple(StudyJob(*item) for item in experiment.jobs())


def smoke_check() -> None:
    jobs = all_jobs()
    counts = {kind: sum(job.kind == kind for job in jobs) for kind in ("fixed", "mistral", "holdout")}
    assert counts == {"fixed": 200, "mistral": 40, "holdout": 40}
    assert len(jobs) == 280
    assert experiment.CHECKPOINTS == (100, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)
    assert experiment.HOLDOUT_CHECKPOINTS == tuple(range(500, 5000, 500))
    assert experiment.RWTHGPT_API_KEY_FILE.is_file()
    assert experiment.RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
    print("Smoke check passed: fixed=200, Mistral=40, chronological-holdout=40.")


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
        slurm_job_name="SynthACtic_O1_5000",
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
    print(f"Submitted 280-task array: {submitted[0].job_id.rsplit('_', 1)[0]}")
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Explicitly submit (the default).")
    args = parser.parse_args()
    print("O1 deterministic: dimensions=(10,25,50,100), seeds=0..9, trials/job=5000")
    print("Fixed depths=(5,10,20,30,20000); 100 trees; split=2; leaf=1")
    print("Compact Mistral checkpoints=(100,250,500,1000,...,4500); holdout at 500,...,4500")
    if args.list_jobs:
        for index, job in enumerate(all_jobs()):
            print(f"{index:03d}: {job}")
    if args.smoke_check:
        smoke_check()
    if args.submit or not (args.smoke_check or args.list_jobs):
        submit()


if __name__ == "__main__":
    main()

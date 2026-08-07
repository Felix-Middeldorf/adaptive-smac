#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit the O6 RWTHGPT RF-policy comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.trials import TrialInfo
import submitit

import experiment


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
ARRAY_PARALLELISM = 48
LLM_ARRAY_PARALLELISM = 2
PYTHON_ENV = Path("/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python")


@dataclass(frozen=True)
class FixedJob:
    dimension: int
    smac_seed: int

    def __call__(self):
        return experiment.run_fixed_100_trees(self.dimension, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


@dataclass(frozen=True)
class LLMJob:
    kind: str
    dimension: int
    smac_seed: int
    model_name: str

    def __call__(self):
        runner = experiment.run_initial_choice if self.kind == "initial_choice" else experiment.run_dynamic
        return runner(self.dimension, self.smac_seed, self.model_name)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def fixed_jobs() -> tuple[FixedJob, ...]:
    return tuple(FixedJob(*arguments) for arguments in experiment.fixed_jobs())


def llm_jobs(kind: str) -> tuple[LLMJob, ...]:
    return tuple(LLMJob(kind, *arguments) for arguments in experiment.llm_jobs(kind))


def common_parameters(name: str, parallelism: int) -> dict[str, Any]:
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": parallelism,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": name,
        "slurm_setup": [
            f"export PYTHONHASHSEED={experiment.PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'export PYTHONPATH="{experiment.base.LOCAL_SMAC_ROOT}:{experiment.HERE}:{experiment.SHARED_POLICY_CODE}:${{PYTHONPATH:-}}"',
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def smoke_check() -> None:
    assert experiment.DIMENSIONS == (50, 100)
    assert experiment.N_INSTANCES == 10 and experiment.N_TRIALS == 1_000
    assert experiment.GPT55_SEEDS == (0, 1, 2)
    assert experiment.GPT54_SEEDS == (0, 1, 2, 3, 4, 5)
    assert len(fixed_jobs()) == 12
    assert len(llm_jobs("initial_choice")) == 18
    assert len(llm_jobs("dynamic")) == 18
    assert experiment.RWTHGPT_API_KEY_FILE.is_file()
    assert experiment.RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
    assert experiment.base.LOCAL_SMAC_ROOT.is_dir()
    problem, instances = experiment._make_problem(50)
    cost = np.asarray(
        problem.evaluate(
            TrialInfo(
                config=problem.configspace.get_default_configuration(),
                instance=next(iter(instances)),
                seed=0,
            )
        ).cost,
        dtype=float,
    ).reshape(-1)
    assert cost.size == 1 and np.isfinite(cost[0]), cost
    print("Smoke check passed: 12 fixed + 18 initial-choice + 18 dynamic jobs.")


def _submit_group(
    name: str,
    folder: Path,
    jobs: tuple[Any, ...],
    parallelism: int = ARRAY_PARALLELISM,
) -> list[submitit.Job]:
    folder.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder)
    executor.update_parameters(**common_parameters(name, min(parallelism, len(jobs))))
    with executor.batch():
        submitted = [executor.submit(job) for job in jobs]
    print(f"Submitted {name}: {submitted[0].job_id.rsplit('_', 1)[0]} ({len(submitted)} tasks)")
    return submitted


def submit() -> tuple[list[submitit.Job], list[submitit.Job], list[submitit.Job]]:
    fixed = _submit_group("SynthACtic_O6_Fixed100", experiment.HERE / "submitit_logs_fixed", fixed_jobs())
    initial = _submit_group("SynthACtic_O6_RWTH_Initial", experiment.HERE / "submitit_logs_initial", llm_jobs("initial_choice"), LLM_ARRAY_PARALLELISM)
    dynamic = _submit_group("SynthACtic_O6_RWTH_Dynamic", experiment.HERE / "submitit_logs_dynamic", llm_jobs("dynamic"), LLM_ARRAY_PARALLELISM)
    return fixed, initial, dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    print("O6-Multimodal: dimensions=(50, 100), instances=10, trials=1000")
    print("Fixed baseline: 12 jobs; initial-choice: 18 jobs; dynamic: 18 jobs; total: 48 jobs")
    if args.list_jobs:
        for job in (*fixed_jobs(), *llm_jobs("initial_choice"), *llm_jobs("dynamic")):
            print(job)
    if args.smoke_check:
        smoke_check()
    if args.submit or not (args.smoke_check or args.list_jobs):
        submit()


if __name__ == "__main__":
    main()

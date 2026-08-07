#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Validate or submit the depth-only LLM ablation."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

import depth_experiment as exp


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
API_KEY_FILE = Path("/home/io632776/.config/openai/smac_api_key")


@dataclass(frozen=True)
class FixedJob:
    dimension: int
    depth: int
    smac_seed: int

    def __call__(self):
        return exp.run_fixed_depth(self.dimension, self.depth, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


@dataclass(frozen=True)
class LLMJob:
    dimension: int
    smac_seed: int

    def __call__(self):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is unavailable.")
        return exp.run_depth_policy(self.dimension, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def fixed_jobs() -> tuple[FixedJob, ...]:
    return tuple(FixedJob(*args) for args in exp.fixed_jobs())


def llm_jobs() -> tuple[LLMJob, ...]:
    return tuple(LLMJob(*args) for args in exp.llm_jobs())


def validate_key() -> None:
    if not API_KEY_FILE.is_file() or (API_KEY_FILE.stat().st_mode & 0o777) != 0o600:
        raise RuntimeError(f"Expected a mode-600 API key file at {API_KEY_FILE}.")
    if not API_KEY_FILE.read_text(encoding="utf-8").strip().startswith("sk-"):
        raise RuntimeError("API key file has an unexpected prefix.")


def parameters(name: str, parallelism: int) -> dict[str, Any]:
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": parallelism,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": name,
        "slurm_setup": [
            f"export PYTHONHASHSEED={exp.PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'export PYTHONPATH="{exp.base.LOCAL_SMAC_ROOT}:{exp.HERE}:${{PYTHONPATH:-}}"',
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def summary() -> None:
    print(f"Dimensions: {exp.DIMENSIONS}; benchmark seed: {exp.BENCHMARK_SEED}")
    print(f"SMAC seeds: {exp.SMAC_SEEDS}; trials/job: {exp.N_TRIALS}; instances: {exp.N_INSTANCES}")
    print(f"Depth-only LLM jobs: {len(llm_jobs())}; fixed-depth jobs: {len(fixed_jobs())}; total: {len(llm_jobs()) + len(fixed_jobs())}")
    print(f"Fixed RF: trees={exp.N_TREES}, split={exp.MIN_SAMPLES_SPLIT}, leaf={exp.MIN_SAMPLES_LEAF}, ratio={exp.FEATURE_RATIO}, PCA={exp.PCA_COMPONENTS}")
    print(f"Random design probability: {exp.RANDOM_DESIGN_PROBABILITY}")
    print(f"LLM depth range/checkpoints: 1-30 / {exp.CHECKPOINTS}")
    print(f"Slurm: {SLURM_ACCOUNT}, {SLURM_PARTITION}, {TIMEOUT_MIN} min, {MEM_GB} GB")


def smoke_check() -> None:
    assert len(fixed_jobs()) == 75 and len(set(fixed_jobs())) == 75
    assert len(llm_jobs()) == 15 and len(set(llm_jobs())) == 15
    assert exp.DECISION_SCHEMA["properties"]["max_depth"] == {"type": "integer", "minimum": 1, "maximum": 30}
    chosen, normalized = exp.validate_depth_decision({"max_depth": 17, "confidence": 0.7, "reason": "test"})
    assert chosen == exp.settings(17) and set(normalized) == {"max_depth", "confidence", "reason"}
    assert chosen.n_trees == 100 and chosen.min_samples_split == 2 and chosen.min_samples_leaf == 1
    assert exp.RANDOM_DESIGN_PROBABILITY == 0.0
    assert str(exp.base.LOCAL_SMAC_ROOT.resolve()) in exp.base.local_smac_metadata()["module"]
    validate_key()
    print("Smoke check passed: 15 depth-only LLM + 75 fixed jobs.")


def submit() -> tuple[list[submitit.Job], list[submitit.Job]]:
    validate_key()
    fixed_executor = submitit.AutoExecutor(folder=exp.HERE / "submitit_logs_fixed")
    fixed_executor.update_parameters(**parameters("SynthACtic_O1_Depth_Fixed100", 75))
    llm_executor = submitit.AutoExecutor(folder=exp.HERE / "submitit_logs_llm")
    llm_parameters = parameters("SynthACtic_O1_Depth_LLM100", 15)
    llm_parameters["slurm_setup"] += [
        f'test -r "{API_KEY_FILE}"',
        f'export OPENAI_API_KEY="$(< "{API_KEY_FILE}")"',
    ]
    llm_executor.update_parameters(**llm_parameters)
    with fixed_executor.batch():
        fixed = [fixed_executor.submit(job) for job in fixed_jobs()]
    with llm_executor.batch():
        llm = [llm_executor.submit(job) for job in llm_jobs()]
    print(f"Submitted fixed array: {fixed[0].job_id.rsplit('_', 1)[0]}")
    print(f"Submitted LLM array: {llm[0].job_id.rsplit('_', 1)[0]}")
    return fixed, llm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    summary()
    if args.smoke_check:
        smoke_check()
    if args.submit:
        submit()


if __name__ == "__main__":
    main()

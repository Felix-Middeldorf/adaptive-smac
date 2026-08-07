"""Submitit support for ACLib random-configuration sampling."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

from random_sampling_experiment import (
    CONFIGSPACE_SEED,
    N_CONFIGURATIONS,
    QUANTILE_SEEDS,
    RandomSamplingDefinition,
    _canonical_configuration,
    run_random_sampling,
    sample_unique_configurations,
)
from surrogate_benchmark import get_benchmark_spec, load_benchmark_data


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"
SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "lect0190"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"


@dataclass(frozen=True)
class RandomSamplingJob:
    definition: RandomSamplingDefinition

    def __call__(self) -> dict[str, Any]:
        actual = os.environ.get("PYTHONHASHSEED")
        if actual != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, found {actual!r}."
            )
        return run_random_sampling(self.definition)

    def checkpoint(self) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(self)


def slurm_parameters(
    definition: RandomSamplingDefinition,
) -> dict[str, Any]:
    pythonpath = (
        f"{EPM_SOURCE}:{HERE}:{definition.directory}:"
        "${PYTHONPATH:-}"
    )
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": (
            f"ACLib_{definition.benchmark_key}_06_random_sample_1000"
        ),
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(definition: RandomSamplingDefinition) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    evaluations = (
        N_CONFIGURATIONS
        * len(data.training_instances)
        * len(QUANTILE_SEEDS)
    )
    print(f"Benchmark: {spec.display_name}")
    print(f"Unique random configurations: {N_CONFIGURATIONS}")
    print(f"ConfigSpace seed: {CONFIGSPACE_SEED}")
    print(f"Training instances: {len(data.training_instances)}")
    print(f"Quantile seeds: {QUANTILE_SEEDS}")
    print(f"Target predictions: {evaluations}")
    print(f"Output: {definition.output_directory}")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Time limit: {TIMEOUT_MIN} minutes; memory: {MEM_GB} GB")
    print(f"Job name: {slurm_parameters(definition)['slurm_job_name']}")


def smoke_check(definition: RandomSamplingDefinition) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    configurations, draws = sample_unique_configurations(
        data.configspace,
        n_configurations=8,
        seed=CONFIGSPACE_SEED,
    )
    assert len(configurations) == 8
    assert draws >= 8
    assert len(
        {
            _canonical_configuration(config)
            for config in configurations
        }
    ) == 8
    assert slurm_parameters(definition)["slurm_account"] == "lect0190"
    print(
        f"PASS: {spec.display_name}; unique sampling and Slurm settings are valid."
    )


def submit_job(definition: RandomSamplingDefinition) -> None:
    print_summary(definition)
    executor = submitit.AutoExecutor(
        folder=str(definition.directory / "submitit_logs"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(**slurm_parameters(definition))
    job = executor.submit(RandomSamplingJob(definition))
    print(f"Submitted {definition.benchmark_key}: {job.job_id}")


def main(definition: RandomSamplingDefinition) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()
    if args.smoke_check:
        smoke_check(definition)
    elif args.dry_run:
        print_summary(definition)
    else:
        submit_job(definition)

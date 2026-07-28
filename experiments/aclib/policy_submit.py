"""Submitit support for deterministic ACLib adaptive-depth policies."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import submitit

from fixed_depth_experiment import LOCAL_SMAC_ROOT
from fixed_depth_submit import EPM_SOURCE
from policy_experiment import (
    HERE as ACLIB_EXPERIMENT_ROOT,
    N_TRIALS,
    POLICIES,
    SMAC_SEEDS,
    PolicyExperimentDefinition,
    derive_oracle_schedule,
    run_policy,
)
from surrogate_benchmark import get_benchmark_spec, load_benchmark_data


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "lect0190"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"
MAX_SUBMITTED_JOBS = 80


@dataclass(frozen=True)
class PolicyJob:
    definition: PolicyExperimentDefinition
    policy: str
    smac_seed: int

    def __call__(self):
        if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected worker PYTHONHASHSEED={PYTHONHASHSEED}, "
                f"found {os.environ.get('PYTHONHASHSEED')!r}."
            )
        return run_policy(
            definition=self.definition,
            policy=self.policy,
            smac_seed=self.smac_seed,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs(
    definition: PolicyExperimentDefinition,
) -> tuple[PolicyJob, ...]:
    jobs = tuple(
        PolicyJob(definition, policy, smac_seed)
        for policy in POLICIES
        for smac_seed in SMAC_SEEDS
    )
    expected = len(POLICIES) * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("Policy job matrix is incomplete or duplicated.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError("Policy job matrix exceeds the submission limit.")
    return jobs


def slurm_parameters(
    definition: PolicyExperimentDefinition,
) -> dict[str, Any]:
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{EPM_SOURCE}:{ACLIB_EXPERIMENT_ROOT}:"
        f"{definition.directory}:${{PYTHONPATH:-}}"
    )
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": len(experiment_jobs(definition)),
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": (
            f"ACLib_{definition.benchmark_key}_02_depth_policies"
        ),
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(
    definition: PolicyExperimentDefinition,
    *,
    list_jobs: bool,
) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    jobs = experiment_jobs(definition)
    data = load_benchmark_data(spec)
    print(f"Benchmark: {spec.display_name}")
    print(f"Policies: {POLICIES}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Jobs: {len(jobs)}")
    print(f"Trials per job: {N_TRIALS}")
    print("Scenario deterministic: True")
    print("Target/quantile seed: fixed at 0")
    print(f"Training instances: all {len(data.training_instances)}")
    print("Test instances used: 0")
    print(f"Local SMAC: {LOCAL_SMAC_ROOT}")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Slurm job name: {slurm_parameters(definition)['slurm_job_name']}")
    print(f"Output root: {definition.output_root}")
    for smac_seed in SMAC_SEEDS:
        schedule = derive_oracle_schedule(definition, smac_seed)
        print(
            f"Oracle seed {smac_seed}: "
            f"{len(schedule['segments'])} segments through trial "
            f"{schedule['common_trial_limit']}, then hold depth "
            f"{schedule['segments'][-1]['depth']}"
        )
    if list_jobs:
        for index, job in enumerate(jobs):
            print(
                f"job={index:02d} policy={job.policy} "
                f"smac_seed={job.smac_seed}"
            )


def smoke_check(definition: PolicyExperimentDefinition) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    jobs = experiment_jobs(definition)
    assert len(jobs) == 18
    assert N_TRIALS == 5_000
    assert len(data.training_instances) == spec.expected_training_instances
    assert len(data.test_instances) == spec.expected_test_instances
    assert not set(data.training_instances) & set(data.test_instances)
    assert definition.initial_choice_file.is_file()
    for smac_seed in SMAC_SEEDS:
        schedule = derive_oracle_schedule(definition, smac_seed)
        assert schedule["segments"]
        assert schedule["segments"][0]["trial"] == 1
        assert schedule["comparison_quantile_seed"] == 0
        assert schedule["common_trial_limit"] >= 1
    parameters = slurm_parameters(definition)
    assert parameters["slurm_account"] == "lect0190"
    assert parameters["slurm_setup"][0] == "export PYTHONHASHSEED=0"
    assert definition.benchmark_key in parameters["slurm_job_name"]
    print(
        f"PASS: {spec.display_name}; 18 deterministic policy jobs; "
        f"{len(data.training_instances)} training instances."
    )
    print("PASS: quantile seed=0, SMAC seeds=(0,1,2), local SMAC configured.")


def submit_jobs(definition: PolicyExperimentDefinition) -> None:
    jobs = experiment_jobs(definition)
    print_summary(definition, list_jobs=False)
    executor = submitit.AutoExecutor(
        folder=str(definition.directory / "submitit_logs"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(**slurm_parameters(definition))
    submitted = []
    with executor.batch():
        for job_spec in jobs:
            submitted.append((job_spec, executor.submit(job_spec)))
    print(f"Submitted {len(submitted)} policy jobs.")
    for job_spec, job in submitted:
        print(
            f"policy={job_spec.policy}, seed={job_spec.smac_seed}: "
            f"{job.job_id}"
        )


def main(definition: PolicyExperimentDefinition) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()
    if args.smoke_check:
        smoke_check(definition)
    elif args.dry_run or args.list_jobs:
        print_summary(definition, list_jobs=args.list_jobs)
    else:
        submit_jobs(definition)

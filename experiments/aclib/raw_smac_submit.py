"""Submitit support for native-output-only ACLib timing controls."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

from raw_smac_experiment import (
    DETERMINISTIC,
    DEPTHS,
    LOCAL_SMAC_ROOT,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_TREES,
    N_TRIALS,
    PCA_COMPONENTS,
    RANDOM_DESIGN_PROBABILITY,
    SMAC_SEEDS,
    SURROGATE_QUANTILE_SEED,
    RawExperimentDefinition,
    build_components,
    load_initial_choice,
    local_smac_metadata,
    run_raw_smac,
)
from surrogate_benchmark import REPOSITORY_ROOT, get_benchmark_spec


SLURM_PARTITION = "c23ms"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"
ACLIB_EXPERIMENT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RunJob:
    definition: RawExperimentDefinition
    depth: int
    smac_seed: int
    pca_components: int | None

    def __call__(self):
        actual_hash_seed = os.environ.get("PYTHONHASHSEED")
        if actual_hash_seed != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, "
                f"found {actual_hash_seed!r}."
            )
        return run_raw_smac(
            definition=self.definition,
            depth=self.depth,
            smac_seed=self.smac_seed,
            pca_components=self.pca_components,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs(
    definition: RawExperimentDefinition,
) -> tuple[RunJob, ...]:
    jobs = tuple(
        RunJob(definition, depth, smac_seed, pca_components)
        for pca_components in PCA_COMPONENTS
        for depth in DEPTHS
        for smac_seed in SMAC_SEEDS
    )
    expected = len(PCA_COMPONENTS) * len(DEPTHS) * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("Incomplete or duplicate raw-run job matrix.")
    return jobs


def slurm_parameters(
    definition: RawExperimentDefinition,
) -> dict[str, Any]:
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{EPM_SOURCE}:{ACLIB_EXPERIMENT_ROOT}:"
        f"{definition.directory}:${{PYTHONPATH:-}}"
    )
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_array_parallelism": len(experiment_jobs(definition)),
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": f"ACLib_{definition.initials}_30raw",
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(
    definition: RawExperimentDefinition,
    *,
    list_jobs: bool = False,
) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    choice = load_initial_choice(definition.initial_choice_file)
    jobs = experiment_jobs(definition)
    print(f"Benchmark: {spec.display_name}")
    print(f"Depths: {DEPTHS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"Trials per run: {N_TRIALS}")
    print(f"Total jobs: {len(jobs)}")
    training_instance_count = (
        spec.expected_training_instances
        if definition.training_instance_limit is None
        else min(
            definition.training_instance_limit,
            spec.expected_training_instances,
        )
    )
    print(f"Training instances: {training_instance_count}")
    print("Test instances used: 0")
    print(f"Scenario deterministic: {DETERMINISTIC}")
    print(f"Surrogate quantile seed: {SURROGATE_QUANTILE_SEED} (median)")
    print(f"Initial configuration: {choice.kind}")
    print(
        f"RF: trees={N_TREES}, depths={DEPTHS}, "
        f"split={MIN_SAMPLES_SPLIT}, leaf={MIN_SAMPLES_LEAF}, "
        f"PCA modes={PCA_COMPONENTS}"
    )
    print(f"Random-design probability: {RANDOM_DESIGN_PROBABILITY}")
    print("Telemetry callbacks: none")
    print("Custom acquisition wrapper: none")
    print("Custom files in SMAC output: none")
    print("Slurm account: not specified (use personal/default account)")
    print(f"Time/memory: {TIMEOUT_MIN} minutes / {MEM_GB} GB")
    print(f"Output root: {definition.output_root}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(
                f"job={index:02d} depth={job.depth} "
                f"smac_seed={job.smac_seed} "
                f"pca_components={job.pca_components}"
            )


def smoke_check(definition: RawExperimentDefinition) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    jobs = experiment_jobs(definition)
    checked_components = [
        build_components(
            definition,
            depth=DEPTHS[0],
            smac_seed=SMAC_SEEDS[0],
            pca_components=pca_components,
        )
        for pca_components in PCA_COMPONENTS
    ]
    (
        _spec,
        data,
        scenario,
        initial,
        _initial_design,
        model,
        random_design,
    ) = checked_components[0]
    pca_model = checked_components[1][5]
    choice = load_initial_choice(definition.initial_choice_file)
    parameters = slurm_parameters(definition)
    metadata = local_smac_metadata()

    assert len(jobs) == 16
    assert DEPTHS == (5, 10, 20, 30)
    assert SMAC_SEEDS == (0, 1)
    assert N_TRIALS == 5_000
    expected_instances = data.training_instances[
        : definition.training_instance_limit
    ]
    assert len(data.training_instances) == spec.expected_training_instances
    assert tuple(scenario.instances or ()) == expected_instances
    assert scenario.deterministic is True
    assert DETERMINISTIC is True
    assert SURROGATE_QUANTILE_SEED == 0
    assert len(initial) > 0
    if choice.kind == "default":
        assert initial == data.configspace.get_default_configuration()
    else:
        assert initial != data.configspace.get_default_configuration()
    assert model.meta["n_estimators"] == 100
    assert model.meta["max_depth"] == 5
    assert model.meta["min_samples_split"] == 1
    assert model.meta["min_samples_leaf"] == 1
    assert PCA_COMPONENTS == (None, 4)
    assert model.meta["pca_components"] is None
    assert pca_model.meta["pca_components"] == 4
    assert random_design.meta["probability"] == 0.0
    assert "slurm_account" not in parameters
    assert parameters["mem_gb"] == 4
    assert parameters["timeout_min"] == 16 * 60
    assert parameters["slurm_setup"][0] == "export PYTHONHASHSEED=0"
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["module"]
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["random_forest"]
    print(
        f"PASS: {spec.display_name}; 16 jobs, native SMAC output only, "
        f"{len(expected_instances)} training instances."
    )


def submit_jobs(definition: RawExperimentDefinition) -> None:
    jobs = experiment_jobs(definition)
    print_summary(definition)
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
    print(f"Submitted {len(submitted)} jobs.")
    for job_spec, job in submitted:
        print(
            f"depth={job_spec.depth}, smac_seed={job_spec.smac_seed}, "
            f"pca_components={job_spec.pca_components}: "
            f"{job.job_id}"
        )


def main(definition: RawExperimentDefinition) -> None:
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

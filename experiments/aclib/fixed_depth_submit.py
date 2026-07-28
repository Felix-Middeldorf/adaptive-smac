"""Submitit support shared by the ACLib fixed-depth experiments."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import submitit

from fixed_depth_experiment import (
    AlgorithmConfigurationFacade,
    DEPTHS,
    HERE as ACLIB_EXPERIMENT_ROOT,
    LOCAL_SMAC_ROOT,
    MIN_SAMPLES_LEAF,
    MIN_SAMPLES_SPLIT,
    N_TRIALS,
    N_TREES,
    PCA_COMPONENTS,
    RANDOM_DESIGN_PROBABILITY,
    Scenario,
    SMAC_SEEDS,
    ExperimentDefinition,
    load_initial_choice,
    local_smac_metadata,
    resolve_initial_configuration,
    run_fixed_depth,
)
from surrogate_benchmark import (
    REPOSITORY_ROOT,
    get_benchmark_spec,
    load_benchmark_data,
)


MAX_SUBMITTED_JOBS = 80
SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "lect0190"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"
EPM_SOURCE = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "epm"


@dataclass(frozen=True)
class RunJob:
    definition: ExperimentDefinition
    depth: int
    smac_seed: int

    def __call__(self):
        actual_hash_seed = os.environ.get("PYTHONHASHSEED")
        if actual_hash_seed != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected worker PYTHONHASHSEED={PYTHONHASHSEED}, "
                f"found {actual_hash_seed!r}."
            )
        return run_fixed_depth(
            definition=self.definition,
            depth=self.depth,
            smac_seed=self.smac_seed,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs(definition: ExperimentDefinition) -> tuple[RunJob, ...]:
    jobs = tuple(
        RunJob(definition=definition, depth=depth, smac_seed=smac_seed)
        for depth in DEPTHS
        for smac_seed in SMAC_SEEDS
    )
    expected = len(DEPTHS) * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("Job matrix is incomplete or contains duplicates.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Experiment has {len(jobs)} jobs, above limit {MAX_SUBMITTED_JOBS}."
        )
    return jobs


def slurm_parameters(definition: ExperimentDefinition) -> dict[str, Any]:
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
            f"ACLib_{definition.benchmark_key}_{definition.directory.name}"
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
    definition: ExperimentDefinition,
    *,
    list_jobs: bool = False,
) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    deterministic = (
        spec.deterministic
        if definition.deterministic is None
        else bool(definition.deterministic)
    )
    choice, _ = load_initial_choice(definition.initial_choice_file)
    jobs = experiment_jobs(definition)
    print(f"Benchmark: {spec.display_name}")
    print(f"SMAC surrogate depths: {DEPTHS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Training instances: all {spec.expected_training_instances}")
    print(f"Test instances used by SMAC: 0")
    print(f"Completed trials per run: {N_TRIALS}")
    print(f"Scenario deterministic: {deterministic}")
    print(
        "Target/quantile seed: "
        + ("fixed at 0" if deterministic else "SMAC evaluation seed")
    )
    print(f"Initial configuration: {choice.kind}")
    print(f"Local SMAC checkout: {LOCAL_SMAC_ROOT}")
    print(f"Total SMAC runs / Slurm jobs: {len(jobs)}")
    print(f"Time limit per job: {TIMEOUT_MIN} minutes")
    print(f"Memory per job: {MEM_GB} GB")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Worker PYTHONHASHSEED: {PYTHONHASHSEED}")
    print(f"Slurm job name: {slurm_parameters(definition)['slurm_job_name']}")
    print(f"Output root: {definition.output_root}")
    if list_jobs:
        for index, job in enumerate(jobs):
            print(
                f"job={index:02d} depth={job.depth} "
                f"smac_seed={job.smac_seed}"
            )


def submit_jobs(definition: ExperimentDefinition) -> None:
    jobs = experiment_jobs(definition)
    print_summary(definition)
    log_directory = definition.directory / "submitit_logs"
    executor = submitit.AutoExecutor(
        folder=str(log_directory),
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
            f"depth={job_spec.depth}, smac_seed={job_spec.smac_seed}: "
            f"{job.job_id}"
        )


def smoke_check(definition: ExperimentDefinition) -> None:
    spec = get_benchmark_spec(definition.benchmark_key)
    deterministic = (
        spec.deterministic
        if definition.deterministic is None
        else bool(definition.deterministic)
    )
    data = load_benchmark_data(spec)
    choice, _ = load_initial_choice(definition.initial_choice_file)
    initial = resolve_initial_configuration(data, choice)
    jobs = experiment_jobs(definition)
    training_features = {
        instance: data.features[instance]
        for instance in data.training_instances
    }
    scenario = Scenario(
        configspace=data.configspace,
        deterministic=deterministic,
        n_trials=N_TRIALS,
        use_default_config=choice.kind == "default",
        instances=list(data.training_instances),
        instance_features=training_features,
        seed=SMAC_SEEDS[0],
        n_workers=1,
    )
    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        n_trees=N_TREES,
        max_depth=DEPTHS[0],
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        pca_components=PCA_COMPONENTS,
    )
    random_design = AlgorithmConfigurationFacade.get_random_design(
        scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )

    assert len(jobs) == 15
    assert N_TRIALS == 5_000
    assert len(data.training_instances) == spec.expected_training_instances
    assert len(data.test_instances) == spec.expected_test_instances
    assert not set(data.training_instances) & set(data.test_instances)
    assert tuple(scenario.instances or ()) == data.training_instances
    assert scenario.deterministic == deterministic
    if definition.deterministic is not None:
        assert deterministic is True
    assert len(initial) > 0
    if choice.kind == "default":
        assert initial == data.configspace.get_default_configuration()
    else:
        assert initial != data.configspace.get_default_configuration()
    assert model.meta["n_estimators"] == 100
    assert model.meta["min_samples_split"] == 1
    assert model.meta["min_samples_leaf"] == 1
    assert model.meta["pca_components"] is None
    assert random_design.meta["probability"] == 0.0
    parameters = slurm_parameters(definition)
    assert parameters["slurm_account"] == "lect0190"
    assert parameters["slurm_setup"][0] == "export PYTHONHASHSEED=0"
    assert definition.benchmark_key in parameters["slurm_job_name"]
    assert definition.directory.name in parameters["slurm_job_name"]
    metadata = local_smac_metadata()
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["module"]
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["random_forest"]
    print(
        f"PASS: {spec.display_name}; 15 jobs; "
        f"{len(data.training_instances)} training and 0 test instances."
    )
    print(f"PASS: initial configuration kind is {choice.kind}.")
    print("PASS: RF trees=100, split=1, leaf=1, PCA=None; random design=0%.")
    print("PASS: Slurm account=lect0190 and worker PYTHONHASHSEED=0.")
    print(f"PASS: local SMAC imported from {metadata['module']}.")


def main(definition: ExperimentDefinition) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    args = parser.parse_args()
    if args.dry_run or args.list_jobs:
        print_summary(definition, list_jobs=args.list_jobs)
    else:
        submit_jobs(definition)

"""Submitit support for ACLib fixed-checkpoint RF schedule experiments."""

from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import submitit

from fixed_depth_experiment import (
    LOCAL_SMAC_ROOT,
    AlgorithmConfigurationFacade,
    Scenario,
    local_smac_metadata,
)
from smac.utils.configspace import convert_configurations_to_array
from fixed_depth_submit import EPM_SOURCE
from rf_schedule_catalog import (
    CHECKPOINTS,
    DEPTHS,
    FEATURE_RATIOS,
    N_SCHEDULES,
    SCHEDULES,
    SPLIT_SIZES,
)
from rf_schedule_experiment import (
    FixedCheckpointRFScheduleCallback,
    MIN_SAMPLES_LEAF,
    N_TREES,
    N_TRIALS,
    PCA_COMPONENTS,
    RANDOM_DESIGN_PROBABILITY,
    SMAC_SEEDS,
    RFScheduleExperimentDefinition,
    run_rf_schedule,
)
from surrogate_benchmark import load_benchmark_data, get_benchmark_spec


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "lect0190"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"
MAX_SUBMITTED_JOBS = 200
ACLIB_ROOT = EPM_SOURCE.parents[2] / "experiments" / "aclib"


@dataclass(frozen=True)
class RFScheduleJob:
    definition: RFScheduleExperimentDefinition
    schedule_index: int
    smac_seed: int

    def __call__(self):
        actual = os.environ.get("PYTHONHASHSEED")
        if actual != PYTHONHASHSEED:
            raise RuntimeError(
                f"Expected PYTHONHASHSEED={PYTHONHASHSEED}, found {actual!r}."
            )
        return run_rf_schedule(
            definition=self.definition,
            schedule_index=self.schedule_index,
            smac_seed=self.smac_seed,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs(
    definition: RFScheduleExperimentDefinition,
) -> tuple[RFScheduleJob, ...]:
    jobs = tuple(
        RFScheduleJob(definition, schedule.index, seed)
        for schedule in SCHEDULES
        for seed in SMAC_SEEDS
    )
    expected = N_SCHEDULES * len(SMAC_SEEDS)
    if len(jobs) != expected or len(set(jobs)) != expected:
        raise RuntimeError("RF schedule job matrix is incomplete or duplicated.")
    if len(jobs) > MAX_SUBMITTED_JOBS:
        raise RuntimeError(
            f"Job matrix has {len(jobs)} jobs; limit is {MAX_SUBMITTED_JOBS}."
        )
    return jobs


def slurm_parameters(
    definition: RFScheduleExperimentDefinition,
) -> dict[str, Any]:
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{EPM_SOURCE}:{ACLIB_ROOT}:"
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
            f"ACLib_{definition.benchmark_key}_05_rf_schedule_grid"
        ),
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def write_schedule_catalog(
    definition: RFScheduleExperimentDefinition,
) -> None:
    path = definition.directory / "schedule_catalog.json"
    path.write_text(
        json.dumps(
            {
                "depths": list(DEPTHS),
                "split_sizes": list(SPLIT_SIZES),
                "feature_ratios": list(FEATURE_RATIOS),
                "checkpoints": list(CHECKPOINTS),
                "n_trees": N_TREES,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "pca_components": PCA_COMPONENTS,
                "schedules": [
                    schedule.to_dict() for schedule in SCHEDULES
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def print_summary(
    definition: RFScheduleExperimentDefinition,
    *,
    list_jobs: bool = False,
) -> None:
    write_schedule_catalog(definition)
    spec = get_benchmark_spec(definition.benchmark_key)
    print(f"Benchmark: {spec.display_name}")
    print(f"Schedules: {N_SCHEDULES}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Jobs: {len(experiment_jobs(definition))}")
    print(f"Trials per job: {N_TRIALS}")
    print(f"Depth choices: {DEPTHS}")
    print(f"Split-size choices: {SPLIT_SIZES}")
    print(f"Feature-ratio choices: {FEATURE_RATIOS}")
    print(f"Trees: {N_TREES}; leaf size: {MIN_SAMPLES_LEAF}")
    print(f"PCA components: {PCA_COMPONENTS}")
    print(f"Random design probability: {RANDOM_DESIGN_PROBABILITY}")
    print("Scenario deterministic: True; target quantile seed: 0")
    print(f"Local SMAC: {LOCAL_SMAC_ROOT}")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Time limit: {TIMEOUT_MIN} minutes; memory: {MEM_GB} GB")
    print(f"Slurm job name: {slurm_parameters(definition)['slurm_job_name']}")
    print(f"Output root: {definition.output_root}")
    print(f"Schedule catalog: {definition.directory / 'schedule_catalog.json'}")
    if list_jobs:
        for index, job in enumerate(experiment_jobs(definition)):
            print(
                f"job={index:03d} schedule={job.schedule_index:02d} "
                f"smac_seed={job.smac_seed}"
            )


def smoke_check(definition: RFScheduleExperimentDefinition) -> None:
    jobs = experiment_jobs(definition)
    assert len(SCHEDULES) == 50
    assert len(jobs) == 150
    assert len(set(schedule.phases for schedule in SCHEDULES)) == 50
    for phase in range(3):
        assert {
            schedule.phases[phase].depth for schedule in SCHEDULES
        } == set(DEPTHS)
        assert {
            schedule.phases[phase].min_samples_split
            for schedule in SCHEDULES
        } == set(SPLIT_SIZES)
        assert {
            schedule.phases[phase].feature_ratio
            for schedule in SCHEDULES
        } == set(FEATURE_RATIOS)
    for schedule in SCHEDULES:
        assert schedule.phase_index(499) == 0
        assert schedule.phase_index(500) == 1
        assert schedule.phase_index(1_999) == 1
        assert schedule.phase_index(2_000) == 2

    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    first = SCHEDULES[4].phases[0]
    scenario = Scenario(
        configspace=data.configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        use_default_config=False,
        instances=list(data.training_instances),
        instance_features={
            instance: data.features[instance]
            for instance in data.training_instances
        },
        seed=0,
        n_workers=1,
    )
    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        n_trees=N_TREES,
        ratio_features=first.feature_ratio,
        max_depth=first.depth,
        min_samples_split=first.min_samples_split,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        pca_components=PCA_COMPONENTS,
    )
    sampling_space = copy.deepcopy(data.configspace)
    sampling_space.seed(91_703)
    configurations = sampling_space.sample_configuration(size=12)
    config_array = convert_configurations_to_array(configurations)
    feature_array = np.asarray(
        [
            data.features[instance]
            for instance in data.training_instances[:12]
        ],
        dtype=float,
    )
    X = np.hstack((config_array, feature_array))
    y = np.linspace(1.0, 12.0, num=12)
    model.train(X, y)
    expected_features = config_array.shape[1] + PCA_COMPONENTS
    initial_expected_max_features = max(
        1,
        int(expected_features * first.feature_ratio),
    )
    assert model._apply_pca is True
    assert model._rf is not None
    assert model._rf.n_features_in_ == expected_features
    assert model._rf.max_features == initial_expected_max_features
    means, variances = model.predict_marginalized(config_array[:2])
    assert np.isfinite(means).all()
    assert np.isfinite(variances).all()

    class _Selector:
        def __init__(self, completed_trials: int) -> None:
            self._runhistory = [None] * completed_trials

    schedule = SCHEDULES[4]
    with tempfile.TemporaryDirectory(prefix="aclib-rf-schedule-smoke-") as tmp:
        callback = FixedCheckpointRFScheduleCallback(
            schedule=schedule,
            output_directory=Path(tmp),
            model=model,
            overwrite=True,
        )
        for completed_trials, expected_phase in ((0, 0), (500, 1), (2_000, 2)):
            expected = schedule.phases[expected_phase]
            callback.on_next_configurations_start(_Selector(completed_trials))
            model.train(X, y)
            expected_max_features = max(
                1,
                int(expected_features * expected.feature_ratio),
            )
            assert callback.state["active_phase"] == expected_phase
            assert model._rf.max_depth == expected.depth
            assert model._rf.min_samples_split == expected.min_samples_split
            assert model._rf.max_features == expected_max_features
        assert len(callback.state["transitions"]) == 3

    parameters = slurm_parameters(definition)
    assert parameters["slurm_account"] == "lect0190"
    assert parameters["timeout_min"] == 16 * 60
    assert parameters["mem_gb"] == 4
    assert parameters["slurm_array_parallelism"] == 150
    assert parameters["slurm_setup"][0] == "export PYTHONHASHSEED=0"
    metadata = local_smac_metadata()
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["module"]
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["random_forest"]
    write_schedule_catalog(definition)
    print(
        f"PASS: {spec.display_name}; 50 schedules x 3 seeds = 150 jobs."
    )
    print(
        "PASS: every phase covers all depth, split-size, and feature-ratio "
        "choices; checkpoints are 500 and 2000."
    )
    print(
        f"PASS: PCA=4 transformed features={expected_features}, "
        f"ratio={first.feature_ratio}, "
        f"max_features={initial_expected_max_features}; "
        "marginalized prediction works."
    )
    print(
        "PASS: actual RF fits switch depth, min_samples_split, and "
        "post-PCA max_features at completed trials 500 and 2000."
    )
    print(
        "PASS: deterministic quantile seed=0; RF trees=100, leaf=1; "
        "local SMAC; lect0190; 16h; 4GB."
    )


def submit_jobs(definition: RFScheduleExperimentDefinition) -> None:
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
    print(f"Submitted {len(submitted)} RF schedule jobs.")
    for job_spec, job in submitted:
        print(
            f"schedule={job_spec.schedule_index:02d}, "
            f"seed={job_spec.smac_seed}: {job.job_id}"
        )


def main(definition: RFScheduleExperimentDefinition) -> None:
    import argparse

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

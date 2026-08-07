#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit frozen LLM-proposed RF configurations across target dimensions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

import experiment
import llm_config_cross_dimension as cross
from submit_experiment import ARRAY_PARALLELISM, common_parameters


LOG_DIRECTORY = experiment.HERE / "submitit_logs_llm_config_cross_dimension"


@dataclass(frozen=True)
class CrossDimensionConfigurationJob:
    target_dimension: int
    configuration_id: str
    smac_seed: int

    def __call__(self):
        return cross.run_configuration(
            self.target_dimension, self.configuration_id, self.smac_seed
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def jobs(
    target_dimensions: tuple[int, ...] = experiment.DIMENSIONS,
    smac_seeds: tuple[int, ...] = cross.DEFAULT_EVALUATION_SMAC_SEEDS,
) -> tuple[CrossDimensionConfigurationJob, ...]:
    return tuple(
        CrossDimensionConfigurationJob(*arguments)
        for arguments in cross.job_arguments(target_dimensions, smac_seeds)
    )


def smoke_check() -> None:
    assert len(cross.SELECTED_CONFIGURATIONS) == 10
    assert len(cross.CONFIGURATIONS_BY_ID) == 10
    assert {item.source_dimension for item in cross.SELECTED_CONFIGURATIONS} == set(
        experiment.DIMENSIONS
    )
    for dimension in experiment.DIMENSIONS:
        selected = [
            item
            for item in cross.SELECTED_CONFIGURATIONS
            if item.source_dimension == dimension
        ]
        assert len(selected) == 2
        assert {item.source_checkpoint for item in selected} == {100, 500}
        for item in selected:
            cross.validate_source(item)
    planned = jobs()
    assert len(planned) == 50 and len(set(planned)) == 50
    assert len(jobs(smac_seeds=experiment.SMAC_SEEDS)) == 250
    assert experiment.RANDOM_DESIGN_PROBABILITY == 0.0
    assert experiment.base.LOCAL_SMAC_ROOT.is_dir()
    slurm_setup = common_parameters("smoke", 1)["slurm_setup"]
    assert any(str(experiment.base.LOCAL_SMAC_ROOT) in line for line in slurm_setup)
    print("Smoke check passed: 10 sourced configurations and 50 default jobs.")


def submit(
    target_dimensions: tuple[int, ...], smac_seeds: tuple[int, ...]
) -> list[submitit.Job]:
    planned = jobs(target_dimensions, smac_seeds)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=LOG_DIRECTORY)
    executor.update_parameters(
        **common_parameters(
            "SynthACtic_O1_Dim_LLMFixed",
            min(ARRAY_PARALLELISM, len(planned)),
        )
    )
    with executor.batch():
        submissions = [executor.submit(job) for job in planned]
    array_id = submissions[0].job_id.rsplit("_", 1)[0]
    print(f"Submitted {len(submissions)} cross-dimension jobs as array {array_id}.")
    return submissions


def print_summary(
    target_dimensions: tuple[int, ...],
    smac_seeds: tuple[int, ...],
    list_jobs: bool,
) -> None:
    planned = jobs(target_dimensions, smac_seeds)
    print(
        f"Cross-dimension fixed-configuration jobs: {len(planned)}; "
        f"targets={target_dimensions}; configurations={len(cross.SELECTED_CONFIGURATIONS)}; "
        f"evaluation SMAC seeds={smac_seeds}"
    )
    for item in cross.SELECTED_CONFIGURATIONS:
        print(
            f"{item.identifier}: source seed={item.source_smac_seed}, "
            f"checkpoint={item.source_checkpoint}, settings={item.settings.to_dict()}"
        )
    if list_jobs:
        for index, job in enumerate(planned):
            print(
                f"job[{index:03d}] target_dimension={job.target_dimension} "
                f"configuration={job.configuration_id} smac_seed={job.smac_seed}"
            )


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
        help="Target benchmark dimensions.",
    )
    parser.add_argument(
        "--smac-seeds",
        nargs="+",
        type=int,
        choices=experiment.SMAC_SEEDS,
        default=list(cross.DEFAULT_EVALUATION_SMAC_SEEDS),
        help="Evaluation seeds; the default 0 produces the requested 50-job grid.",
    )
    args = parser.parse_args()
    dimensions = tuple(dict.fromkeys(args.dimensions))
    smac_seeds = tuple(dict.fromkeys(args.smac_seeds))
    print_summary(dimensions, smac_seeds, args.list_jobs)
    if args.smoke_check:
        smoke_check()
    if args.submit or not (args.smoke_check or args.list_jobs):
        submit(dimensions, smac_seeds)


if __name__ == "__main__":
    main()

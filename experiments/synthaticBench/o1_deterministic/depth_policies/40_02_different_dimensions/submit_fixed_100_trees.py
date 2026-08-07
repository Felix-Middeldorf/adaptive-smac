#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Submit all dimension/depth/seed fixed controls with 100 RF trees."""


from __future__ import annotations

import argparse
from dataclasses import dataclass

import submitit

import experiment
from submit_experiment import ARRAY_PARALLELISM, common_parameters


N_TREES = experiment.COMPARISON_FIXED_N_TREES
LOG_DIRECTORY = experiment.HERE / "submitit_logs_fixed_100_trees"


@dataclass(frozen=True)
class FixedDepth100TreesJob:
    dimension: int
    depth: int
    smac_seed: int

    def __call__(self):
        return experiment.run_fixed_depth(
            self.dimension,
            self.depth,
            self.smac_seed,
            n_trees=N_TREES,
        )

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def jobs(
    dimensions: tuple[int, ...] = experiment.DIMENSIONS,
) -> tuple[FixedDepth100TreesJob, ...]:
    return tuple(
        FixedDepth100TreesJob(*args) for args in experiment.fixed_jobs(dimensions)
    )


def smoke_check() -> None:
    planned = jobs()
    assert N_TREES == 100
    assert len(planned) == 125 and len(set(planned)) == 125
    assert len(jobs((50, 100))) == 50
    for job in planned:
        output = experiment.fixed_output_directory(
            job.dimension,
            job.depth,
            job.smac_seed,
            n_trees=N_TREES,
        )
        assert output.parent.name == f"fixed_depth_{job.depth}_{N_TREES}_trees"
    print("Smoke check passed: 125 distinct fixed-depth jobs with 100 trees.")


def submit(dimensions: tuple[int, ...]) -> list[submitit.Job]:
    planned = jobs(dimensions)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=LOG_DIRECTORY)
    executor.update_parameters(
        **common_parameters(
            "SynthACtic_O1_Dim_Fixed100Trees",
            min(ARRAY_PARALLELISM, len(planned)),
        )
    )
    with executor.batch():
        submissions = [executor.submit(job) for job in planned]
    array_id = submissions[0].job_id.rsplit("_", 1)[0]
    print(f"Submitted {len(submissions)} fixed 100-tree jobs as array {array_id}.")
    return submissions


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
        help="Submit/list only these objective dimensions.",
    )
    args = parser.parse_args()
    dimensions = tuple(dict.fromkeys(args.dimensions))
    planned = jobs(dimensions)
    print(
        f"Fixed-depth 100-tree jobs: {len(planned)}; dimensions={dimensions}; "
        f"depths={experiment.FIXED_DEPTHS}; seeds={experiment.SMAC_SEEDS}"
    )
    if args.list_jobs:
        for index, job in enumerate(planned):
            print(
                f"fixed100[{index:03d}] dimension={job.dimension} "
                f"depth={job.depth} smac_seed={job.smac_seed}"
            )
    if args.smoke_check:
        smoke_check()
    if args.submit or not (args.smoke_check or args.list_jobs):
        submit(dimensions)


if __name__ == "__main__":
    main()

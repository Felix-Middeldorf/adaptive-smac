from pathlib import Path
from typing import Optional

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario

from master_utils.benchmarks import hartmann


N_TRIALS = 500
SEEDS = range(5)
RF_DEPTHS = [None, 2, 4, 7, 10, 15, 20, 25]
OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/03_fixed_depths/hartmann_6d_fixed_depths"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


def setting_name(max_depth: Optional[int]) -> str:
    if max_depth is None:
        return "default"
    return f"depth_{max_depth}"


def run_smac(seed: int, max_depth: Optional[int]):
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])

    scenario = Scenario(
        name=setting_name(max_depth),
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    facade_arguments = {
        "scenario": scenario,
        "target_function": hartmann_6d_eval,
    }
    if max_depth is not None:
        facade_arguments["model"] = HPOFacade.get_model(
            scenario=scenario,
            max_depth=max_depth,
        )

    smac = HPOFacade(
        **facade_arguments,
        # overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = smac.runhistory.get_cost(incumbent)

    print(
        f"setting={setting_name(max_depth)}, seed={seed}, "
        f"incumbent_cost={incumbent_cost}"
    )
    print(incumbent)

    return incumbent


executor = submitit.AutoExecutor(
    folder="logs",
    cluster="slurm",
    slurm_max_num_timeout=1000,
)

executor.update_parameters(
    timeout_min=60 * 24,
    slurm_array_parallelism=40,
    cpus_per_task=1,
    mem_gb=2.4,
    slurm_job_name="HartmannRFDepth",
    slurm_setup=["export PYTHONHASHSEED=0"],
    slurm_additional_parameters={"requeue": True},
)

jobs = []
with executor.batch():
    for max_depth in RF_DEPTHS:
        for seed in SEEDS:
            job = executor.submit(run_smac, seed, max_depth)
            jobs.append((setting_name(max_depth), seed, job))

print("submitted_jobs:")
for setting, seed, job in jobs:
    print(f"{setting}, seed={seed}: {job.job_id}")

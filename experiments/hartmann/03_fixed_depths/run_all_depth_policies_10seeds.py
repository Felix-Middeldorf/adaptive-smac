from pathlib import Path
from typing import List, Optional, Tuple

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback import Callback

from master_utils.benchmarks import hartmann


N_TRIALS = 500
N_INITIAL_DESIGN_TRIALS = 60
TRIALS_PER_ROTATING_DEPTH = 30
SEEDS = range(10)
FIXED_DEPTHS = [2, 4, 7, 10, 15, 20, 25]
POLICIES = [
    "default",
    *[f"depth_{depth}" for depth in FIXED_DEPTHS],
    "rotating_3_8_12_default",
    "staged_4_until_150_10_until_250_then_20",
]
OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/03_fixed_depths/"
    "hartmann_6d_all_depth_policies_10seeds"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


class RotatingDepthCallback(Callback):
    """Rotate 3 -> 8 -> 12 -> default every 30 model-based trials."""

    def __init__(self, default_depth: int) -> None:
        super().__init__()
        self._depth_cycle = (3, 8, 12, default_depth)
        self._last_depth = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        model_based_trials = max(0, n_trials - N_INITIAL_DESIGN_TRIALS)
        phase = model_based_trials // TRIALS_PER_ROTATING_DEPTH
        depth = self._depth_cycle[phase % len(self._depth_cycle)]
        config_selector._model._rf_opts["max_depth"] = depth

        if depth != self._last_depth:
            label = "default" if depth == self._depth_cycle[-1] else depth
            print(
                f"[RotatingDepthCallback] trials={n_trials}, "
                f"model_based_trials={model_based_trials}, max_depth={label}"
            )
            self._last_depth = depth


class StagedDepthCallback(Callback):
    """Use depth 4 through trial 150, depth 10 through 250, then 20."""

    def __init__(self) -> None:
        super().__init__()
        self._last_depth = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        if n_trials < 150:
            depth = 4
        elif n_trials < 250:
            depth = 10
        else:
            depth = 20

        config_selector._model._rf_opts["max_depth"] = depth

        if depth != self._last_depth:
            print(f"[StagedDepthCallback] trials={n_trials}, max_depth={depth}")
            self._last_depth = depth


def model_and_callbacks(
    scenario: Scenario,
    policy: str,
) -> Tuple[Optional[object], List[Callback]]:
    if policy == "default":
        return None, []

    if policy.startswith("depth_"):
        depth = int(policy.removeprefix("depth_"))
        return HPOFacade.get_model(scenario=scenario, max_depth=depth), []

    model = HPOFacade.get_model(scenario=scenario)
    if policy == "rotating_3_8_12_default":
        default_depth = model._rf_opts["max_depth"]
        return model, [RotatingDepthCallback(default_depth=default_depth)]

    if policy == "staged_4_until_150_10_until_250_then_20":
        model._rf_opts["max_depth"] = 4
        return model, [StagedDepthCallback()]

    raise ValueError(f"Unknown policy: {policy!r}")


def run_smac(seed: int, policy: str):
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])

    scenario = Scenario(
        name=policy,
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model, callbacks = model_and_callbacks(scenario, policy)
    facade_arguments = {
        "scenario": scenario,
        "target_function": hartmann_6d_eval,
        "callbacks": callbacks,
    }
    if model is not None:
        facade_arguments["model"] = model

    smac = HPOFacade(
        **facade_arguments,
        # overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = smac.runhistory.get_cost(incumbent)

    print(f"policy={policy}, seed={seed}, incumbent_cost={incumbent_cost}")
    print(incumbent)
    return incumbent


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder="logs",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_array_parallelism=100,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannAllDepths",
        slurm_setup=["export PYTHONHASHSEED=0"],
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for policy in POLICIES:
            for seed in SEEDS:
                job = executor.submit(run_smac, seed, policy)
                jobs.append((policy, seed, job))

    print("submitted_jobs:")
    for policy, seed, job in jobs:
        print(f"{policy}, seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

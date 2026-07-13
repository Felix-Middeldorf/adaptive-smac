from pathlib import Path

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback import Callback

from master_utils.benchmarks import hartmann


N_TRIALS = 500
N_INITIAL_DESIGN_TRIALS = 60
TRIALS_PER_DEPTH = 30
SEEDS = range(5)
OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/03_fixed_depths/hartmann_6d_rotating_depths"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


class RotatingDepthCallback(Callback):
    """Rotate RF depth every 30 model-based trials.

    Hartmann 6D's default initial design contains 60 Sobol trials. The first
    surrogate fitted after that design uses depth 3. Subsequent phases use
    depths 8, 12, and SMAC's default depth before the cycle starts again.
    """

    def __init__(
        self,
        default_depth: int,
        initial_design_trials: int = N_INITIAL_DESIGN_TRIALS,
        trials_per_depth: int = TRIALS_PER_DEPTH,
    ) -> None:
        super().__init__()
        self._depth_cycle = (3, 8, 12, default_depth)
        self._initial_design_trials = initial_design_trials
        self._trials_per_depth = trials_per_depth
        self._last_depth = None

    def on_next_configurations_start(self, config_selector) -> None:
        model = config_selector._model
        n_trials = len(config_selector._runhistory)
        model_based_trials = max(0, n_trials - self._initial_design_trials)
        phase = model_based_trials // self._trials_per_depth
        depth = self._depth_cycle[phase % len(self._depth_cycle)]

        model._rf_opts["max_depth"] = depth

        if depth != self._last_depth:
            depth_label = "default" if depth == self._depth_cycle[-1] else depth
            print(
                f"[RotatingDepthCallback] trials={n_trials}, "
                f"model_based_trials={model_based_trials}, max_depth={depth_label}"
            )
            self._last_depth = depth


def run_smac(seed: int):
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])

    scenario = Scenario(
        name="rotating_3_8_12_default",
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model = HPOFacade.get_model(scenario=scenario)
    default_depth = model._rf_opts["max_depth"]
    callback = RotatingDepthCallback(default_depth=default_depth)

    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        model=model,
        callbacks=[callback],
        # overwrite=True,
    )

    incumbent = smac.optimize()
    incumbent_cost = smac.runhistory.get_cost(incumbent)
    print(f"seed={seed}, incumbent_cost={incumbent_cost}")
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
        slurm_array_parallelism=5,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannRotateDepth",
        slurm_setup=["export PYTHONHASHSEED=0"],
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for seed in SEEDS:
            job = executor.submit(run_smac, seed)
            jobs.append((seed, job))

    print("submitted_jobs:")
    for seed, job in jobs:
        print(f"seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

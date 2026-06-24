from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import List, Tuple

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.callback import Callback

from master_utils.benchmarks import hartmann


LEAF_VALUES = [1, 2, 3, 5]
POLICY_LENGTH = 4
SEEDS = range(5)
N_TRIALS = 400
TRIALS_PER_BLOCK = 100

# Keep this aligned with the offline 100-trial block experiments: 25 Sobol
# configs, then model-based search starts at trial 26.
N_INITIAL_DESIGN_CONFIGS = 25

# Set to a small integer, e.g. 10, if you want a quick submit/debug run first.
MAX_POLICIES: int | None = None
RUNS_PER_SLURM_JOB = 16
SLURM_PARTITION = "c23ms"

OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/06_brute_force_leaf_size/"
    "hartmann_6d_bruteforce_leaf_policies_400_trials_leaf_1_2_3_5"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


def all_leaf_policies() -> List[Tuple[int, ...]]:
    policies = list(itertools.product(LEAF_VALUES, repeat=POLICY_LENGTH))
    if MAX_POLICIES is not None:
        return policies[:MAX_POLICIES]
    return policies


def policy_name(policy: Tuple[int, ...]) -> str:
    return "leaf_policy_" + "_".join(str(value) for value in policy)


class BlockwiseMinSamplesLeafCallback(Callback):
    """Switch min_samples_leaf every 100 trials according to a fixed policy."""

    def __init__(self, policy: Tuple[int, ...]) -> None:
        super().__init__()
        if len(policy) != POLICY_LENGTH:
            raise ValueError(f"Expected policy length {POLICY_LENGTH}, got {policy!r}")
        self._policy = policy
        self._last_leaf = None

    def on_next_configurations_start(self, config_selector) -> None:
        n_trials = len(config_selector._runhistory)
        block_idx = min(n_trials // TRIALS_PER_BLOCK, len(self._policy) - 1)
        min_samples_leaf = self._policy[block_idx]
        config_selector._model._rf_opts["min_samples_leaf"] = min_samples_leaf

        if min_samples_leaf != self._last_leaf:
            print(
                f"[BlockwiseMinSamplesLeafCallback] trials={n_trials}, "
                f"block={block_idx + 1}, min_samples_leaf={min_samples_leaf}"
            )
            self._last_leaf = min_samples_leaf


def make_configspace(seed: int) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])
    return configspace


def ordered_costs_from_runhistory(runhistory) -> List[float]:
    ordered_trials = sorted(
        runhistory.items(),
        key=lambda item: (item[1].starttime, item[1].endtime),
    )
    return [float(value.cost) for _, value in ordered_trials]


def best_so_far(costs: List[float]) -> List[float]:
    best = []
    current = float("inf")
    for cost in costs:
        current = min(current, cost)
        best.append(current)
    return best


def run_smac(seed: int, policy: Tuple[int, ...]):
    configspace = make_configspace(seed)
    name = policy_name(policy)

    scenario = Scenario(
        name=name,
        output_directory=OUTPUT_DIRECTORY,
        configspace=configspace,
        deterministic=True,
        n_trials=N_TRIALS,
        seed=seed,
    )

    model = HPOFacade.get_model(
        scenario=scenario,
        min_samples_leaf=policy[0],
    )
    initial_design = HPOFacade.get_initial_design(
        scenario=scenario,
        n_configs=N_INITIAL_DESIGN_CONFIGS,
    )
    callbacks = [BlockwiseMinSamplesLeafCallback(policy)]

    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        model=model,
        initial_design=initial_design,
        callbacks=callbacks,
        #overwrite=True,
    )
    incumbent = smac.optimize()
    incumbent_cost = float(smac.runhistory.get_cost(incumbent))

    costs = ordered_costs_from_runhistory(smac.runhistory)
    trajectory = best_so_far(costs)
    milestones = {
        str(n): trajectory[n - 1]
        for n in [100, 200, 300, 400]
        if len(trajectory) >= n
    }

    summary = {
        "seed": seed,
        "policy": list(policy),
        "policy_name": name,
        "n_trials": N_TRIALS,
        "trials_per_block": TRIALS_PER_BLOCK,
        "n_initial_design_configs": N_INITIAL_DESIGN_CONFIGS,
        "final_incumbent_cost": incumbent_cost,
        "milestone_best_costs": milestones,
        "incumbent": dict(incumbent),
        "run_dir": str(scenario.output_directory),
    }

    with open(scenario.output_directory / "policy_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"policy={policy}, seed={seed}, incumbent_cost={incumbent_cost}")
    print(incumbent)
    return incumbent


def chunks(items: List[Tuple[int, Tuple[int, ...]]], chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def run_smac_batch(job_specs: List[Tuple[int, Tuple[int, ...]]]):
    results = []
    print(f"Running batch with {len(job_specs)} SMAC runs.")
    for batch_idx, (seed, policy) in enumerate(job_specs, start=1):
        print(
            f"[batch {batch_idx}/{len(job_specs)}] "
            f"policy={policy_name(policy)}, seed={seed}"
        )
        results.append(run_smac(seed=seed, policy=policy))

    return results


def submit_jobs() -> None:
    policies = all_leaf_policies()
    job_specs = [(seed, policy) for policy in policies for seed in SEEDS]
    job_batches = list(chunks(job_specs, RUNS_PER_SLURM_JOB))
    n_runs = len(job_specs)
    n_slurm_jobs = len(job_batches)
    print(
        f"Submitting {n_runs} SMAC runs as {n_slurm_jobs} Slurm jobs "
        f"({len(policies)} policies x {len(list(SEEDS))} seeds, "
        f"{RUNS_PER_SLURM_JOB} runs per Slurm job)."
    )

    executor = submitit.AutoExecutor(
        folder="logs_bruteforce_leaf_policies",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        slurm_array_parallelism=80,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannLeafBruteForce",
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for batch_idx, job_batch in enumerate(job_batches):
            job = executor.submit(run_smac_batch, job_batch)
            jobs.append((batch_idx, job_batch, job))

    print("submitted_jobs:")
    for batch_idx, job_batch, job in jobs:
        first_seed, first_policy = job_batch[0]
        last_seed, last_policy = job_batch[-1]
        print(
            f"batch={batch_idx}, size={len(job_batch)}, "
            f"first={policy_name(first_policy)}/seed={first_seed}, "
            f"last={policy_name(last_policy)}/seed={last_seed}: {job.job_id}"
        )


if __name__ == "__main__":
    submit_jobs()

    

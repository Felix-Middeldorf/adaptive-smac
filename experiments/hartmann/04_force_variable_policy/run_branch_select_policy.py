from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import submitit
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from smac.runhistory.runhistory import RunHistory

from master_utils.benchmarks import hartmann


DEPTHS = [2, 5, 10, 15, 20]
SEEDS = range(5)
N_BLOCKS = 5
TRIALS_PER_BLOCK = 100
N_SELECTED_TRIALS = N_BLOCKS * TRIALS_PER_BLOCK

OUTPUT_DIRECTORY = Path(
    "/home/io632776/experiments/adaptive-smac/"
    "experiments/hartmann/04_force_variable_policy/"
    "hartmann_6d_branch_select_policy"
)


def hartmann_6d_eval(config: Configuration, seed: int = 0) -> float:
    return hartmann([config[f"x{i}"] for i in range(1, 7)], 6)


def make_configspace(seed: int) -> ConfigurationSpace:
    configspace = ConfigurationSpace(seed=seed)
    configspace.add([Float(f"x{i}", (0, 1)) for i in range(1, 7)])
    return configspace


def clone_runhistory(runhistory: RunHistory, configspace: ConfigurationSpace) -> RunHistory:
    """Deep-copy a SMAC runhistory through SMAC's JSON serialization."""
    tmp_file = tempfile.NamedTemporaryFile(
        prefix="hartmann_branch_select_rh_",
        suffix=".json",
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()

    runhistory.save(tmp_path)

    cloned = RunHistory()
    cloned.load(tmp_path, configspace=configspace)
    tmp_path.unlink(missing_ok=True)
    return cloned


def load_runhistory_into_smac(smac: HPOFacade, source: RunHistory, configspace: ConfigurationSpace) -> None:
    """Warmstart a freshly built SMAC object with an existing selected runhistory."""
    if source.finished == 0:
        return

    warmstart_path = smac.scenario.output_directory / "warmstart_runhistory.json"
    warmstart_path.parent.mkdir(parents=True, exist_ok=True)
    source.save(warmstart_path)

    # Important: load into the existing RunHistory object. SMAC components already
    # hold references to this object, so replacing it would leave stale references.
    smac.runhistory.load(warmstart_path, configspace=configspace)


def incumbent_from_runhistory(runhistory: RunHistory) -> tuple[Configuration, float]:
    configs = runhistory.get_configs()
    if len(configs) == 0:
        raise ValueError("Cannot find an incumbent in an empty runhistory.")

    best_config = min(configs, key=runhistory.get_cost)
    return best_config, float(runhistory.get_cost(best_config))


def run_branch(
    seed: int,
    block_idx: int,
    depth: int,
    selected_runhistory: RunHistory,
) -> dict[str, Any]:
    configspace = make_configspace(seed)
    n_trials = selected_runhistory.finished + TRIALS_PER_BLOCK
    block_dir = OUTPUT_DIRECTORY / f"seed_{seed}" / f"block_{block_idx + 1:02d}"
    branch_name = f"depth_{depth}"

    if (block_dir / branch_name).exists():
        shutil.rmtree(block_dir / branch_name)

    scenario = Scenario(
        name=branch_name,
        output_directory=block_dir,
        configspace=configspace,
        deterministic=True,
        n_trials=n_trials,
        seed=seed,
    )

    model = HPOFacade.get_model(scenario=scenario, max_depth=depth)
    smac = HPOFacade(
        scenario=scenario,
        target_function=hartmann_6d_eval,
        model=model,
        overwrite=True,
    )
    load_runhistory_into_smac(smac, selected_runhistory, configspace)

    smac.optimize()
    incumbent, incumbent_cost = incumbent_from_runhistory(smac.runhistory)

    return {
        "seed": seed,
        "block": block_idx + 1,
        "depth": depth,
        "branch_dir": str(scenario.output_directory),
        "n_trials": smac.runhistory.finished,
        "incumbent_cost": incumbent_cost,
        "incumbent": dict(incumbent),
    }


def run_seed(seed: int) -> dict[str, Any]:
    """Run one complete offline branch-and-select trajectory for one seed.

    Selected trajectory length: 500 trials.
    Actual function evaluations: 5 depths * 100 trials * 5 blocks = 2500.
    """
    seed_dir = OUTPUT_DIRECTORY / f"seed_{seed}"
    if seed_dir.exists():
        shutil.rmtree(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)

    selected_runhistory = RunHistory()
    all_blocks: list[dict[str, Any]] = []
    selected_depths: list[int] = []

    for block_idx in range(N_BLOCKS):
        print(f"\n[seed={seed}] block {block_idx + 1}/{N_BLOCKS}")

        branch_results = [
            run_branch(
                seed=seed,
                block_idx=block_idx,
                depth=depth,
                selected_runhistory=selected_runhistory,
            )
            for depth in DEPTHS
        ]

        best_branch = min(branch_results, key=lambda result: result["incumbent_cost"])
        selected_depths.append(int(best_branch["depth"]))

        best_branch_rh = RunHistory()
        best_branch_scenario = Scenario.load(Path(best_branch["branch_dir"]))
        best_branch_rh.load(
            Path(best_branch["branch_dir"]) / "runhistory.json",
            configspace=best_branch_scenario.configspace,
        )
        selected_runhistory = clone_runhistory(best_branch_rh, best_branch_scenario.configspace)

        print(
            f"[seed={seed}] selected depth={best_branch['depth']} "
            f"after {selected_runhistory.finished} trials, "
            f"incumbent_cost={best_branch['incumbent_cost']:.12g}"
        )

        all_blocks.append(
            {
                "block": block_idx + 1,
                "start_trial": block_idx * TRIALS_PER_BLOCK + 1,
                "end_trial": (block_idx + 1) * TRIALS_PER_BLOCK,
                "branches": branch_results,
                "selected": best_branch,
            }
        )

    final_incumbent, final_cost = incumbent_from_runhistory(selected_runhistory)
    selected_runhistory.save(seed_dir / "selected_runhistory.json")

    summary = {
        "seed": seed,
        "depths": DEPTHS,
        "trials_per_block": TRIALS_PER_BLOCK,
        "n_blocks": N_BLOCKS,
        "selected_depths": selected_depths,
        "selected_trials": N_SELECTED_TRIALS,
        "actual_evaluations": len(DEPTHS) * TRIALS_PER_BLOCK * N_BLOCKS,
        "final_incumbent_cost": final_cost,
        "final_incumbent": dict(final_incumbent),
        "blocks": all_blocks,
    }

    with open(seed_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"[seed={seed}] selected_depths={selected_depths}")
    print(f"[seed={seed}] final_incumbent_cost={final_cost:.12g}")
    return summary


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder="logs_branch_select",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_array_parallelism=20,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="HartmannBranchSelect",
        slurm_additional_parameters={"requeue": True},
    )

    jobs = []
    with executor.batch():
        for seed in SEEDS:
            job = executor.submit(run_seed, seed)
            jobs.append((seed, job))

    print("submitted_jobs:")
    for seed, job in jobs:
        print(f"seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

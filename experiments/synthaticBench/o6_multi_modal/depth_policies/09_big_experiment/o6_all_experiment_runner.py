from __future__ import annotations

from typing import Any

import o6_big_experiment_runner as big
import o6_consensus_ramp_runner as consensus

BENCHMARK_SEEDS = big.BENCHMARK_SEEDS
SMAC_SEEDS = big.SMAC_SEEDS
PYTHONHASHSEED = big.PYTHONHASHSEED
DIMENSION = big.DIMENSION
N_INSTANCES = big.N_INSTANCES
N_TRIALS = big.N_TRIALS
RANDOM_DESIGN_PROBABILITY = big.RANDOM_DESIGN_PROBABILITY
HERE = big.HERE


def validate_matching_experiment_configuration() -> None:
    shared_names = (
        "BENCHMARK_SEEDS",
        "SMAC_SEEDS",
        "PYTHONHASHSEED",
        "DIMENSION",
        "N_INSTANCES",
        "N_TRIALS",
        "RANDOM_DESIGN_PROBABILITY",
        "PROBLEM_CONFIG",
        "OUTPUT_DIRECTORY",
    )
    mismatches = {
        name: (getattr(big, name), getattr(consensus, name))
        for name in shared_names
        if getattr(big, name) != getattr(consensus, name)
    }
    if mismatches:
        raise RuntimeError(
            "The main and consensus experiments do not share one "
            f"configuration: {mismatches}"
        )


def policy_count() -> int:
    return len(big.all_policies()) + 1


def run_complete_seed_pair(
    benchmark_seed: int,
    smac_seed: int,
) -> dict[str, Any]:
    """Run all main policies and the consensus policy in one Slurm job."""
    validate_matching_experiment_configuration()
    print(
        f"Running {policy_count()} policies for "
        f"benchmark_seed={benchmark_seed}, smac_seed={smac_seed}."
    )

    main_results = big.run_seed_pair(benchmark_seed, smac_seed)
    main_policy_count = len(main_results)
    del main_results

    consensus_result = consensus.run_consensus_ramp(
        benchmark_seed,
        smac_seed,
    )
    return {
        "benchmark_seed": benchmark_seed,
        "smac_seed": smac_seed,
        "main_policies": main_policy_count,
        "consensus_policy": consensus_result["policy"],
        "total_policies": main_policy_count + 1,
    }


validate_matching_experiment_configuration()

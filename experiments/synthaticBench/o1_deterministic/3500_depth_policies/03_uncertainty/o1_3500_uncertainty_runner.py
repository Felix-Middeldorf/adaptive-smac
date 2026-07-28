from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

DEPTHS = (5, 10, 15, 20, 25, 30)
BENCHMARK_SEEDS = (40, 42)
SMAC_SEEDS = tuple(range(10))
DIMENSION = 15
N_INSTANCES = 15
N_TRIALS = 3_500
PYTHONHASHSEED = "12345"
MIN_SAMPLES_LEAF = 1
MIN_SAMPLES_SPLIT = 1
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1

HERE = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = HERE / "smac_output"
SOURCE_RUNNER = (
    HERE.parents[1]
    / "depth_policies/11_uncertainty/o1_uncertainty_runner.py"
)


def _load_shared_runner():
    """Load and configure the tested proposal-diagnostics implementation."""
    if not SOURCE_RUNNER.is_file():
        raise FileNotFoundError(f"Shared uncertainty runner not found: {SOURCE_RUNNER}")
    module_name = "_o1_shared_uncertainty_runner"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared runner from {SOURCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    module.DEPTHS = DEPTHS
    module.BENCHMARK_SEEDS = BENCHMARK_SEEDS
    module.SMAC_SEEDS = SMAC_SEEDS
    module.DIMENSION = DIMENSION
    module.N_INSTANCES = N_INSTANCES
    module.N_TRIALS = N_TRIALS
    module.PYTHONHASHSEED = PYTHONHASHSEED
    module.MIN_SAMPLES_LEAF = MIN_SAMPLES_LEAF
    module.MIN_SAMPLES_SPLIT = MIN_SAMPLES_SPLIT
    module.RANDOM_DESIGN_PROBABILITY = RANDOM_DESIGN_PROBABILITY
    module.EXPERIMENT_VERSION = EXPERIMENT_VERSION
    module.OUTPUT_DIRECTORY = OUTPUT_DIRECTORY
    return module


_SHARED = _load_shared_runner()
PROPOSAL_SCHEMA_VERSION = _SHARED.PROPOSAL_SCHEMA_VERSION
INSTANCE_SEED = _SHARED.INSTANCE_SEED
INSTANCE_STD = _SHARED.INSTANCE_STD
PROBLEM_CONFIG = _SHARED.PROBLEM_CONFIG


def make_instance_map() -> dict[str, float]:
    return _SHARED.make_instance_map()


def output_directory(benchmark_seed: int, smac_seed: int, depth: int) -> Path:
    return _SHARED.output_directory(benchmark_seed, smac_seed, depth)


def trajectory_is_complete(benchmark_seed: int, smac_seed: int, depth: int) -> bool:
    return _SHARED.trajectory_is_complete(benchmark_seed, smac_seed, depth)


def run_fixed_depth(
    benchmark_seed: int,
    smac_seed: int,
    depth: int,
) -> dict[str, Any]:
    return _SHARED.run_fixed_depth(benchmark_seed, smac_seed, depth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-seed", type=int, required=True)
    parser.add_argument("--smac-seed", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    args = parser.parse_args()
    run_fixed_depth(args.benchmark_seed, args.smac_seed, args.depth)


if __name__ == "__main__":
    main()

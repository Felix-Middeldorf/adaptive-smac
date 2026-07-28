"""Definition of the deterministic CPLEX RCW fixed-depth experiment."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from fixed_depth_experiment import ExperimentDefinition

DEFINITION = ExperimentDefinition(
    benchmark_key="cplex_rcw",
    initials="cr",
    directory=HERE,
    initial_directory=HERE.parent / "01_initial_cr",
    deterministic=True,
)

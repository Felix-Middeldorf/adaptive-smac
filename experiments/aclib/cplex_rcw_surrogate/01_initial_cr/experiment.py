"""Definition of the CPLEX RCW fixed-depth experiment."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ACLIB_EXPERIMENT_ROOT = HERE.parents[1]
if str(ACLIB_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_EXPERIMENT_ROOT))

from fixed_depth_experiment import ExperimentDefinition


DEFINITION = ExperimentDefinition(
    benchmark_key="cplex_rcw",
    initials="cr",
    directory=HERE,
)

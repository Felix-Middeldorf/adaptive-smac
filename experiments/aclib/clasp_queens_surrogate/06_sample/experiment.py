"""Definition of the Clasp Queens random-sampling experiment."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from random_sampling_experiment import RandomSamplingDefinition


DEFINITION = RandomSamplingDefinition(
    benchmark_key="clasp_queens",
    initials="cq",
    directory=HERE,
)


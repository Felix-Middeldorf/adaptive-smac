"""Raw native-SMAC timing definition for Clasp Queens."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from raw_smac_experiment import RawExperimentDefinition


DEFINITION = RawExperimentDefinition(
    benchmark_key="clasp_queens",
    initials="cq",
    directory=HERE,
    initial_choice_file=HERE.parent / "01_initial_cq" / "initial_config.json",
)

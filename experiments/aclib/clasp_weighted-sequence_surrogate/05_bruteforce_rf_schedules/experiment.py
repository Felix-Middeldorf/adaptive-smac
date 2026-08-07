"""Definition of the Clasp weighted-sequence RF schedule grid."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from rf_schedule_experiment import RFScheduleExperimentDefinition

DEFINITION = RFScheduleExperimentDefinition(
    benchmark_key="clasp_weighted",
    initials="cw",
    directory=HERE,
    initial_directory=HERE.parent / "01_initial_cw",
)

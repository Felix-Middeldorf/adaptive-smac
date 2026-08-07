"""Raw native-SMAC timing definition for Lingeling CircuitFuzz."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from raw_smac_experiment import RawExperimentDefinition


DEFINITION = RawExperimentDefinition(
    benchmark_key="lingeling_circuitfuzz",
    initials="lc",
    directory=HERE,
    initial_choice_file=HERE.parent / "01_initial_lc" / "initial_config.json",
)

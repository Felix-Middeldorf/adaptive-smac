"""Raw native-SMAC CPLEX RCW runs restricted to 50 training instances."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from raw_smac_experiment import RawExperimentDefinition


DEFINITION = RawExperimentDefinition(
    benchmark_key="cplex_rcw",
    initials="cr50",
    directory=HERE,
    initial_choice_file=HERE.parent / "01_initial_cr" / "initial_config.json",
    training_instance_limit=50,
)

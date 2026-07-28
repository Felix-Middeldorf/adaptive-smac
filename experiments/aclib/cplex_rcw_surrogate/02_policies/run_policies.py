#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACLIB_EXPERIMENT_ROOT = HERE.parents[1]
if str(ACLIB_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_EXPERIMENT_ROOT))

from policy_experiment import PolicyExperimentDefinition
from policy_submit import main


DEFINITION = PolicyExperimentDefinition(
    benchmark_key="cplex_rcw",
    initials="cr",
    directory=HERE,
    initial_directory=HERE.parent / "01_initial_cr",
)


if __name__ == "__main__":
    main(DEFINITION)

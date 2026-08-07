#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Validate Clasp weighted-sequence against its official archived runs."""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ACLIB_ROOT = HERE.parents[1]
if str(ACLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(ACLIB_ROOT))

from minimal_surrogate_validation import ValidationDefinition, cli


DEFINITION = ValidationDefinition(
    benchmark_key="clasp_weighted",
    archive_name="clasp_weighted-sequence",
    archive_configspace="clasp-option-config.txt",
    directory=HERE,
)


if __name__ == "__main__":
    cli(DEFINITION)

#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the raw Lingeling CircuitFuzz timing matrix."""

from experiment import DEFINITION
from raw_smac_submit import main


if __name__ == "__main__":
    main(DEFINITION)

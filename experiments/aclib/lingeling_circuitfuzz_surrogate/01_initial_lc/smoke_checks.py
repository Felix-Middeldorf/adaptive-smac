#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Validate the Lingeling CircuitFuzz experiment without submitting jobs."""

from experiment import DEFINITION
from fixed_depth_submit import smoke_check


if __name__ == "__main__":
    smoke_check(DEFINITION)

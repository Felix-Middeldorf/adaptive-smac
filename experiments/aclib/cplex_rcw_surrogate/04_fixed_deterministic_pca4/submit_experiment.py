#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the deterministic CPLEX RCW PCA=4 fixed-depth matrix."""

from experiment import DEFINITION
from fixed_depth_submit import main

if __name__ == "__main__":
    main(DEFINITION)

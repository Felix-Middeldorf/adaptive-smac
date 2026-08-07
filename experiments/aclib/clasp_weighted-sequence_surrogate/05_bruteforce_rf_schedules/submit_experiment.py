#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the 150-job Clasp weighted-sequence RF schedule matrix."""

from experiment import DEFINITION
from rf_schedule_submit import main

if __name__ == "__main__":
    main(DEFINITION)

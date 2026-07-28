# O1 fixed depths over 7,500 trials

This experiment compares five fixed random-forest depths on a larger O1
configuration with:

- fixed depths: 5, 10, 15, 20, and 25
- benchmark seeds: 40 and 41
- SMAC seeds: 0-4
- trials per run: 7,500
- dimension: 50
- deterministic instances: 100 offsets drawn from `N(0, 2)` with seed 0
- `min_samples_leaf`: 1
- `min_samples_split`: 1
- random-design probability: 0.10

The Cartesian product contains 50 SMAC runs. Each run is submitted as its own
Slurm job, with at most 50 jobs running concurrently. Jobs request one CPU,
4 GB of memory, and 24 hours on `c23ms`. Completed, metadata-valid trajectories
are skipped if the launcher is run again.

Validate the design without submitting:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/7500_depth_policies/01_initial/submit_depths.py \
  --dry-run
```

Submit all jobs:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/7500_depth_policies/01_initial/submit_depths.py
```

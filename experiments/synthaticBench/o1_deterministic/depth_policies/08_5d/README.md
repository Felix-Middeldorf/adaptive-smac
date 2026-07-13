# O1 5D fixed-depth experiment

- Dimension: 5
- Benchmark/problem seed: 52
- Instances: 10
- Trials per run: 1,500
- Fixed depths: 5, 8, 12, 15, 20
- SMAC seeds: 0–4
- Random-design injection probability: 0%
- Other RF hyperparameters: SMAC defaults

The submission creates one independent Slurm job for every
`fixed depth × SMAC seed` pair: 25 jobs and 25 SMAC runs in total. All 25 jobs
may run concurrently, staying below the limit of 80.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/08_5d/submit_fixed_depths.py
```

Completed metadata-valid trajectories are skipped on resubmission.

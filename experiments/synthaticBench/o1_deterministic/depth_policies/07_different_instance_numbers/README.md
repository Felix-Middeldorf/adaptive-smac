# O1 fixed depths across instance counts

This experiment tests how the number of benchmark instances changes the
performance of fixed-depth RF surrogates.

- Dimension: 10
- Benchmark/problem seed: 52
- Instance counts: 1, 2, 5, 10, 20, 50
- Fixed depths: 5, 8, 12, 15, 20
- SMAC seeds: 0–4
- Trials per run: 2,500
- Random-design injection probability: 0%
- Other RF hyperparameters: SMAC defaults

Instance sets are nested. For example, the two-instance condition contains the
same first instance as the one-instance condition, plus one additional
instance. This makes comparisons across instance counts paired.

One Slurm job runs the five SMAC seeds sequentially for one
`instance count × depth` combination. The submission therefore creates 30
jobs and 150 SMAC runs, staying well below the concurrent-job limit of 80.
Completed metadata-valid trajectories are skipped when a job is restarted.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/07_different_instance_numbers/submit_fixed_depths.py
```

After the jobs finish, run `analyze_instance_counts.ipynb`. It creates a
separate confidence-band best-regret plot and final-regret boxplot for every
instance count.

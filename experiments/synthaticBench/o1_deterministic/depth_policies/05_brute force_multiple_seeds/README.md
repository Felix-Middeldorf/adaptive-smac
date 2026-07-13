# O1 brute-force schedules across benchmark seeds

This experiment runs all 64 depth schedules on O1 benchmark seeds 52–56. Each
schedule uses five paired SMAC seeds per benchmark seed, for 25 runs per policy
and 1,600 dynamic runs total. Four separately implemented fixed depths add 100
runs.

One Slurm job handles all 25 runs for one schedule. Together with four fixed
jobs, this creates 68 jobs and remains below the limit of 80. Batches are
restart-safe: completed, metadata-valid trajectories are skipped.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  'experiments/synthaticBench/o1_deterministic/depth_policies/05_brute force_multiple_seeds/submit_bruteforce_policies.py'

/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  'experiments/synthaticBench/o1_deterministic/depth_policies/05_brute force_multiple_seeds/submit_fixed_depths.py'
```

After completion, run `analyze_multiseed_depth_policies.ipynb`.

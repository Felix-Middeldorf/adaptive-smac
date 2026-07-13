# O1 incrementing-depth policies across benchmark seeds

This experiment uses the same general conditions as `05_brute force_multiple_seeds`:
O1 benchmark seeds 52–56, five paired SMAC seeds per benchmark seed, 1,000
trials, ten instances, and SMAC's default surrogate and random-design settings.

It evaluates 47 policies:

- Family a: increase depth by 1 every 100 trials, starting at depths 4–10
  (7 policies).
- Family b: increase depth by 1 every 200 trials, starting at depths 5–14
  (10 policies).
- Family c: increase depth by 1 every 50 trials, starting at depths 3–10,
  increasing through trial 400 and then holding the reached depth
  (8 policies).
- Family d: increase depth by 1 every 75 trials, starting at depths 5–15
  (11 policies).
- Family e: increase depth by 1 every 150 trials, starting at depths 5–15
  (11 policies).

One Slurm job runs all 25 benchmark/SMAC-seed combinations for one policy.
Together with four fixed-depth baseline jobs, this creates 51 jobs and stays
below the cluster limit of 80. Jobs are restart-safe and skip complete,
metadata-valid trajectories.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  'experiments/synthaticBench/o1_deterministic/depth_policies/05_02_individual_polciies/submit_increment_policies.py'

/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  'experiments/synthaticBench/o1_deterministic/depth_policies/05_02_individual_polciies/submit_fixed_depths.py'
```

After completion, run `analyze_increment_policies.ipynb`.

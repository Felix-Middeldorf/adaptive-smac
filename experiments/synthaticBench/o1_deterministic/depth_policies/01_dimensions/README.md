# O1 dimension experiment

This experiment compares fixed random-forest depths as the O1 search-space
dimension increases.

- dimensions: 5, 10, 25, and 50
- fixed depths: 5, 10, 15, and 20
- benchmark seeds: 40 and 41
- SMAC seeds: 0-4
- trials per run: 1,000
- deterministic instances: 10 shared offsets from `N(0, 2)` using seed 0
- random-design probability: 0
- other random-forest hyperparameters: SMAC defaults

The full design contains 160 SMAC runs. Each Slurm job runs two depths
sequentially for one `dimension x benchmark seed x SMAC seed` setting. This
produces exactly 80 jobs, with at most 80 running concurrently. Completed,
metadata-valid trajectories are skipped when the launcher is run again.

Check the design without submitting:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/01_dimensions/submit_dimensions.py \
  --dry-run
```

Submit:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/01_dimensions/submit_dimensions.py
```

After all trajectories finish, run `analyze_dimensions.ipynb` from the
repository root or this experiment directory. It contains best-regret curves
with 95% confidence intervals and final-regret boxplots for every landscape
and dimension.

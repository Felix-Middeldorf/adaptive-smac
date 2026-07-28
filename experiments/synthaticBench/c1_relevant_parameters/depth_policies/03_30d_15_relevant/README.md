# C1 depth policies, 30D / 15 relevant

Same fixed-depth sweep as `01_initial`, plus staged-depth policies, with:

- total dimension: 30
- relevant quadratic parameters: 15
- irrelevant/noisy parameters: 15
- fixed depths: 3, 6, 9, 12, 15, 20
- staged depths: 3 for trials 1–500, then 20, 15, or 12 for trials 501–1,000
- trials: 1,000
- SMAC seeds: 0–4
- instances: 10
- benchmark seed: 52

The fixed-depth submission creates one Slurm job for each `depth x SMAC seed`
pair: 30 jobs and 30 SMAC runs in total.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/c1_relevant_parameters/depth_policies/03_30d_15_relevant/submit_fixed_depths.py
```

The staged-depth submission creates one Slurm job for each
`depth schedule x SMAC seed` pair: 15 jobs and 15 SMAC runs in total.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/c1_relevant_parameters/depth_policies/03_30d_15_relevant/submit_staged_depths_after_500.py
```

After all jobs finish, run `analyze_depths.ipynb` from this directory or the
repository root.

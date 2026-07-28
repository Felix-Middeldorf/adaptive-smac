# O1 instance-value scale experiment

This experiment measures how the scale of deterministic instance offsets
changes fixed-depth SMAC behavior on SynthACticBench O1.

Setup:

- benchmark seeds: 40 and 41
- SMAC seeds: 0-4
- fixed random-forest depths: 5, 10, 15, and 20
- 10 instances per run
- instance offsets sampled with NumPy from normals with mean 0 and standard
  deviations 2, 20, 200, 2,000, and 20,000
- shared instance RNG seed: 0
- trials per run: 1,000
- random-design probability: 0
- search-space dimension: 10

The same standardized random draws are rescaled for all five distributions,
so differences are attributable to offset scale rather than unrelated instance
samples. The original three scales contain 120 SMAC runs in 60 Slurm jobs. The
additional scales 2 and 20,000 contain 80 SMAC runs in 40 Slurm jobs. Two
depths run sequentially in each job.

Dry-run validation:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/10_instance_values/submit_instance_values.py \
  --dry-run

/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/10_instance_values/submit_additional_instance_values.py \
  --dry-run
```

After the runs finish, execute `analyze_instance_values.ipynb` from the
repository root or this experiment directory.

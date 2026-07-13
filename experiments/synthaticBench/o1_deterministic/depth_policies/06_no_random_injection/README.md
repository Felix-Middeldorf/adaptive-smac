# O1 fixed depths with and without random-design injection

This experiment compares SMAC's default
`AlgorithmConfigurationFacade.get_random_design(..., probability=0.5)` against
`probability=0.0` on fixed-depth random forest surrogates.

- Benchmark: SynthACticBench O1 deterministic objective
- Fixed benchmark/problem seed: `52`
- SMAC seeds: `0, 1, 2, 3, 4`
- Fixed depths: `5, 8, 12, 15, 20`
- Trials per run: `1000`

Each submit script creates 25 Slurm jobs: one job for each
`fixed depth x SMAC seed` pair. Running both scripts creates 50 jobs total.

Submit the no-random-design runs:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/06_no_random_injection/submit_fixed_depths_no_random.py
```

Submit the default 50% random-design runs:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/06_no_random_injection/submit_fixed_depths_default_random.py
```

After both batches finish, run `analyze_random_design_fixed_depths.ipynb`.

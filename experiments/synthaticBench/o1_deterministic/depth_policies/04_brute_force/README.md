# O1 brute-force depth policies

All 64 three-stage schedules over depths `(5, 10, 15, 20)` are evaluated on
SynthACticBench O1 problem seed 52 with five paired SMAC seeds and 1,000 trials.

- Stage 1: trials 1–250
- Stage 2: trials 251–600
- Stage 3: trials 601–1000

Each schedule is one Slurm job that runs five seeds sequentially. The fixed
baselines use four additional jobs, keeping the combined total at 68.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/04_brute_force/submit_bruteforce_policies.py

/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/04_brute_force/submit_fixed_depths.py
```

Run `analyze_bruteforce_depth_policies.ipynb` after all jobs finish.

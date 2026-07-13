# 20D O1 brute-force depth policies without RF size restrictions

All 64 three-stage schedules over depths `(5, 10, 15, 20)` are evaluated on
the 20-dimensional SynthACticBench O1 problem with problem seed 52, five
paired SMAC seeds, and 1,000 trials. Every surrogate explicitly uses
`min_samples_leaf=1` and `min_samples_split=1`.

- Stage 1: trials 1–250
- Stage 2: trials 251–600
- Stage 3: trials 601–1000

Each schedule is one Slurm job that runs five seeds sequentially. Four fixed
depth jobs are submitted separately, for 68 jobs in total.

Do not submit `04_03_20d` and `04_04_20d_no_size` concurrently: together they
would create 136 jobs and exceed the cluster's 80-job limit. Finish or cancel
the other experiment before submitting this one.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/04_04_20d_no_size/submit_bruteforce_policies.py

/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/04_04_20d_no_size/submit_fixed_depths.py
```

Run `analyze_bruteforce_depth_policies.ipynb` after all jobs finish. The
notebook rejects results with the wrong dimension or RF size settings.

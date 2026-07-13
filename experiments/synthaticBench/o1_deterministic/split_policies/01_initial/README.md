# O1 minimum-split policies

This experiment studies SMAC's random-forest `min_samples_split` setting on
SynthACticBench O1 deterministic.

- benchmark seeds: 40-44
- SMAC seeds: 0-4
- trials per run: 1,000
- random-design probability: 0
- maximum tree depth: 2,000
- minimum leaf size: 1
- split values: 1, 2, and 3
- stage boundaries: completed trials 200 and 500

There are 27 unique policies: three fixed controls and all 24 nonconstant
three-stage assignments. Constant stage assignments are represented by the
fixed controls, so no SMAC run is duplicated.

To stay below the 80-job submission limit, the policies are divided into
three shards for each benchmark-seed x SMAC-seed pair. This gives exactly 75
Slurm jobs. Each job runs 9 policies sequentially, for 675 total SMAC runs,
with at most 75 jobs active at once.

Dry-run validation:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/split_policies/01_initial/submit_split_policies.py \
  --dry-run
```

Submit the jobs by running the same command without `--dry-run`. After all
jobs finish, execute `analyze_split_policies.ipynb` from the repository root.

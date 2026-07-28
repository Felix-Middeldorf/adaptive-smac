# O1 leaf-size policy experiment

This experiment compares the complete three-stage `min_samples_leaf` policy
space over leaf sizes 1, 2, and 3. Constant schedules are represented by the
three fixed controls; the other 24 schedules switch at completed trials 200
and 500.

Setup:

- benchmark seeds: 40-44
- SMAC seeds: 0-4
- trials per run: 1,000
- random-design probability: 0
- random-forest maximum depth: 2,000
- `min_samples_split`: fixed at 1
- leaf sizes: 1, 2, and 3
- stage boundaries: completed trials 200 and 500

The 27 policies across 25 benchmark-seed x SMAC-seed pairs give 675 SMAC
runs. Policies are divided into three shards for each seed pair, producing 75
Slurm jobs with 9 sequential runs per job. At most 75 jobs run concurrently.

Dry-run validation:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/leaf_policies/07_bigger_experiment/submit_leaf_policies.py \
  --dry-run
```

After the runs finish, execute `analyze_leaf_policies.ipynb` from the
repository root or this experiment directory.

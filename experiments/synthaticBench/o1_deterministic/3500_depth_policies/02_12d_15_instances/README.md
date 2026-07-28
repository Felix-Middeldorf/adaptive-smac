# O1 deterministic fixed-depth experiment: 12D / 15 instances

This experiment runs fixed random-forest depths `5, 10, 15, 20, 25` on
benchmark seed 40 with SMAC seeds `0..4`. Each run uses 12 dimensions, the same
deterministic 15-instance map, and 3,500 completed SMAC trials.

The setup matches the preceding 3,500-trial experiment: it uses
`AlgorithmConfigurationFacade`, random-design probability `0.0`,
`min_samples_leaf=1`, and `min_samples_split=1`.

The requested matrix contains 25 unique runs:

```text
5 depths × 1 benchmark seed × 5 SMAC seeds = 25 runs
```

The launcher assigns one run to each Slurm array task, runs up to all 25 tasks
in parallel, and gives every task a 90-minute time limit. Creating 75 unique
one-run jobs would require three benchmark seeds or 15 SMAC seeds; duplicating
the same 25 runs would make jobs overwrite each other.

Validate and submit with:

```bash
python submit_fixed_depths.py --dry-run
python submit_fixed_depths.py
```

Completed outputs are validated and skipped when the launcher is rerun.

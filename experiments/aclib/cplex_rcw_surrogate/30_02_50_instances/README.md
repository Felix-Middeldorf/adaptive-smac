# CPLEX RCW raw SMAC runs on 50 training instances

This repeats the `30_raw_run` matrix using one fixed subset: the first 50
entries of CPLEX RCW's canonical `training.txt`. The identical ordered subset
is used by every depth, PCA mode, and SMAC seed; test instances are never used.

Sixteen jobs cross depths 5, 10, 20, and 30 with SMAC seeds 0 and 1 and PCA
disabled or set to 4 components. Each run uses 5,000 trials, 100 RF trees,
split/leaf size 1, 0% random design, SMAC's deterministic flag, and surrogate
quantile seed 0 (the median). The initial configuration is inherited from
`01_initial_cr`.

There are no callbacks, telemetry acquisition wrappers, validation calls, or
custom result files. Results contain only SMAC's native output under
`results/pca_none/` and `results/pca_4/`.

```bash
./submit_experiment.py --smoke-check
./submit_experiment.py --list-jobs
./submit_experiment.py
```

Like `30_raw_run`, the launcher requests 16 hours, 4 GB, and one CPU per job
and leaves the Slurm account unspecified so the personal/default account is
used.

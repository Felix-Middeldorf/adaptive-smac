# Clasp Queens raw SMAC timing run

Sixteen jobs: depths 5, 10, 20 and 30 crossed with SMAC seeds 0 and 1 and
PCA modes disabled and 4 components. Every run has 5,000 trials, 100 RF trees,
split/leaf size 1 and 0% random design. It uses every training instance and no
test instances.

All runs force SMAC's deterministic flag and use surrogate quantile seed 0
(the median prediction). The initial configuration is reproduced from
`01_initial_cq`.

No callbacks, telemetry acquisition wrapper, validation, or custom result
files are used. The `results/pca_none/` and `results/pca_4/` trees contain only
SMAC's native output.

```bash
./submit_experiment.py --smoke-check
./submit_experiment.py --list-jobs
./submit_experiment.py
```

The final command leaves the Slurm account unspecified, using the personal
default account, with 16 hours and 4 GB per job.

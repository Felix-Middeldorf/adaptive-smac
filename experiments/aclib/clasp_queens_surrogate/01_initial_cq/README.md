# Clasp Queens: initial fixed-depth experiment

This experiment submits 15 independent SMAC runs:

- SMAC random-forest depths: 5, 10, 15, 20, and 30
- SMAC seeds: 0, 1, and 2
- 5,000 target trials per run
- 100 trees, minimum split size 1, minimum leaf size 1, and no PCA
- SMAC random-design probability 0
- all 484 training instances and no test instances
- Slurm account `lect0190` and worker `PYTHONHASHSEED=0`

The ACLib default ranked 128th in a 129-configuration screen. It is therefore
not an unusually good starting point and remains the initial configuration.

Each run also writes `configuration_telemetry.jsonl`. Every unique proposal has
an `event_type="proposal"` record containing the SMAC RF's instance-marginalized
PAR10 prediction, variance and standard deviation, current Expected Improvement,
and actual depth of every fitted tree. A separate
`event_type="first_completed_evaluation"` record links it to the runhistory.
The initial pre-model proposal has null model fields with an explicit reason.

Validate and inspect the matrix:

```bash
./smoke_checks.py
./submit_experiment.py --dry-run --list-jobs
```

Submit it:

```bash
./submit_experiment.py
```

Results are written below `results/depth_<depth>/<smac-seed>/`.

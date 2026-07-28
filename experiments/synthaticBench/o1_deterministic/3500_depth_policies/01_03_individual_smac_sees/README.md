# O1 deterministic: individual SMAC seeds and 7,000-trial extension

This directory contains two related pieces of work:

- `analyze_individual_smac_seeds.ipynb` compares existing fixed-depth runs for
  benchmark seeds 42 and 46.
- `submit_experiment.py` launches a new fixed-depth extension for benchmark
  seed 40.

## New 7,000-trial experiment

The extension evaluates fixed depths `5, 10, 15, 20, 25, 30, 40, 50` using
SMAC seeds `0..6`. This gives:

```text
8 depths × 1 benchmark seed × 7 SMAC seeds = 56 SMAC runs
```

Every run is assigned its own Slurm task, producing 56 concurrent tasks. This
is below the configured limit of 70 without splitting or duplicating any SMAC
run.

The experimental setup matches `01_15d_20_instances`:

- SynthACticBench O1 deterministic
- benchmark seed 40
- 15 dimensions
- the same deterministic map of 20 instances (`instance_seed=0`, normal
  standard deviation 2)
- 7,000 completed SMAC trials
- `AlgorithmConfigurationFacade`
- ACFacade defaults except random-design probability `0.0`,
  `min_samples_leaf=1`, and `min_samples_split=1`
- configuration-selector retraining cadence `retrain_after=8`
- `PYTHONHASHSEED=12345`

Each task requests one CPU, 4 GB RAM and six hours on partition `c23ms`.
Submitit requeue/checkpoint support is enabled. A rerun validates and skips a
complete trajectory; an incomplete run is restarted with SMAC's
`overwrite=True` behavior.

Validate the matrix without submitting:

```bash
python submit_experiment.py --dry-run
python submit_experiment.py --list-jobs
```

Submit all 56 tasks:

```bash
python submit_experiment.py
```

Outputs are kept separate from the source experiments:

```text
smac_output/benchmark_seed_40/fixed_depth_<depth>/<smac_seed>/
```

Every completed run saves SMAC's standard output together with
`trajectory.json`, `runhistory.json`, `incumbent.json`, `runtime.json`, and
`policy_events.json`. The policy-event file records the fixed depth and
incumbent improvement at every surrogate-training interval.

Run the focused validation tests with:

```bash
python test_experiment.py
```

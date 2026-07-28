# One-hot instance features

This is the PCA-path counterpart to `../01_initial`. It uses exactly the same
SynthACticBench O1 setup:

- one 10-dimensional landscape (benchmark seed 40),
- the same 10 deterministic instances in every run,
- five SMAC seeds (`0` through `4`),
- 1,000 completed trials per run, and
- the unmodified random-forest surrogate provided by
  `AlgorithmConfigurationFacade`.

The only experimental difference is the feature representation. Each instance
receives a 10-dimensional one-hot vector:

```python
{
    "i0": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "i1": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    # ...
    "i9": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
}
```

SMAC 2.4.0's AC facade defaults to four PCA components. Since this experiment
has 10 instance-feature columns, it deliberately exercises the path that
reduces the instance-feature block from 10 columns to four. The 10 configuration
variables remain unchanged, so a correctly transformed RF input has 14 columns.

## Run

Submit the five jobs directly:

```bash
./submit_experiment.py
```

Preview them without submitting:

```bash
./submit_experiment.py --dry-run --list-jobs
```

Results are written under
`smac_output/default_rf_one_hot_instance_features/`. Each successful run saves
the exact one-hot mapping, default RF options, and observed PCA state in its
`trajectory.json`.

## Confirmed SMAC 2.4.0 result

An 80-trial smoke run reaches PCA training and reproduces the same
`predict_marginalized_over_instances_batch` assertion seen with the ACLib
features. The forest is trained with the four PCA components, while the
marginalized-prediction path supplies the original 10 feature columns.

The runner deliberately leaves SMAC unchanged so this remains a faithful
countercheck. If a run fails, it writes `failure.json` with the completed-trial
count, PCA state, raw feature dimension, and trained forest input dimension,
then re-raises the exception so Slurm correctly marks the task as failed.

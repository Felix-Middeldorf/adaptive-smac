# Enumerated instance features

This countercheck runs SynthACticBench O1 deterministic with:

- one 10-dimensional landscape (benchmark seed 40),
- the same 10 deterministic instances in every run,
- five SMAC seeds (`0` through `4`),
- 1,000 completed trials per run, and
- the completely default random-forest surrogate from
  `AlgorithmConfigurationFacade`.

The benchmark instances still have their usual normally distributed additive
offsets. Those offsets are **not** passed as SMAC features. Instead, each
instance is assigned its enumeration as a one-dimensional feature:

```python
{
    "i0": [0.0],
    "i1": [1.0],
    # ...
    "i9": [9.0],
}
```

No model or random-design object is constructed or overridden in the runner.
The only non-default information supplied to SMAC is the scenario, including
the instances and the feature mapping.

## Run

From this directory, submit the five jobs with:

```bash
./submit_experiment.py
```

Preview the jobs without submitting them:

```bash
./submit_experiment.py --dry-run --list-jobs
```

Results are written below `smac_output/default_rf_enumerated_instance_features/`.
Each seed directory contains SMAC's output plus `trajectory.json`,
`runhistory.json`, `incumbent.json`, and `runtime.json`. The trajectory also
records the exact instance map, enumerated feature mapping, default RF options,
and whether SMAC applied PCA.

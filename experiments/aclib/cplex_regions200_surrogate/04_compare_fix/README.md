# Batched `get_x_best` comparison

This experiment tests the batched `_get_x_best` implementation in the local
SMAC checkout at `external/SMAC3`.

The comparison uses the same ACLib `cplex_regions200` setup in both modes:

- depth 20;
- the first 150 training instances and their 148-dimensional features;
- 3,000 completed trials;
- SMAC seeds 0 and 1;
- `retrain_after=8`; and
- all other `AlgorithmConfigurationFacade` defaults.

There are four jobs:

- `original_singleton` reproduces the original SMAC 2.4.0 row-by-row
  `_get_x_best` method;
- `fixed_batched` uses the patched default implementation from
  `external/SMAC3`.

Both modes therefore use the same local SMAC checkout and dependencies. The
runner records total runtime, target-function runtime, model training,
`_get_x_best`, every marginalized-prediction call, and 500-trial checkpoints.
The analyzer also verifies whether the paired run histories are identical
after excluding timing fields.

## Validate and submit

```bash
experiments/aclib/cplex_regions200_surrogate/04_compare_fix/smoke_checks.py
experiments/aclib/cplex_regions200_surrogate/04_compare_fix/submit_experiment.py \
  --dry-run --list-jobs
experiments/aclib/cplex_regions200_surrogate/04_compare_fix/submit_experiment.py
```

The launcher puts `external/SMAC3` first in `PYTHONPATH`; the runner fails
immediately if `ConfigSelector` was imported from another installation. Each
job requests one CPU, 6 GB memory, and 48 hours on `c23ms`. Jobs do not requeue
on timeout.

For a notebook analysis of the results currently available (including exact
common-prefix comparisons while the original jobs are still running), open:

```text
analyze_current_results.ipynb
```

After all four jobs complete, the compact command-line summary can also be
regenerated with:

```bash
experiments/aclib/cplex_regions200_surrogate/04_compare_fix/analyze_comparison.py
```

Results are stored below:

```text
results/<MODE>/<SMAC_SEED>/
```

The main outputs are `runtime_summary.json`, `runtime_events.json`, and
`comparison.csv`.

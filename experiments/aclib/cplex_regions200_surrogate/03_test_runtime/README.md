# ACLib runtime root-cause experiment

This experiment profiles the unexpectedly high SMAC overhead observed in
`../02_full_run`. It uses depth 20, 150 training instances, 1,000 trials, and
SMAC seeds 0 and 1. Fourteen independent jobs compare:

- `baseline`: the full-run settings;
- `no_instance_features`: removes marginalized instance-feature prediction;
- `challengers_500`: reduces acquisition challengers from 5,000 to 500;
- `local_search_1`: reduces local-search starts from 10 to 1;
- `random_search_only`: removes acquisition local search;
- `retrain_after_64`: retrains/selects after 64 rather than 8 configurations;
- `no_periodic_save`: disables SMAC's full-state save after every trial and
  performs one final save.

Each run records inclusive timings for model training, marginalized prediction,
random and local acquisition search, data collection, `ask`, `tell`, and SMAC
state serialization. Results are written to:

```text
results/<VARIANT>/<SMAC_SEED>/
```

The important files are `runtime_summary.json` and `runtime_events.json`.
Timings of nested phases are inclusive and must not be summed.

## Validate and submit

```bash
experiments/aclib/cplex_regions200_surrogate/03_test_runtime/smoke_checks.py
experiments/aclib/cplex_regions200_surrogate/03_test_runtime/submit_experiment.py --dry-run --list-jobs
experiments/aclib/cplex_regions200_surrogate/03_test_runtime/submit_experiment.py
```

The jobs request one CPU, 6 GB memory, and eight hours on `c23ms`. They do not
requeue on timeout.

After completion:

```bash
experiments/aclib/cplex_regions200_surrogate/03_test_runtime/analyze_runtime.py
```

For a full Python call profile of one run, invoke `profile_runtime.py` directly
with `--cprofile`; it writes `runtime.prof`, which can be inspected with:

```bash
python -m pstats results/baseline/0/runtime.prof
```

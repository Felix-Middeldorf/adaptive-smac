# ACLib `cplex_regions200` surrogate with modern SMAC

This directory runs the installed ACLib `cplex_regions200_surrogate` model with
SMAC 2.4's `AlgorithmConfigurationFacade`. No CPLEX installation or new Python
package download is required: target evaluations are predictions from the
pretrained 778 MB PyRFR model.

## Runtime

The repository already has the required packages split across two compatible
Python 3.9 environments. Always use `run_in_env.sh`; it selects the ACLib Python
interpreter and adds the existing SMAC 2.4 packages to `PYTHONPATH`.

```bash
cd experiments/aclib/cplex_regions200_surrogate
./run_in_env.sh smoke_checks.py
./run_in_env.sh smoke_checks.py --model
```

The legacy EPM code was patched for NumPy 2 (`np.float`/`np.int` were removed)
and its child processes now use the active interpreter instead of a bare
`python` command. `external/aclib-surrogates/epm` and `aclib2` are independent
git checkouts, so those compatibility edits appear in their own `git status`.
The modern runner does not depend on the removed NumPy aliases and remains
usable if those nested checkouts are later refreshed; the patches additionally
repair the original legacy wrapper/server path.

## Benchmark semantics

- 74 CPLEX hyperparameters and four conditions are loaded from the original PCS.
- The original 1,000 training instances are used during SMAC optimization.
- The disjoint 1,000 test instances are reserved for optional final validation.
- All original 148 normalized instance features are passed to SMAC as well as
  to the target surrogate.
- The historical quantile forest is evaluated with seed `0`, which selects its
  deterministic median prediction.
- The cutoff is 10,000 seconds. A surrogate `CUTOFF` is converted to the
  original PAR10 cost of 100,000; ordinary predictions are returned unchanged.
- The modern Scenario does not expose a runtime cutoff to its intensifier. The
  adapter applies the original fixed cutoff itself; adaptive capping would not
  make model predictions cheaper and would introduce censored targets into the
  modern scalar target interface.
- The pipeline uses `AlgorithmConfigurationFacade` defaults. The original
  algorithm default is explicitly included in the initial design.

Each completed SMAC trial is one configuration-instance evaluation. A budget of
5,000 trials therefore does **not** mean 5,000 complete sweeps over all 1,000
instances.

## Local run

This short command is useful before submitting a large experiment:

```bash
./run_in_env.sh run_smac.py \
  --smac-seed 0 \
  --n-trials 100 \
  --n-instances 20 \
  --output-root /tmp/cplex_regions200_smoke
```

For a full run using all training instances and optional held-out validation:

```bash
./run_in_env.sh run_smac.py \
  --smac-seed 0 \
  --n-trials 5000 \
  --n-instances 1000 \
  --validate-test
```

## Submitit jobs

The defaults are ten SMAC seeds, 5,000 trials, all 1,000 training instances,
one SMAC run per job, 6 GB per job, and a 12-hour limit. The script enforces the
80-job cluster limit.

Inspect the exact matrix without submitting:

```bash
./run_in_env.sh submit_experiment.py --dry-run --list-jobs
```

Submit it:

```bash
./run_in_env.sh submit_experiment.py
```

All important dimensions are configurable, for example:

```bash
./run_in_env.sh submit_experiment.py \
  --smac-seeds 0 1 2 3 4 \
  --n-trials 10000 \
  --n-instances 1000 \
  --validate-test
```

## Outputs and continuation

Runs are stored below:

```text
results/cplex_regions200_train<N_INSTANCES>_trials<N_TRIALS>/<SMAC_SEED>/
```

Alongside SMAC's `scenario.json`, `runhistory.json`, `intensifier.json`, and
`optimization.json`, the runner saves:

- `run_metadata.json`: exact setup, versions, assets, and instance list
- `incumbent.json`: final incumbent and its aggregate training cost
- `trajectory.json`: incumbent changes over completed trials
- `test_validation.json`: per-instance held-out results when requested
- `summary.json`: concise final result
- `completed.json`: written last and used as the completion marker

A matching complete run is skipped. If a job is interrupted after SMAC has
saved state, rerunning it uses `overwrite=False`, so SMAC continues from its
saved runhistory and intensifier state. The manual `--overwrite` option on
`run_smac.py` intentionally starts that exact run from scratch.

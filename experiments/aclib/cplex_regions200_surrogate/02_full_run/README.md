# Full fixed-depth ACLib run

This experiment runs the ACLib `cplex_regions200` surrogate with:

- the first 150 instances from ACLib's predefined training split,
- 10,000 completed SMAC trials per run,
- SMAC seeds `0` through `4`, and
- fixed SMAC random-forest depths `5`, `10`, `15`, `20`, `25`, `30`, and `40`.

This produces `7 × 5 = 35` independent SMAC runs. The Submitit launcher uses
35 tasks, with one run per task, so all runs can execute concurrently while
remaining below the 80-job limit.

SMAC's instance-feature PCA is disabled (`pca_components=None`). This is the
same working setup established in `../01_initial_c` and avoids the confirmed
SMAC 2.4.0 PCA/marginalized-prediction dimension bug. Apart from fixed depth
and disabled PCA, `AlgorithmConfigurationFacade` uses its defaults.

## Validate and submit

The files are directly executable from any working directory:

```bash
experiments/aclib/cplex_regions200_surrogate/02_full_run/smoke_checks.py
experiments/aclib/cplex_regions200_surrogate/02_full_run/submit_experiment.py --dry-run --list-jobs
experiments/aclib/cplex_regions200_surrogate/02_full_run/submit_experiment.py
```

Each task requests one CPU, 6 GB memory, and 48 hours on `c23ms`. Completed
runs are skipped on resubmission. An incomplete directory is restarted rather
than resumed, preventing SMAC from opening an interactive scenario-overwrite
prompt inside a Slurm task.

Results are written to:

```text
results/depth_<DEPTH>/<SMAC_SEED>/
```

Each completed directory contains SMAC's runhistory and state together with
`run_metadata.json`, `incumbent.json`, `trajectory.json`, `summary.json`, and a
final `completed.json` marker.

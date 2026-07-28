# Fixed SMAC-surrogate depths on ACLib `cplex_regions200`

This experiment compares SMAC random-forest depths `5`, `10`, `20`, and `30`.
Every depth uses the same first 100 instances from ACLib's predefined training
list and is run for 1,000 completed SMAC trials with SMAC seeds `0` through `4`.
That gives 20 independent SMAC runs and 20 Submitit jobs.

All `AlgorithmConfigurationFacade` settings other than `max_depth` remain at
their defaults. The pretrained ACLib target surrogate is identical across runs.

```bash
cd experiments/aclib/cplex_regions200_surrogate/01_initial_c
./smoke_checks.py
./submit_experiment.py --dry-run --list-jobs
./submit_experiment.py
```

The scripts resolve the shared ACLib adapter relative to their own location, so
they work from any current directory without setting `PYTHONPATH`. Their shebang
uses the `aclib2-surrogates-py39` environment directly.

Results are written to `results/depth_<DEPTH>/<SMAC_SEED>/`. Matching completed
runs are skipped, interrupted matching runs are resumed by SMAC, and a run lock
prevents duplicate jobs from writing the same directory concurrently.

Open `analyze_fixed_depths.ipynb` after the jobs finish for confidence-interval
trajectories, checkpoint boxplots, and a final ranking table.

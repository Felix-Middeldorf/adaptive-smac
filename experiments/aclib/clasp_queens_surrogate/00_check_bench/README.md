# Clasp Queens surrogate validation

This experiment checks the downloaded ACLib Clasp Queens surrogate against the
original real-Clasp measurements published with the EPM.

It deliberately separates two questions:

1. Does the in-process adapter used by the SMAC experiments reproduce the
   official ACLib wrapper's numerical request path?
2. How closely does the EPM reproduce the archived real Clasp measurements?

The validation includes:

- scenario, ConfigSpace, instance-list, and feature-preprocessing checks;
- exact parity between ACLib's official request conversion/prediction path and
  `ACLibSurrogateBenchmark`;
- an end-to-end five-point check through the legacy command-line wrapper,
  communicator, and local HTTP EPM server;
- a stratified point-level comparison against real runs;
- all archived configurations observed on all 484 training instances;
- uncertainty calibration using repeatedly evaluated configuration-instance
  pairs.

The legacy multi-row status-label issue is intentionally outside the scope of
this experiment.

## Inputs

- Performance data:
  `external/aclib-performance-data/clasp_queens`
- Installed surrogate:
  `external/aclib-surrogates/aclib2/target_algorithms/surrogate/clasp_queens`

## Run

```bash
cd /home/io632776/experiments/adaptive-smac/experiments/aclib/clasp_queens_surrogate/00_check_bench
./run_validation.py
```

For a quick development check:

```bash
./run_validation.py --smoke --overwrite
```

The full defaults use 10,000 stratified archived runs, 100 wrapper-parity
points, every fully covered configuration, and 500 repeated pairs with 100 EPM
quantile draws each. The first command-line request takes about ten seconds
because the legacy communicator polls server startup in ten-second intervals.
Use `--skip-cli-wrapper` only in environments that prohibit local sockets.

Open `analyze_validation.ipynb` after `results/summary.json` has been written.
All generated tables are stored under `results/` and are ignored by Git.

The focused five-seed distribution comparison is generated with:

```bash
./run_five_seed_comparison.py
```

It uses all 737 configuration-instance pairs having exactly five distinct real
Clasp seeds and draws 1,000 surrogate outcomes for each identical pair. It
stores both raw distributions as NumPy arrays, plus pair-level quantiles,
timeout mass, interval coverage, KS statistics, and Wasserstein distances.

## Interpretation

- Wrapper parity failure means the local call path is wrong.
- Wrapper parity success but poor agreement with archived measurements points
  to limitations of the packaged EPM or its training.
- Good archived-data agreement combined with strange new random samples points
  to out-of-distribution extrapolation.

Because this archive supplied data for EPM construction, the statistical
comparison is an integrity/reproduction check, not an independent test of
generalization.

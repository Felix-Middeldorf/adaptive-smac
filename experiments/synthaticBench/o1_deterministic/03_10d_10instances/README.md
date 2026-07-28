# O1 deterministic fixed depths: 10D / 10 instances / 5,000 trials

This experiment evaluates fixed random-forest depths
`5, 10, 15, 20, 30, 40, 50` on benchmark seed `40`, using SMAC seeds `0..4`.
Every run has 10 dimensions, the same deterministic map of 10 instances, and
5,000 completed SMAC trials.

The setup follows the existing fixed-depth experiments:

- SynthACticBench O1 deterministic
- `AlgorithmConfigurationFacade`
- random-design probability `0.0`
- `min_samples_leaf=1`
- `min_samples_split=1`
- all remaining ACFacade settings left at their defaults
- `PYTHONHASHSEED=12345`

The complete matrix contains 35 unique runs:

```text
7 depths × 1 benchmark seed × 5 SMAC seeds = 35 runs
```

The Submitit launcher assigns one complete SMAC run to every Slurm task. Each
task requests one CPU, 4 GB RAM, and three hours on partition `c23ms`. Requeue
support is enabled. Complete outputs are validated and skipped on rerun; an
incomplete run restarts with `overwrite=True`.

Validate without submitting:

```bash
python submit_fixed_depths.py --dry-run
python submit_fixed_depths.py --list-jobs
```

Submit all 35 tasks:

```bash
python submit_fixed_depths.py
```

Each run writes to:

```text
smac_output/benchmark_seed_40/fixed_depth_<depth>/<smac_seed>/
```

Completed outputs include `trajectory.json`, `runhistory.json`,
`incumbent.json`, `runtime.json`, and SMAC's standard metadata files.

Run the focused validation tests with:

```bash
python test_experiment.py
```

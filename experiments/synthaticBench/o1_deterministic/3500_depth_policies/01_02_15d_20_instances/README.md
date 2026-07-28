# Follow-up O1 deterministic experiment: 15D / 20 instances

This follow-up uses exactly the same benchmark seeds (`40, 42, 44, 46`), SMAC
seeds (`0..5`), deterministic 20-instance map, 15 dimensions, and 3,500-trial
budget as `../01_15d_20_instances`.

It runs only eight new policies. The existing `fixed_depth_20` and `g` results
are reused directly by the analytics notebook; they are not copied or
resubmitted.

Every run uses `AlgorithmConfigurationFacade`, random-design probability `0.0`,
`min_samples_leaf=1`, and `min_samples_split=1`. Depth changes occur immediately
before the first real surrogate retraining after a completed-trial boundary.

## Policies

Depth scores use only events from the current evaluation window:

```text
score = sum(positive incumbent improvements) / configurations suggested
```

Best-score ties prefer the shallower depth. Rotation starts reproducibly at a
random member and then advances through the sorted depths with wraparound.

### New fixed policies

- `fixed_depth_18`: use depth 18 for 0–100%.
- `fixed_depth_23`: use depth 23 for 0–100%.
- `fixed_depth_26`: use depth 26 for 0–100%.
- `fixed_depth_29`: use depth 29 for 0–100%.

### `g_freeze_on_zero`

- 0–20%: depth 20 (ACFacade default).
- 20–40%: rotate `{10, 14, 18, 22, 25}`.
- 40–60%: use the rounded-up midpoint of the best two depths.
- 60–75%: rotate `{currD-4, currD, currD+4}`.
- 75–100%: use the rounded-up midpoint of the best two local depths.
- If either evaluation window has all-zero scores, stop that rotation and
  freeze at the fixed depth used before the window (depth 20 for the first
  window, or `currD` for the second).

### `g_high_band`

- 0–20%: depth 20.
- 20–40%: rotate the higher band `{20, 23, 26, 29}`.
- 40–60%: use the rounded-up midpoint of the best two depths.
- 60–75%: rotate `{currD-3, currD, currD+3}`.
- 75–100%: use the rounded-up midpoint of the best two local depths.
- As in the original `g`, an all-zero window retains its current rotation.

### `b2_long_refinement`

- 0–5%: depth 20.
- 5–15%: rotate `{5, 10, 15, 20}`.
- 15–25%: rotate the best two plus their rounded-up midpoint.
- 25–40%: use the best depth from the preceding window.
- 40–55%: rotate `{currD-3, currD, currD+3, currD+6}`.
- 55–70%: rotate the best two plus their rounded-up midpoint.
- 70–100%: use the best depth from the preceding window.
- All-zero windows retain the current set, matching `b2`; the two refinement
  evaluation windows are lengthened from 10% to 15% each.

### `early_then_high`

- 0–20%: depth 20.
- 20–40%: rotate `{10, 14, 18, 22, 25}`.
- 40–60%: use the rounded-up midpoint of the best two depths.
- 60–100%: force depth 20, which was the stable late winner in the preceding
  experiment.
- If the 20–40% scores are all zero, use depth 20 for 40–60% as well.

## Run and job counts

```text
8 new policies × 4 benchmark seeds × 6 SMAC seeds = 192 new SMAC runs
192 runs / 3 runs per task = 64 Submitit/Slurm tasks
```

This is below the 80-concurrent-job limit. Every task requests one CPU, 4 GB of
memory, and four hours. Runs are placed in distinct policy/benchmark/SMAC-seed
directories. Completed trajectories are validated and skipped after reruns or
requeues; an incomplete run restarts cleanly.

Validate without submitting:

```bash
python submit_experiment.py --dry-run
python submit_experiment.py --list-packs
python test_experiment.py
```

Submit the 64 tasks:

```bash
python submit_experiment.py
```

### Extended high fixed depths

`submit_extended_fixed_depths.py` runs only the additional fixed depths
`32, 35, 38, 41, 45, 50` using the same four benchmark seeds and six SMAC
seeds:

```text
6 depths × 4 benchmark seeds × 6 SMAC seeds = 144 SMAC runs
144 runs / 2 runs per task = 72 Submitit/Slurm tasks
```

It does not resubmit any completed fixed or adaptive policy. Validate and submit
the isolated extension with:

```bash
python submit_extended_fixed_depths.py --dry-run
python submit_extended_fixed_depths.py --list-packs
python submit_extended_fixed_depths.py
```

Each extension task requests one CPU, 4 GB of memory, and four hours. Completed
extended runs are validated and skipped on rerun or requeue.

## Outputs and analytics

New results are written beneath:

```text
smac_output/benchmark_seed_<benchmark>/<policy>/<smac_seed>/
```

Each completed run saves its runhistory, trajectory, incumbent, runtime, all
surrogate-training depths, evaluation scores, selections, and policy
transitions.

`analyze_experiment.ipynb` combines the eight new policies with the existing
`fixed_depth_20` and `g` trajectories from `../01_15d_20_instances`. It provides
the same main analyses as the original notebook: per-benchmark confidence
intervals, 500-trial boxplots, within- and across-benchmark rankings,
leave-one-landscape-out selection, early-budget rankings, and adaptive-depth
transition diagnostics. It can be opened before completion to inspect coverage;
rerun all cells after the jobs finish for the complete comparison.

`analyze_all_fixed_depths.ipynb` focuses on all fixed controls from both
experiments: depths `5, 10, 15, 18, 20, 23, 26, 29, 32, 35, 38, 41, 45, 50`. It includes per-landscape
95% confidence-interval trajectories, 500-trial boxplots, and 200-trial
heatmaps identifying both the lowest mean incumbent regret and the largest mean
incumbent improvement in each window.

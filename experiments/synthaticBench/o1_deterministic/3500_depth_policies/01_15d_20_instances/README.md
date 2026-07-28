# O1 deterministic: 15D, 20 instances, 3,500 trials

This directory contains the fixed and adaptive random-forest-depth experiment
for SynthACticBench O1 deterministic. It uses benchmark seeds `40, 42, 44, 46`
from the landscapes evaluated in `../../depth_policies/09_02_new policies`, SMAC
seeds `0..5`, 15 dimensions, 20 instances, and 3,500 completed target-function
trials per run.

The implementation uses `AlgorithmConfigurationFacade` defaults except for:

- random-design probability `0.0`
- random-forest `min_samples_leaf=1`
- random-forest `min_samples_split=1`

The ACFacade default configuration-selector cadence is retained
(`retrain_after=8`). Depth control wraps the model's actual `train()` call, so a
budget transition is first applied immediately before the first real surrogate
retraining after its completed-trial boundary. A configuration-selector callback
counts configurations suggested after each training event.

## Policy count

The requested names contain **15 policies**, not 13: four fixed policies plus
`a1`, `a2`, `b1`, `b2`, `c1`, `c2`, `c3`, `e1`, `e2`, `e3`, and `g`. Therefore
the explicit policy list produces `15 × 4 × 6 = 360` SMAC runs. The launcher
packs five runs into each task, yielding 72 Slurm array tasks, below the limit of
80. No named policy is silently omitted.

## Policy overview

For every rotating window, the first depth is selected randomly and
reproducibly. Each subsequent surrogate retraining advances to the next higher
depth, wrapping back to the lowest depth when necessary.

The ACFacade default maximum depth used by these policies is 20. Moving-boundary
policies clip downward expansion at depth 1.

Depths are scored only with surrogate-training events from the current
evaluation window:

```text
score = sum(positive incumbent improvements) / configurations suggested
```

Best-score ties prefer the shallower depth, while worst-score ties remove the
deeper depth. If every score is zero, the current depth set is retained.

### Fixed-depth controls

#### `fixed_depth_5`

- 0–100%: always use depth 5.

#### `fixed_depth_10`

- 0–100%: always use depth 10.

#### `fixed_depth_15`

- 0–100%: always use depth 15.

#### `fixed_depth_20`

- 0–100%: always use depth 20.

### Staged adaptive policies

#### `a1`

- 0–5%: use the ACFacade default depth.
- 5–15%: rotate `{5, 10, 15, 20}`.
- 15–25%: rotate the best two depths from the preceding window.
- 25–100%: use the best depth from the 15–25% window.

#### `a2`

- 0–5%: use the ACFacade default depth.
- 5–15%: rotate `{5, 10, 15, 20}`.
- 15–25%: rotate the best two depths from the preceding window.
- 25–50%: use the best depth from the 15–25% window; call it `currD`.
- 50–60%: rotate `{currD-3, currD, currD+3, currD+6}`.
- 60–70%: rotate the best two depths from the preceding window.
- 70–100%: use the best depth from the 60–70% window.

#### `b1`

- 0–5%: use the ACFacade default depth.
- 5–15%: rotate `{5, 10, 15, 20}`.
- 15–25%: select the best two depths, add their rounded-up midpoint, and
  rotate the resulting three-depth set.
- 25–100%: use the best depth from the 15–25% window.

#### `b2`

- 0–5%: use the ACFacade default depth.
- 5–15%: rotate `{5, 10, 15, 20}`.
- 15–25%: select the best two depths, add their rounded-up midpoint, and
  rotate the resulting set.
- 25–50%: use the best depth from the 15–25% window; call it `currD`.
- 50–60%: rotate `{currD-3, currD, currD+3, currD+6}`.
- 60–70%: select the best two depths, add their rounded-up midpoint, and
  rotate the resulting set.
- 70–100%: use the best depth from the 60–70% window.

### Moving-boundary policies

These policies evaluate their current set in consecutive 5%-budget windows.
After every window, scores are reset and rotation restarts from a new random
position in the retained or updated set.

#### `c1`

- Start with `{4, 6, 8, 10}` and step size 2.
- If the worst depth is below the set median, remove it and add
  `maximum depth + 2`.
- If the worst depth is above the median, remove it and add
  `minimum depth - 2`.
- If the median is worst, retain the complete set.

#### `c2`

- Use the same update rule as `c1`.
- Start with `{5, 7, 9}` and use step size 2.

#### `c3`

- Use the same update rule as `c1`.
- Start with `{4, 7, 10}` and use step size 3.

#### `e1`

- Start with `{4, 6, 8, 10}` and step size 2.
- Expand upward only when the lowest depth is worst or the highest depth is
  best.
- Expand downward only when the highest depth is worst or the lowest depth is
  best.
- Otherwise retain the complete set.

#### `e2`

- Use the same strict update rule as `e1`.
- Start with `{5, 7, 9}` and use step size 2.

#### `e3`

- Use the same strict update rule as `e1`.
- Start with `{4, 7, 10}` and use step size 3.

### Coarse-to-local policy

#### `g`

- 0–20%: use the ACFacade default depth.
- 20–40%: rotate `{10, 14, 18, 22, 25}`.
- 40–60%: use the rounded-up midpoint of the best two depths.
- 60–75%: call the current depth `currD` and rotate
  `{currD-4, currD, currD+4}`.
- 75–100%: use the rounded-up midpoint of the best two depths from the
  60–75% window.

## Submit

Validate the full matrix without submitting:

```bash
python submit_experiment.py --dry-run
python submit_experiment.py --list-packs
```

Submit all 72 packed tasks:

```bash
python submit_experiment.py
```

Each task runs five policies for one benchmark-seed/SMAC-seed pair. It requests
one CPU, 4 GB RAM, and 12 hours. Completed trajectories are validated and
skipped on rerun or requeue; an incomplete current run restarts with
`overwrite=True`.

## Outputs

Every run has its own directory:

```text
smac_output/benchmark_seed_<benchmark>/<policy>/<smac_seed>/
```

Alongside SMAC's standard scenario and optimization files, each completed run
contains:

- `runhistory.json`
- `trajectory.json`
- `incumbent.json`
- `runtime.json`
- `policy_events.json`

The trajectory and policy-event files contain the depth at every surrogate
training, incumbent costs on both sides of each event, improvements, numbers of
suggested configurations, per-window scores, rankings, selected depths, depth
changes, and all policy transitions. Rotation starts are random but reproducible
for each policy/benchmark-seed/SMAC-seed combination.

Run the focused validation suite with:

```bash
python test_experiment.py
```

## Analytics

`analyze_experiment.ipynb` contains per-benchmark confidence-interval
trajectories, 500-trial checkpoint boxplots, within- and across-benchmark policy
rankings, leave-one-benchmark-out policy selection, rank-stability and anytime
diagnostics, and adaptive terminal-depth summaries.

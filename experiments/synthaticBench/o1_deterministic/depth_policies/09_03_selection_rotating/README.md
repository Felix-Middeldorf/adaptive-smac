# O1 rotating depth-selection experiment

This experiment uses the O1 big-experiment setup: benchmark seeds 40--46,
SMAC seeds 0--9, 10 dimensions, 10 instances, 1,000 trials, and no random
design injection.

## Policy

- Before trial 250, every surrogate retraining rotates through depths
  `5, 10, 15, 20`.
- From trial 250 through 299, the last exploration depth is held. This gap is
  not included in the selection score.
- At trial 300, each depth is ranked by the sum of its positive reductions in
  the best average configuration cost over its exploration segments. Exact
  ties prefer the smaller depth.
- From trial 300 through 499, surrogate retraining alternates between the two
  highest-ranked depths and scores their segments in the same way.
- At trial 500, the better of those two depths is selected and held through
  trial 1,000.

The configuration selector retrains after every newly proposed configuration,
so each actual surrogate retraining advances the rotation. Intensification can
assign a variable number of instance evaluations to a configuration; callback
accounting therefore cuts the score windows exactly at trials 250, 300, and
500 even when a trained-depth segment crosses a boundary. The trajectory
records every segment, score, ranking, and selection.

## Slurm layout

There are exactly 10 jobs, one per SMAC seed. Each job runs all seven benchmark
seeds sequentially, for 70 independent SMAC runs in total. Each job requests
one CPU, 4 GB, and 12 hours.

Validate without submitting:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/09_03_selection_rotating/submit_selection_rotating.py \
  --dry-run
```

Submit:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/depth_policies/09_03_selection_rotating/submit_selection_rotating.py
```

## Analysis

`analyze_selection_rotating.ipynb` validates all 70 original adaptive
trajectories, all 70 rotating-delay/forced-depth-20 trajectories, and the 280
matched fixed-depth controls. It summarizes both selection stages and their
improvement evidence, plots example depth schedules, compares both adaptive
policies against fixed depths 5, 10, 15, and 20 from `09_big_experiment`, and
ranks all six policies across benchmark landscapes.

## Rotating-delay / forced-depth-20 variant

`o1_selection_rotating_force20_runner.py` implements a separate variant without
changing or overwriting the original policy:

- Trials 0–249: rotate `5, 10, 15, 20` and score positive incumbent
  improvements exactly as in the original policy.
- Trials 250–299: continue the same rotation, but do not include this gap in
  either selection score.
- At trial 300: select the best two depths from the scored 0–250 phase.
- Trials 300–499: alternate and score the selected pair.
- At trial 500: retain the second-stage ranking for diagnostics, but ignore its
  winner and force depth 20 through trial 1,000.

It uses a separate policy/output directory named
`selection_rotating_delay_rotation_force20`, so existing adaptive runs remain
untouched. The Slurm layout remains ten jobs, one per SMAC seed, with seven
benchmark landscapes executed sequentially in each job (70 runs total).

Validate and submit with:

```bash
python test_selection_rotating_force20.py
python submit_selection_rotating_force20.py --dry-run
python submit_selection_rotating_force20.py
```

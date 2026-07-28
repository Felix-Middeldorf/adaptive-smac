# O6 Multi-Modal big depth-policy experiment

This study runs SynthACticBench O6 Multi-Modal with:

- benchmark seeds 40–46 (7 landscapes),
- SMAC seeds 0–9,
- 10 dimensions and 10 instances,
- 1,500 trials per SMAC run,
- SMAC `AlgorithmConfigurationFacade` with 0% random-design injection,
- SMAC's default RF hyperparameters except for the controlled maximum depth.

## Policies

Each `(benchmark seed, SMAC seed)` job executes the same 109 policies:

| Family | Count | Definition |
|---|---:|---|
| Fixed | 17 | Every integer depth from 4 through 20 |
| Sampled three-stage | 40 | Switch after 250 and 660 completed trials; depths from `{5,10,15,20}` |
| Increase by 1 | 10 | Add 1 every 100 trials, starting at 4–13 |
| Increase by 2 | 10 | Add 2 every 100 trials, starting at 4–13 |
| Increase by 2, then hold | 10 | Add 2 every 100 trials through trial 500, then hold |
| Increase by 3 | 10 | Add 3 every 100 trials, starting at 4–13 |
| Alternating | 8 | Every 100 trials alternate `+4, -2`, starting at 5–12 |
| Cyclic | 4 | Every 50 trials cycle through one of the four requested depth sequences |

All increasing schedules are capped at depth 20. Alternating schedules cap every
`+4` step at 20 and still perform the following `-2` step. Cyclic schedules wrap
back to their first depth after reaching the end of their sequence.

The 40 three-stage policies are frozen in
`sampled_three_stage_policies.json`. The four constant schedules were excluded
because fixed depths 5, 10, 15, and 20 already provide identical controls. The
sample is balanced: every candidate depth appears exactly 10 times in each
stage, and all non-constant transition-shape classes are represented.

## Combined overnight Slurm layout

Use `submit_all.py` to run the entire study, including the consensus-ramp
policy, in one job for each benchmark/SMAC-seed pair:

```text
7 benchmark seeds × 10 SMAC seeds = 70 Slurm jobs
70 jobs × 110 policies = 7,700 SMAC runs
7,700 runs × 1,500 trials = 11,550,000 SMAC trials
```

The array parallelism is 70, below the limit of 80 simultaneous jobs. Each job
has a 12-hour limit, requests one CPU and 4 GB of memory, and runs its 110
policies sequentially. Every O6 trajectory is metadata-validated and skipped
when already complete, so requeued or resubmitted jobs safely continue after
the last completed policy.

Validate without submitting:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o6_multi_modal/depth_policies/09_big_experiment/submit_all.py \
  --dry-run
```

Submit the complete experiment as 70 jobs:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o6_multi_modal/depth_policies/09_big_experiment/submit_all.py
```

`submit_big_experiment.py` and `submit_consensus_ramp.py` remain available for
rerunning either component separately. Do not launch both separate scripts when
you want to stay within the 80-job limit; together they submit 140 jobs.

## Analysis

`analyze_big_experiment.ipynb` mirrors the O1 big-experiment analysis for the
1,500-trial O6 study. It validates trajectory coverage, ranks policies within
each landscape, plots confidence intervals and checkpoint distributions,
analyzes the sampled three-stage depths, derives individual-landscape and
leave-one-landscape-out depth schedules, and reports cross-landscape rank
correlations. The consensus-ramp policy is included automatically when all 70
of its trajectories are present.

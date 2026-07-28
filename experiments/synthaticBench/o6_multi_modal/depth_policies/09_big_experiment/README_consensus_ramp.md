# O6 Multi-Modal consensus-ramp validation

This experiment evaluates on O6 the same deliberately conservative policy that
was derived from the seven-landscape O1 `09_big_experiment` results:

| Completed trials | 0 | 100 | 200 | 300 | 400 | 500+ |
|---:|---:|---:|---:|---:|---:|---:|
| Random-forest maximum depth | 9 | 11 | 13 | 16 | 19 | 20 |

The policy combines the aggregate observations that depth 9 was strongest near
trial 200, smoothly increasing policies were strongest in the middle of the
run, and depth 20 was strongly preferred late. It is not an exact replay of any
policy evaluated in `09_big_experiment`.

The benchmark configuration matches the O6 `09_big_experiment`: O6 Multi-Modal,
benchmark seeds 40-46, SMAC seeds 0-9, 10 dimensions, 10 instances, 1,500
trials, and 0% random-design injection.

## Slurm layout

The recommended `submit_all.py` launcher includes this policy after the 109
main policies inside each of the same 70 jobs. Therefore the complete study
uses only 70 jobs. The standalone layout below is only for rerunning the
consensus policy by itself.

Each `(benchmark seed, SMAC seed)` pair is an independent Slurm job containing
exactly one SMAC run:

```text
7 benchmark seeds x 10 SMAC seeds = 70 Slurm jobs
70 jobs x 1 policy = 70 SMAC runs
70 runs x 1,500 trials = 105,000 SMAC trials
```

All 70 jobs are submitted together with array parallelism 70. Completed and
metadata-valid trajectories are skipped on resubmission.

Validate without submitting:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o6_multi_modal/depth_policies/09_big_experiment/submit_consensus_ramp.py \
  --dry-run
```

Submit the batch:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o6_multi_modal/depth_policies/09_big_experiment/submit_consensus_ramp.py
```

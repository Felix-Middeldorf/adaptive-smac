# Leave-one-landscape-out depth policies

This experiment turns the fixed-depth improvement analysis in
`../09_big_experiment/analyze_big_experiment.ipynb` into seven deployable depth
schedules. For each held-out landscape, the depth used in each 100-trial block
was selected only from the other six landscapes.

Each learned policy is evaluated on all seven landscapes. The launcher creates
49 Slurm jobs (7 learned policies x 7 evaluation landscapes), with the 10 SMAC
seeds run sequentially inside each job, for 490 SMAC runs in total. Completed
trajectories are detected and skipped when the launcher is run again.

```bash
python submit_loo_depth_policies.py --dry-run
python submit_loo_depth_policies.py
```

The analysis notebook compares each learned policy against every complete
baseline policy in `../09_big_experiment/smac_output`. Training ranks pool the
60 final-regret observations from the six training landscapes into one score
per policy; they are not averages of six separately computed ranks. Held-out
ranks use the 10 SMAC-seed observations on the excluded landscape.

## In-sample landscape-specific policies

`submit_individual_landscape_depth_policies.py` launches a separate diagnostic
experiment based on the per-landscape block analysis. Each landscape receives
its own ten-block schedule and is evaluated only on that same landscape. This
creates exactly 70 Slurm jobs, one for every landscape x SMAC-seed pair.

```bash
python submit_individual_landscape_depth_policies.py --dry-run
python submit_individual_landscape_depth_policies.py
```

These results are intentionally in-sample and should be interpreted as an
optimistic diagnostic, not as leave-one-landscape-out generalization evidence.

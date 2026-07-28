# O1 deterministic uncertainty experiment: 15D / 15 instances

This experiment is the 15-dimensional, 15-instance counterpart of
`../../depth_policies/11_uncertainty`.

It evaluates fixed depths `5, 10, 15, 20, 25, 30` on benchmark seeds `40` and
`42`, using SMAC seeds `0..9` and 3,500 completed trials per run:

```text
6 depths × 2 benchmark seeds × 10 SMAC seeds = 120 SMAC runs
```

The setup uses `AlgorithmConfigurationFacade` defaults except for random-design
probability `0.0`, `min_samples_leaf=1`, and `min_samples_split=1`. The same
deterministic 15-instance map is reused for every run.

The tested proposal-diagnostics implementation is shared directly with
`11_uncertainty`. For every configuration-selector proposal it records the
configuration, proposal origin, completed-trial count, runhistory configuration
ID, surrogate training-data count, fixed depth, marginalized predicted cost,
predictive variance and standard deviation, Expected Improvement, `eta`, and
`xi`. Initial-design and fallback-random proposals are explicitly identified.

Each run saves:

- `trajectory.json`
- `runhistory.json`
- `incumbent.json`
- `runtime.json`
- `proposal_diagnostics.json`
- `proposal_diagnostics.jsonl`
- SMAC's standard metadata files

## Submit

Because 120 one-run jobs would exceed the 80-job cluster limit, the launcher
packs two sequential SMAC runs into each task. This produces 60 Slurm tasks.
Each task requests one CPU, 4 GB RAM, and four hours on partition `c23ms`.

Validate without submitting:

```bash
python submit_uncertainty.py --dry-run
python submit_uncertainty.py --list-packs
```

Submit all 60 tasks:

```bash
python submit_uncertainty.py
```

Completed runs are validated and skipped on rerun or requeue. Incomplete runs
restart from scratch.

## Analytics

`analyze_uncertainty.ipynb` provides:

- raw five-panel diagnostics for SMAC seed 0 on both landscapes;
- mean trajectories and 95% confidence intervals over all available SMAC seeds
  for best regret, EI, predicted uncertainty, absolute error, and standardized
  error;
- seed-level boxplots for final regret and the four proposal diagnostics.

No running averages are used. Figures are saved under `plots/`.

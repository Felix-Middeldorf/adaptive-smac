# O1 deterministic surrogate-uncertainty experiment

This experiment evaluates fixed random-forest depths `5, 10, 15, 20, 25, 30`
on benchmark seeds `40` and `42`, using SMAC seeds `0..2`. Each run uses 12
dimensions, the same deterministic map of 12 instances, and 2,000 completed
SMAC trials.

The complete matrix contains 36 runs:

```text
6 depths × 2 benchmark seeds × 3 SMAC seeds = 36 runs
```

The setup uses `AlgorithmConfigurationFacade` defaults except for:

- random-design probability `0.0`
- `min_samples_leaf=1`
- `min_samples_split=1`

## Proposal-time uncertainty logging

`ProposalDiagnosticsCallback` uses SMAC's
`on_next_configurations_end` callback. This hook is called immediately after a
challenger is selected and before it is yielded for evaluation. At that moment,
the surrogate and Expected Improvement acquisition function are still in the
exact state used to choose the configuration.

For every configuration-selector proposal, the callback records:

- proposal index and number of completed trials before the proposal
- complete configuration and numerical configuration array
- SMAC proposal origin
- whether it was selected by the acquisition-function maximizer
- runhistory configuration ID
- number of observations used for the current surrogate training
- fixed model depth
- acquisition-function name, incumbent threshold (`eta`), and `xi`
- marginalized surrogate-predicted cost
- marginalized predictive variance and standard deviation
- Expected Improvement acquisition value

The predicted mean and variance are obtained with
`model.predict_marginalized()`, exactly as SMAC's default Expected Improvement
implementation does. The acquisition value is recomputed immediately with the
unchanged model and acquisition state.

Initial-design and fallback-random configurations are retained in the file with
`selected_by_acquisition=false` and null prediction fields. This makes the
exclusions explicit. A configuration proposal is not the same as a completed
SMAC trial: SMAC's intensifier can schedule multiple instance evaluations of
one proposed configuration.

Every run saves two diagnostics files:

- `proposal_diagnostics.json`: metadata, definitions, counts, and all records
- `proposal_diagnostics.jsonl`: one record per line, also written incrementally
  while the run is active

The usual `trajectory.json`, `runhistory.json`, `incumbent.json`, `runtime.json`,
and SMAC metadata files are saved as well.

## Submit

Validate the matrix without submitting:

```bash
python submit_uncertainty.py --dry-run
python submit_uncertainty.py --list-jobs
```

Submit all 36 one-run tasks:

```bash
python submit_uncertainty.py
```

Each task requests one CPU, 4 GB RAM, and two hours on partition `c23ms`.
Complete runs are validated and skipped on rerun or requeue.

Outputs are stored under:

```text
smac_output/benchmark_seed_<benchmark>/fixed_depth_<depth>/<smac_seed>/
```

Run the focused tests with:

```bash
python test_uncertainty.py
```

## Analytics

`analyze_uncertainty.ipynb` creates one grouped five-panel figure for every
benchmark-seed/SMAC-seed combination. The panels show best-so-far regret,
proposal EI, predicted standard deviation, online absolute prediction error,
and online standardized prediction error. No running averages are used.

For a causal online comparison, the observed cost is the first runhistory
evaluation of the proposed configuration completed after its acquisition
selection. Figures are saved under `plots/`.

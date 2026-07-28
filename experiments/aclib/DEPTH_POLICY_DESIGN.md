# Deterministic ACLib adaptive-depth policy design

The policy implementation is shared by the four `02_policies` experiments in
`policy_experiment.py`. Every run has 5,000 trials, uses SMAC seeds 0–2, all
training instances, no test instances, 100 RF trees, split/leaf size 1, no
PCA, zero random-design probability, local SMAC, `deterministic=True`, and a
fixed ACLib target/quantile seed of 0.

Depth changes are installed immediately before `model.train()`. Consequently,
the recorded requested depth always describes the forest that was actually
fitted; proposals from an already-fitted forest are never relabeled with the
depth intended for the next fit.

## Saturation policies

`saturation_k50`, `saturation_k100`, and `saturation_k250` start at depth 5.
A fit is saturated when the mean actual depth of its 100 trees is at least
`0.9 × requested maximum depth`. After respectively 50, 100, or 250
consecutive saturated **distinct RF fits**, the next fit's depth increases by
5. The streak resets after an unsaturated fit and after every transition.

## Rotating saturation policy

`rotating_saturation_k50` initially alternates the fitted depth between 5 and
10. Only fits at the higher depth contribute to its saturation streak. When
the higher depth is saturated for 50 consecutive high-depth fits, the lower
depth is removed and `higher + 5` is added. Thus `{5,10}` becomes `{10,15}`,
then `{15,20}`, and so on.

## Error/uncertainty trend policy

`error_variance_trend_25` starts at depth 5. For every newly proposed
configuration after the RF is fitted, it records:

1. the RF prediction marginalized over all training instances;
2. the RF prediction variance;
3. deterministic mean PAR10 obtained by evaluating the configuration on all
   training instances with quantile seed 0;
4. standardized absolute error
   `abs(actual - prediction) / max(sqrt(variance), numerical_floor)`.

Over the latest 25 proposed configurations, the log standardized error must
have all of the following:

- Spearman-style rank correlation with time of at least 0.5;
- fitted growth across the window of at least 1.25×;
- second-half median greater than first-half median.

When all conditions hold, the next RF depth increases by 5 and a fresh
25-proposal window begins. This uses standard deviation rather than raw
variance so prediction error and uncertainty have matching units.

## Seed-specific incumbent oracle

`oracle_incumbent_depth` reads the seed's fixed-depth incumbent events from
the corresponding `01_initial_*/analytics_cache` file. At every trial it
selects the depth whose then-current incumbent has the lowest independently
validated full-training PAR10 at quantile seed 0. Ties prefer the shallower
depth.

The schedule only compares trials covered by all five fixed-depth runs. It
then holds the last selected depth:

- Clasp Queens: through trial 5,000;
- Clasp weighted-sequence: through trial 5,000;
- CPLEX RCW: through trial 1,741;
- LPG Zenotravel: through trial 1,398.

The exact schedule and source-file SHA-256 are frozen into each oracle run's
`oracle_schedule.json`, preventing a resumed job from changing policy if the
analytics cache is later regenerated.

## Recorded artifacts

Each run records normal SMAC state plus:

- `policy_events.jsonl`: fit observations, calibration observations, and
  depth changes;
- `policy_state.json`: resume-safe policy state;
- `configuration_telemetry.jsonl`: proposal-time predictions, uncertainty,
  EI, and all 100 actual tree depths;
- `oracle_schedule.json`: frozen schedule for oracle runs;
- `run_metadata.json`, `summary.json`, `trajectory.json`, and
  `completed.json`.

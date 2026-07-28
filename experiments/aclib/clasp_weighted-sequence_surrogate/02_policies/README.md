# Clasp weighted-sequence deterministic adaptive-depth policies

This folder runs six RF-depth policies for 5,000 trials and SMAC seeds 0, 1,
and 2. All 240 training instances are used; test instances are excluded.
The SMAC scenario is deterministic and every ACLib surrogate evaluation uses
quantile seed 0.

Run `run_policies.py --smoke-check`, then `run_policies.py --dry-run
--list-jobs` to inspect the 18 jobs. Running without an option submits them.

The seed-specific oracle schedule is derived from
`01_initial_cw/analytics_cache/incumbent_trajectory_events.csv`, using
full-training quantile-seed-0 incumbent performance.

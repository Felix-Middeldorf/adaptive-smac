# CPLEX RCW deterministic adaptive-depth policies

This folder runs six RF-depth policies for 5,000 trials and SMAC seeds 0, 1,
and 2. All 495 training instances are used; test instances are excluded.
The SMAC scenario is deterministic and every ACLib surrogate evaluation uses
quantile seed 0.

Run `run_policies.py --smoke-check`, then `run_policies.py --dry-run
--list-jobs` to inspect the 18 jobs. Running without an option submits them.

The seed-specific oracle uses the common covered prefix of all five fixed
depths in `01_initial_cr/analytics_cache/incumbent_trajectory_events.csv`,
then holds the final selected depth through trial 5,000.

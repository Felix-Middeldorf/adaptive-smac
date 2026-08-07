# 5,000-trial O1 deterministic RF-policy study

This study uses dimensions `10, 25, 50, 100`, ten benchmark instances, 5,000 SMAC trials, and seeds `0` through `9`.

Run this command to submit it (submission is the default):

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  "/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/40_05_5000 trials/submit_experiment.py"
```

The submission contains 280 trajectories.

- 200 fixed controls: depths `5, 10, 20, 30, 20000` × four dimensions × ten seeds. They use 100 trees, `min_samples_split=2`, `min_samples_leaf=1`, and feature ratio `5/6` (the previous-study default).
- 40 compact-Mistral policies: four dimensions × ten seeds. They use `mistralai-mistral-small-4-119b` through RWTHGPT and keep 100 trees fixed.
- 40 chronological 80/20 holdout RF-selection policies: four dimensions × ten seeds and no API calls.

All tasks may start immediately. Each Mistral worker retries a RWTHGPT 429 or malformed completion after 15 seconds until strict JSON validates. Persisted decisions are reused on requeue.

## Compact Mistral policy

Mistral starts with 100 trees, depth 20, split 3, leaf 3, feature ratio `5/6`. At trials `100, 250, 500, 1000, 1500, ..., 4500`, it receives ten chronological trial aggregates, current settings, prior decisions, objective-space dimensions/bounds, allocation, EI/prediction diagnostics, and recent RF fits. It may select depth `1..30`, split `2..10`, leaf `1..10`, and feature ratio `(0,1]`; it cannot change tree count.

The compact prompt includes `capacity_and_calibration`:

```json
{"fraction_trees_exactly_at_depth_cap":0.63,"tree_depth_q10":14,"tree_depth_median":19,"tree_depth_q90":20,"matched_online_error_count":183,"median_standardized_error":0.88,"q90_standardized_error":2.71,"coverage_within_1_std":0.61,"coverage_within_2_std":0.87,"error_std_correlation":0.29}
```

The tree metrics summarise the latest 50 fitted forests. A high cap fraction indicates genuinely constrained capacity. Standardized error is proposal-time proxy error divided by predicted standard deviation; coverage is the fraction within one/two standard deviations; error/standard-deviation correlation reports whether uncertainty tracks error. These distinguish constrained trees, merely deep trees, confidently wrong predictions, and appropriate uncertainty.

## Chronological 80/20 holdout selector

This policy starts with 100 trees, depth 20000, split 2, leaf 1, and feature ratio `5/6`. Once 500 trials (and then each further 500-trial checkpoint through 4500) have been passed, SMAC's next ordinary surrogate retraining step trains nine candidates on the earliest 80% of its then-current encoded data and scores them on the newest 20%. The lowest validation MSE simply replaces the live surrogate settings for that normal refit. No additional checkpoint-only model fit is introduced.

The 3×3 candidate grid is `min_samples_leaf ∈ {1,2,3}` × `feature_ratio ∈ {0.75,0.90,1.00}`, with trees 100, depth 20000, split 2 fixed. Ties use smaller leaf then smaller feature ratio. Scores/selections live in `llm_policy_state.json` and `llm_policy_events.jsonl`.

## Results and analysis

Results use `results/dimension_<d>/benchmark_seed_40/<policy>/<seed>/`. The [analysis notebook](analyze_5000_trials.ipynb) loads completed runs, plots final best-so-far distributions, ranks policies within/across dimensions, and displays Mistral telemetry and holdout selections. Regenerate it with `build_analysis_notebook.py` after edits.

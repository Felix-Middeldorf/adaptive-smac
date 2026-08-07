# RWTHGPT Mistral O1 policies

This directory contains the Mistral rerun of the O1 compact LLM policy for
dimensions 10, 25, 50, and 100. There are five SMAC seeds per dimension and
1,000 SMAC trials per run. The model is requested through the RWTHGPT
OpenAI-compatible endpoint:

```text
mistralai-mistral-small-4-119b
```

The API key is read on the worker from:

```text
~/.config/kiconnect/rwthgpt_api_key
```

Both policies use the same checkpoints: 100, 250, and 500 completed trials.
Both use the same RF parameter ranges, but `n_trees` is fixed to 100 and is
never selected by Mistral:

| Parameter | Allowed values |
|---|---|
| `n_trees` | fixed at 100 |
| `max_depth` | integer 1–30 |
| `min_samples_split` | integer 2–10 |
| `min_samples_leaf` | integer 1–10 |
| `feature_ratio` | real number in `(0, 1]` |

The two policies are:

- `rwthgpt_mistral_compact_fixed_100_trees`: compact ten-window summaries,
  corresponding to the original compact LLM policy.
- `rwthgpt_mistral_every_second_fixed_100_trees`: every second completed
  trial, including its configuration and proposal-time SMAC diagnostics.

## Common prompt envelope

Every request is sent as two chat messages.

System message:

```text
You select random-forest surrogate hyperparameters for SMAC. Return only the requested JSON object.
```

The user message starts with the same decision instruction for both policies:

```text
Choose random-forest surrogate hyperparameters for the next phase of a
SMAC algorithm-configuration run. Lower objective values are better. The
number of trees is fixed to 100 and must not be proposed or changed.
```

The model is told to choose only these four parameters:

```text
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: number in (0, 1]
```

The required response schema is:

```json
{
  "max_depth":  integer,
  "min_samples_split": integer,
  "min_samples_leaf": integer,
  "feature_ratio": number,
  "confidence": number from 0 to 1,
  "reason": "concise string"
}
```

There must be no `n_trees` key in the model response. The runner inserts
`n_trees=100` after validation. Cached normalized decisions do contain
`n_trees=100`; the validator accepts that cached form only when the value is
exactly 100.

The prompt also defines the SMAC diagnostic fields:

```text
expected_improvement, predicted_mean, and prediction_variance are recorded at
proposal time. absolute_proxy_error compares the proposal-time prediction with
the first observed instance cost for that configuration.
best_config_mean_cost is the best running mean over configurations.
```

The response is parsed as JSON. Markdown code fences are tolerated. Empty,
malformed, or schema-invalid responses are retried after 15 seconds. HTTP 429
responses are also retried after 15 seconds.

## Compact-summary prompt

### Data passed to the model

The compact policy uses the existing `o1_compact_llm_runner` summary. At a
checkpoint `c`, it includes:

- the checkpoint and the actual number of completed trials;
- the objective direction and O1 search-space description;
- the current RF settings and the previous Mistral decisions;
- the fixed `n_trees=100` declaration and the four selectable ranges;
- evaluation allocation: completed trials, number of unique configurations,
  evaluations per configuration, and the best running configuration mean cost;
- all trials 1 through `c`, grouped into ten consecutive equal-width windows;
- per-window observed-cost distributions;
- per-window expected-improvement distributions;
- per-window marginalized RF predicted-mean distributions;
- per-window prediction-variance distributions;
- per-window absolute and relative proxy-error distributions;
- the correlation between log proxy error and log prediction variance;
- recent RF-fit windows, including training rows, realized tree depth, and
  depth utilization.

Raw configuration coordinates are not included in this policy. The ten windows
are a compact representation of the runhistory.

The user prompt ends with:

```text
DATA
<JSON compact summary>
```

### Example compact prompt

The following is a shortened example. A real request contains all ten windows
and recent RF-fit summaries.

```text
Choose random-forest surrogate hyperparameters for the next phase of a
SMAC algorithm-configuration run. Lower objective values are better. The
number of trees is fixed to 100 and must not be proposed or changed.
The data are ten aggregate windows of the completed runhistory.

Choose only from:
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: number in (0, 1]

Return exactly one JSON object with these keys and no others:
{"max_depth": integer, "min_samples_split": integer,
 "min_samples_leaf": integer, "feature_ratio": number,
 "confidence": number from 0 to 1, "reason": concise string}

Definitions: expected_improvement, predicted_mean, and prediction_variance are
recorded at proposal time. absolute_proxy_error compares the proposal-time
prediction with the first observed instance cost for that configuration.
best_config_mean_cost is the best running mean over configurations.

DATA
{
  "checkpoint": 250,
  "actual_completed_trials_at_call": 250,
  "objective_direction": "minimize",
  "current_rf_settings": {
    "n_trees": 100,
    "max_depth": 12,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "feature_ratio": 0.8
  },
  "fixed_rf_hyperparameters": {"n_trees": 100},
  "previous_decisions": [
    {"checkpoint": 100, "settings": {"n_trees": 100, "max_depth": 12,
      "min_samples_split": 3, "min_samples_leaf": 2,
      "feature_ratio": 0.8}, "confidence": 0.72}
  ],
  "evaluation_allocation": {
    "completed_trials": 250,
    "unique_configurations": 139,
    "final_best_config_mean_cost": -123456.7
  },
  "trial_windows": [
    {
      "window": 10,
      "trials": [226, 250],
      "best_cost_improvement_in_window": 842.1,
      "expected_improvement": {"mean": 18.4, "n": 25},
      "relative_proxy_error": {"mean": 0.12, "n": 25},
      "log_error_variance_correlation": 0.41
    }
  ],
  "recent_rf_fit_windows": [
    {
      "settings": {"n_trees": 100, "max_depth": 12,
        "min_samples_split": 3, "min_samples_leaf": 2,
        "feature_ratio": 0.8},
      "actual_tree_depth_mean": {"mean": 10.8, "n": 4},
      "depth_utilization": {"mean": 0.90, "n": 4}
    }
  ]
}
```

An example valid Mistral response is:

```json
{
  "max_depth": 16,
  "min_samples_split": 3,
  "min_samples_leaf": 2,
  "feature_ratio": 0.8,
  "confidence": 0.71,
  "reason": "Recent trees approach the depth cap while proxy error is improving; increase depth modestly and retain the stable regularization and feature ratio."
}
```

## Every-second-trial prompt

### Data passed to the model

The every-second policy uses the same common prompt envelope and the same
checkpoints, but replaces the ten-window aggregate with explicit records for
trials 2, 4, 6, ..., up to the checkpoint. At checkpoint 500 this is 250
records.

Each record contains:

- trial number, configuration ID, and instance;
- all configuration coordinates (`x_0`, ..., `x_(d-1)`);
- observed cost;
- proposal-time expected improvement;
- proposal-time RF predicted mean and prediction variance;
- absolute and relative proxy error;
- the running best configuration mean cost.

The payload also includes the dimension, bounds `[-100, 100]`, checkpoint,
total budget, remaining budget, fixed tree setting, current RF settings, and
previous Mistral decisions.

Odd-numbered trials are intentionally omitted. The records are still the
actual runhistory order, not a random sample.

The user prompt describes this mode as:

```text
The data include every second completed trial with its configuration,
observed cost, and proposal-time SMAC diagnostics.
```

### Example every-second-trial prompt

This is a shortened example; a real checkpoint-500 request contains 250 trial
records.

```text
Choose random-forest surrogate hyperparameters for the next phase of a
SMAC algorithm-configuration run. Lower objective values are better. The
number of trees is fixed to 100 and must not be proposed or changed.
The data include every second completed trial with its configuration,
observed cost, and proposal-time SMAC diagnostics.

Choose only from:
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: number in (0, 1]

Return exactly one JSON object with these keys and no others:
{"max_depth": integer, "min_samples_split": integer,
 "min_samples_leaf": integer, "feature_ratio": number,
 "confidence": number from 0 to 1, "reason": concise string}

DATA
{
  "summary_mode": "every_second_completed_trial_with_configuration_and_smac_diagnostics",
  "checkpoint": 500,
  "actual_completed_trials_at_call": 503,
  "total_trial_budget": 1000,
  "remaining_trials": 500,
  "configuration_space": {
    "total_dimensions": 25,
    "parameter_type": "continuous real-valued",
    "bounds": [-100.0, 100.0],
    "hierarchy": "none"
  },
  "fixed_rf_hyperparameters": {"n_trees": 100},
  "current_rf_settings": {
    "n_trees": 100,
    "max_depth": 16,
    "min_samples_split": 3,
    "min_samples_leaf": 2,
    "feature_ratio": 0.8
  },
  "previous_decisions": [],
  "every_second_trial_records": [
    {
      "trial": 2,
      "config_id": 1,
      "instance": "0",
      "configuration": {"x_0": 12.4, "x_1": -71.0, "x_2": 4.8},
      "observed_cost": -812.5,
      "expected_improvement": null,
      "predicted_mean": null,
      "prediction_variance": null,
      "absolute_proxy_error": null,
      "relative_proxy_error": null,
      "best_config_mean_cost": -812.5
    },
    {
      "trial": 4,
      "config_id": 3,
      "instance": "0",
      "configuration": {"x_0": -3.1, "x_1": 44.2, "x_2": 18.6},
      "observed_cost": -905.2,
      "expected_improvement": 32.1,
      "predicted_mean": -870.0,
      "prediction_variance": 421.5,
      "absolute_proxy_error": 35.2,
      "relative_proxy_error": 0.039,
      "best_config_mean_cost": -905.2
    }
  ]
}
```

An example valid response is:

```json
{
  "max_depth": 20,
  "min_samples_split": 4,
  "min_samples_leaf": 2,
  "feature_ratio": 0.7,
  "confidence": 0.68,
  "reason": "The explicit records show rising depth utilization and persistent proxy error; increase depth modestly while retaining light regularization and adding feature subsampling for stability."
}
```

## Token-size note

The runner checks prompt size with a character-based estimate of
`ceil(len(prompt) / 4)`. The actual RWTHGPT usage metadata can be larger. In
observed checkpoint-500 calls, actual `prompt_tokens` were approximately:

| Dimension | Actual input tokens |
|---:|---:|
| 10 | 69k |
| 25 | 127k |
| 50 | 223k |
| 100 | 411k–415k |

The 100D request was accepted by the endpoint in the observed runs despite the
documented 262.1k context limit. This behavior should not be assumed to be
stable; reducing the raw-record payload would be prudent for future reruns.

## Submission

The submission script creates 40 tasks: two policies × four dimensions × five
SMAC seeds. All tasks may start immediately; rate limiting is handled inside
each worker. Use `--smoke-check` for a non-submitting validation and run the
script without flags to submit the array.

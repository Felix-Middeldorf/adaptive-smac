# Direct-OpenAI RF policy on deterministic O1

This experiment runs 10 independent 1,000-trial SMAC optimizations: benchmark
seeds 40 and 42 crossed with SMAC seeds 0 through 4. Every run uses 10
dimensions, the same 10 deterministic instances, `PYTHONHASHSEED=12345`, no
random-design injection, and the local SMAC checkout.

Trials 0–99 use `AlgorithmConfigurationFacade`'s native RF defaults: 10 trees,
depth 20, split/leaf size 3, and feature ratio 5/6. At the first configuration
selection after 100, 250, and 500 completed target trials, the runner directly
calls the OpenAI Responses API with `gpt-5.6-terra`, medium reasoning effort,
`store=false`, and a strict JSON schema. There is no Codex CLI subprocess.

Each request contains exactly 100 evenly spaced runhistory rows up to its
checkpoint. Thus checkpoint 100 includes trials 1–100, while checkpoint 500
includes trials 5, 10, ..., 500. Every row includes its trial number, the actual
values of all ten configuration dimensions, instance, status, and observed
cost, plus the EI, marginalized prediction, RF variance, first-instance cost
proxy, absolute proxy error, and relative proxy error when proposal-time model
information exists. Evaluation time is not sent to the model.

The prediction, variance, and EI are captured when the configuration is
proposed. The cheap real-value proxy is that configuration's first observed
target cost. Absolute error is the absolute difference between the marginalized
prediction and this proxy; relative error divides it by
`max(abs(proxy), 1e-12)`. This is deliberately a proxy, not full-instance
validation and therefore adds no target-function evaluations.

The API key remains in `/home/io632776/.config/openai/smac_api_key` with mode
600. The Slurm setup reads it into `OPENAI_API_KEY` at job runtime; neither the
secret nor its value is stored in code, Submitit metadata, prompts, or results.
Validated decisions and response IDs/token usage are cached in each run so a
Slurm requeue does not repeat a completed API call.

A measured checkpoint-100 prompt is about 69,600 characters. For this mostly
numeric JSON, expect approximately 18,000–25,000 input tokens per API call and
up to 1,000 output/reasoning tokens for the structured decision. The exact
counts are recorded from the API response in each checkpoint's
`openai_response_metadata.json`. All 10 jobs together make 30 calls, so a
reasonable experiment-wide estimate is 540,000–750,000 input tokens plus at
most 30,000 output/reasoning tokens.

Jobs use project `thes2388`, four hours, 4 GB, and one CPU:

```bash
./submit_llm.py --smoke-check
./submit_llm.py --list-jobs
./submit_llm.py
```

## Compact-summary policy

`submit_compact_llm.py` defines a second, independent set of ten jobs with the
same benchmark seeds, SMAC seeds, 1,000-trial budget, checkpoints, model, and
reasoning effort. Existing `openai_llm_rf_policy` results are never overwritten;
the new runs use the policy/output name `openai_compact_llm_rf_policy` and the
separate Submitit log directory `submitit_logs_compact`.

Instead of sending 100 raw trial records and all ten configuration coordinates,
each request summarizes every completed trial into ten consecutive windows.
Each window contains distributions for observed costs, EI, predictions,
prediction variance, absolute/relative first-instance proxy errors, incumbent
progress, evaluation allocation, and error--variance correlation. The latest
50 RF fits are reduced to five aggregate windows containing actual tree-depth
and depth-utilization statistics. No extra target-function evaluations are
performed.

Every compact request also states that the underlying objective has ten
continuous parameters `x_0` through `x_9`, each constrained to
`-100 <= x_i <= 100`, with no conditional parameters or forbidden
combinations. These are search-space definitions, not sampled configuration
values.

With the search-space description included, the checkpoint-500 aggregate
fixture produces an 11,244-character prompt, and a real 101-trial SMAC
integration run produces a 12,808-character checkpoint-100 prompt: roughly
2,800--3,200 tokens by a simple characters/4 estimate. Actual API token usage
is saved in `openai_response_metadata.json`; these estimates are about one
tenth of the original policy's measured input size.

The structured output permits every valid value in these ranges:

- trees: integer 1--100
- maximum depth: integer 1--30
- minimum split size: integer 2--10
- minimum leaf size: integer 1--10
- feature ratio: any number in `(0, 1]`

Validate or submit this policy with:

```bash
./submit_compact_llm.py --smoke-check
./submit_compact_llm.py --list-jobs
./submit_compact_llm.py
```

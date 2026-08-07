# Codex-controlled SMAC RF policy

Three deterministic 2,500-trial SMAC runs use seeds 0, 1, and 2. Trials 0–499
use `AlgorithmConfigurationFacade`'s native RF defaults: 10 trees, depth 20,
split/leaf size 3, feature ratio 5/6, and PCA=4. At the first RF-selection
checkpoint at or after completed trials 500, 1000, and 1500, the job calls the
locally installed Codex CLI and applies its structured RF choice to the next
model fit.

Codex receives only anonymized aggregate telemetry: runhistory cost and timing
distributions, EI/prediction distributions, RF fitting time, and actual tree
depth utilization. The prompt contains no benchmark name, instance path, or
target-configuration value. Codex runs from an isolated temporary directory in
a read-only sandbox with user configuration and repository rules disabled.

Allowed decisions are bounded to 10/50/100 trees; depths 5/10/15/20/30;
split sizes 2/3/4/8; leaf sizes 1/3/5; and feature ratios 0.3/0.5/5/6/1.0.
PCA remains fixed at 4 because changing the feature representation during a
run would make model-state comparisons unsafe.

Every prompt, anonymized summary, raw response, validated decision, invocation
metadata, transition, and RF fit observation is stored under the native run
directory. Decisions are cached and reused after Slurm requeue, preventing a
checkpoint from consuming a second Codex call.

The jobs use the local SMAC checkout, all training instances, no test
instances, target quantile seed 0, 0% random design, project `thes2388`, 16
hours, 4 GB, and one CPU. Codex reuses the cached ChatGPT authentication; no
API key is placed in the job environment.

```bash
./submit_experiment.py --smoke-check
./submit_experiment.py --list-jobs
./submit_experiment.py
```

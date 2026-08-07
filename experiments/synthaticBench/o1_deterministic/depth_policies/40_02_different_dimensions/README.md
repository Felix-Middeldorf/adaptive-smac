# Compact LLM policy across objective dimensions

This study keeps benchmark seed 40, 10 deterministic instances, 1,000 SMAC
trials, and SMAC seeds 0–4. It varies the O1 objective dimension over 2, 5, 25,
50, and 100 and compares the compact LLM-selected RF policy with 100-tree fixed
controls at maximum depths 5, 10, 15, 20, and 30.

The compact policy starts from SMAC's experiment defaults (10 trees, depth 20,
split/leaf 3, feature ratio 5/6, PCA 4) and calls GPT-5.6 Terra with medium
reasoning after trials 100, 250, and 500. The prompt states the dimension and
the continuous box constraints for every objective parameter. The comparison
controls use 100 trees and otherwise retain the fixed RF defaults while varying
only maximum depth. Their distinct result directories prevent collisions with
the earlier 10-tree fixed runs.

The comparison consists of the 25 existing compact-policy runs and 125 new
100-tree fixed-depth jobs. All use the local `external/SMAC3`, Slurm account
`thes2388`, one CPU, 4 GB, and four hours. Only compact-policy jobs read the API
key from `~/.config/openai/smac_api_key`; it is never copied into this directory.

`submit_experiment.py` is the original combined submission script and retains
its 10-tree fixed controls for reproducibility. Those controls are no longer
loaded by the analysis notebook.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_experiment.py --smoke-check
```

To submit all 125 replacement fixed controls with 100 trees:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_fixed_100_trees.py --smoke-check
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_fixed_100_trees.py
```

To submit only the 50D and 100D 100-tree controls:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_fixed_100_trees.py --dimensions 50 100
```

## Cross-dimension LLM configurations

`llm_config_cross_dimension.py` freezes two settings proposed in the compact-LLM
runs for every source dimension: the seed-0 decisions at checkpoints 100 and
500. `submit_llm_config_cross_dimension.py` evaluates all ten settings on all
five target dimensions. Running it without arguments submits the requested 50
jobs using evaluation SMAC seed 0:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_llm_config_cross_dimension.py
```

To evaluate the same 50 dimension/configuration pairs with all five SMAC seeds
(250 jobs), use:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_llm_config_cross_dimension.py --smac-seeds 0 1 2 3 4
```

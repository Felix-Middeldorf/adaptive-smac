# Clasp weighted-sequence brute-force RF schedules

This experiment evaluates 50 diverse, deterministic three-phase random-forest
schedules for each of SMAC seeds 0, 1, and 2, giving 150 jobs. Settings apply
from trials 0–499, 500–1,999, and 2,000–4,999. In practice a new phase becomes
active at the first RF retraining checkpoint after its completed-trial
threshold.

Each phase chooses:

- maximum depth from 5, 10, 15, 20, or 30;
- `min_samples_split` from 2, 4, or 8;
- feature ratio from 0.3, 0.5, 5/6, or 1.0.

The catalog includes constant controls, monotone and reverse complexity ramps,
one-factor schedules, non-monotone schedules, and broad modular combinations.
The exact definitions are stored in `schedule_catalog.json`.

Every job keeps 100 trees, `min_samples_leaf=1`, PCA=4, random-design
probability 0%, 5,000 trials, all 240 training instances, deterministic target
quantile seed 0, the sampled initial configuration from `01_initial_cw`, local
SMAC, `PYTHONHASHSEED=0`, Slurm account `lect0190`, 16 hours, and 4 GB.

# Depth-only compact LLM ablation

This experiment tests whether dynamic tree depth explains the high-dimensional
results from `40_02_different_dimensions`. It uses dimensions 25, 50, and 100,
benchmark seed 40, ten deterministic instances, 1,000 trials, and SMAC seeds
0–4.

Every LLM and fixed policy uses exactly 100 trees, minimum leaf size 1, minimum
split size 2, feature ratio 5/6, PCA 4, and random-design probability 0%. Split
size 2 is used because scikit-learn trees reject split size 1. The compact LLM
may choose only an integer maximum depth from 1 through 30 at checkpoints 100,
250, and 500. Fixed controls use depths 5, 10, 15, 20, and 30.

The study contains 15 LLM jobs and 75 fixed jobs (90 total). Jobs use local
`external/SMAC3`, account `thes2388`, one CPU, 4 GB, and a four-hour limit.

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_experiment.py --smoke-check
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_experiment.py --submit
```

# Deterministic LPG Zenotravel fixed depths

Fifteen jobs compare RF depth limits 5, 10, 15, 20, and 30 for SMAC
seeds 0, 1, and 2. Every run uses 5,000 trials, all training instances,
`deterministic=True`, and target quantile seed 0. The initial configuration
is reused from `01_initial_lz`.

The SMAC RF uses 100 trees, minimum split/leaf size 1, no PCA, and zero
random-design probability. Jobs use the local `external/SMAC3` checkout,
Slurm account `lect0190`, and `PYTHONHASHSEED=0`.

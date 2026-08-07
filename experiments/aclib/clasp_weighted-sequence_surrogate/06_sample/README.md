# Clasp weighted-sequence random sample

This experiment draws 1,000 unique configurations from the official Clasp
weighted-sequence ConfigSpace using seed `20260729`. Each configuration is
evaluated on all 240 training instances with the deterministic EPM median
(quantile seed zero) and five reproducible stochastic EPM quantile draws.

The resumable result tensor is `results/costs.npy`, with axes configuration,
quantile seed, and training instance. `results/configuration_seed_summary.csv`
contains convenient aggregate statistics. The notebook
`analyze_samples.ipynb` separates variability across configurations, instances,
and stochastic EPM draws.

Run locally with `./run_sampling.py` or submit one Slurm job with
`./submit_sampling.py`. The submitit job uses project `lect0190`, partition
`c23ms`, 16 hours, 4 GB, one CPU, and `PYTHONHASHSEED=0`.


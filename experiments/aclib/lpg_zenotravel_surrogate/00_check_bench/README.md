# LPG Zenotravel surrogate check

This deliberately small validation checks:

1. 50 identical requests through the official ACLib conversion/prediction path
   and the custom in-process wrapper;
2. the EPM median against all 10,000 identical configuration–instance pairs in
   the official `random_train.json` real-performance archive.

Run:

```bash
./run_validation.py
```

Use `--overwrite` to replace existing results. Open `analyze_validation.ipynb`
afterwards. Generated files are stored in `results/` and ignored by Git.

The downloaded input is
`external/aclib-performance-data/lpg_zenotravel`, from the
[AutoML EPM data page](https://www.automl.org/automated-algorithm-design/performance-prediction/epms/).
Because that data contributed to EPM construction, this is an integrity check,
not an independent generalization benchmark.

## Result

The custom and official wrapper-core paths agree exactly on all 50 requests,
so the custom call path is not the problem. The conditional median agrees
poorly with the single archived real run per pair: Spearman 0.113, 30.1% within
a factor of two, and timeout recall 18.2%. Because LPG is stochastic, this
minimal median-versus-one-run check does not by itself invalidate the full
predictive distribution, but it does not certify surrogate fidelity either.
Deterministic quantile-seed-0 experiments should be interpreted as optimizing
a substantially different median surface.

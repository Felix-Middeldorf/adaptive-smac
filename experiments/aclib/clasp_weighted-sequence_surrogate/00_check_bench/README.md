# Clasp weighted-sequence surrogate check

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
`external/aclib-performance-data/clasp_weighted-sequence`, from the
[AutoML EPM data page](https://www.automl.org/automated-algorithm-design/performance-prediction/epms/).
Because that data contributed to EPM construction, this is an integrity check,
not an independent generalization benchmark.

## Result

The custom and official wrapper-core paths agree exactly on all 50 requests.
The EPM ranking is strong (Spearman 0.968), but timeout conversion is not
working as intended at quantile seed 0: 47.43% of predictions are numerically
`899.99994`, just below the 900-second cutoff. The official wrapper therefore
labels them solved instead of assigning PAR10 9000. This affects the official
and custom paths equally, so exact wrapper parity does not make the resulting
objective semantically correct.

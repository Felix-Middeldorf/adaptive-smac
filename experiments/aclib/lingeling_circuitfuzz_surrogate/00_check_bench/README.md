# Lingeling CircuitFuzz surrogate check

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
`external/aclib-performance-data/lingeling_circuitfuzz`, from the
[AutoML EPM data page](https://www.automl.org/automated-algorithm-design/performance-prediction/epms/).
Because that data contributed to EPM construction, this is an integrity check,
not an independent generalization benchmark.

## Result

The custom and official wrapper-core paths agree exactly on all 50 requests.
Agreement with the 10,000 archived points is extremely strong: Spearman 0.999,
99.76% within a factor of two, and timeout recall 99.90%. This check supports
using the installed Lingeling CircuitFuzz surrogate.

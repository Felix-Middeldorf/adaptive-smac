# ACLib cross-benchmark SMAC runtime mechanisms

This experiment identifies which parts of SMAC cause the large runtime
differences between the five 5,000-trial ACLib experiments. It uses the same
downloaded benchmarks, all training instances, local SMAC checkout, RF
settings, initial configurations, and target functions as the main runs.

The profiling runs are deliberately limited to 256 target trials. Lingeling
already takes hours per acquisition/retraining cycle, so a second set of
5,000-trial profiles would be unnecessarily expensive.

## Suites

- `baseline`: depth 10, three SMAC seeds, five benchmarks (15 jobs). This is
  the clean cross-benchmark comparison.
- `mechanisms`: depth 10, seed 0, seven one-factor ablations for every
  benchmark (35 jobs). The matching seed-0 baseline comes from `baseline`.
- `depth`: baseline at depths 5 and 30, three seeds, five benchmarks
  (30 jobs). Together with depth 10 this separates benchmark effects from RF
  depth effects.
- `all`: all of the above, exactly 80 distinct jobs.

The ablations are:

1. stock EI without the custom telemetry callback;
2. no instance features in SMAC's RF;
3. 500 instead of 5,000 acquisition challengers;
4. one instead of ten local-search starts;
5. acquisition random search without local search;
6. retraining after 64 instead of 8 proposed configurations;
7. no periodic optimizer saves.

Changing an acquisition mechanism also changes the search trajectory. For
that reason, the profiler records both total time and inclusive phase timing,
call counts, number of configurations, and input sizes. Inclusive nested
times must not be added together.

## Timed mechanisms

The profiler records:

- target evaluation;
- `ask`, `tell`, and configuration selection;
- runhistory collection;
- RF training;
- incumbent (`x_best`) calculation;
- acquisition update and complete acquisition maximization;
- random-search and local-search portions;
- individual local-search walks;
- one-exchange neighborhood generation, including neighbor counts;
- marginalized RF prediction and its configuration-instance expansion;
- ConfigSpace sampling;
- telemetry snapshot and durable append;
- optimizer, runhistory, and intensifier persistence.

Each job also writes an optional Python `cProfile` file when run manually with
`--cprofile`.

## Commands

Use the ACLib surrogate Python environment:

```bash
PYTHONHASHSEED=0 \
PYTHONPATH="$PWD/external/SMAC3:$PWD/external/aclib-surrogates/epm:$PWD/experiments/aclib" \
/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python \
  "experiments/aclib/additional experiments/01_runtime_mechanisms/submit_jobs.py" \
  --suite all --dry-run --list-jobs
```

Submit one suite by removing `--dry-run`:

```bash
/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python \
  "experiments/aclib/additional experiments/01_runtime_mechanisms/submit_jobs.py" \
  --suite baseline
```

Summarize all completed and partial profiles:

```bash
/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python \
  "experiments/aclib/additional experiments/01_runtime_mechanisms/analyze_results.py"
```

Results are written below `results/<benchmark>/<variant>_depth_<depth>/<seed>/`.
Submission uses project `lect0190`, one CPU per task, fixed
`PYTHONHASHSEED=0`, and the repository's local `external/SMAC3`.

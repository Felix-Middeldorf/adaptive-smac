# C2 Parameter Interactions — Griewank depth policies

This repeats `01_initial` for the C2 `griewank` function instead of `ackley`.

- fixed depths: 3, 6, 9, 12, 15, 20
- staged policy: depth 9 for the first 500 trials, then depth 15
- SMAC seeds: 0–9
- problem seed: 52
- trials: 1000
- instances: 10
- dimension: 10
- Submitit resources: 20 minutes, 1 CPU, 4 GB memory

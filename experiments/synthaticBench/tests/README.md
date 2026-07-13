# SynthACticBench instance tests

Run these tests from the repository root with the experiment interpreter:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  -m unittest discover -s experiments/synthaticBench/tests -v
```

The suite checks the pinned CARPS/SMAC environment, problem loading,
deterministic instance generation, instance offset propagation, seed behavior,
invalid inputs, rejection of unsupported multi-objective use, adaptive-depth
policy boundaries, and a real instance-aware SMAC optimization.

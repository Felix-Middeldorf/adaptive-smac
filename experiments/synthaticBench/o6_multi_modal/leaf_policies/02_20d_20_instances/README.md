# O6 leaf-size sweep, 20D and 20 instances

This folder tests fixed `min_samples_leaf` values on O6-Multimodal:

- leaf sizes: 1, 2, 3, 4, 5
- dimension: 20
- instances: 20
- trials: 1,500
- SMAC seeds: 0–4
- benchmark seed: 52
- `min_samples_split=1`
- 10 random initial configurations

Run `submit_fixed_leafs.py`, then use `analyze_leafs.ipynb`.

# O6 leaf policies, initial run

This folder tests O6-Multimodal leaf-size policies with:

- dimension: 10
- instances: 10
- trials: 1,000
- SMAC seeds: 0–4
- benchmark seed: 52
- `min_samples_split=1` for every run
- 10 random initial configurations

Submit scripts:

- `submit_fixed_leafs.py`: fixed leaf sizes 1, 2, 3, 4, and 5.
- `submit_adaptive_leafs.py`:
  - `rotate_leaf_5_4_3_2_1_every_100`
  - `rotate_leaf_4_3_2_1_every_100`
  - `rotate_leaf_3_2_1_every_100`
  - `staged_leaf_3_2_1_200_200_rest`

The staged policy uses leaf size 3 for trials 1–200, leaf size 2 for trials
201–400, and leaf size 1 afterwards.

# O1 extended minimum-split policies

This experiment extends the effective `min_samples_split` policy space to the
values 2, 3, 4, and 5. Split value 1 is omitted because scikit-learn maps it to
2 when `min_samples_leaf=1`.

The setup matches `01_initial`:

- benchmark seeds: 40-44
- SMAC seeds: 0-4
- trials per run: 1,000
- random-design probability: 0
- maximum tree depth: 2,000
- minimum leaf size: 1
- stage boundaries: completed trials 200 and 500

The previous experiment already contains the eight effective schedules using
only split values 2 and 3. This extension runs:

- fixed split values 4 and 5
- a deterministic space-filling sample of 25 from the 54 missing nonconstant
  three-stage schedules containing at least one 4 or 5

The greedy sampler prioritizes uncovered stage-pair combinations, Hamming
distance, and balanced per-stage marginals. The selected schedules cover all
48 pairwise stage/value combinations and all 16 directed adjacent-stage
transitions. Every split value occurs 6 or 7 times in each stage.

That gives 27 new policies and 675 SMAC runs. Policies are divided into three
shards for every benchmark-seed x SMAC-seed pair, producing 75 Slurm jobs.
Each job executes 9 policies sequentially, and at most 75 jobs run at once.

Dry-run validation:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python \
  experiments/synthaticBench/o1_deterministic/split_policies/01_02_more_splits/submit_more_split_policies.py \
  --dry-run
```

After both experiments are complete, execute `analyze_all_split_policies.ipynb`
from the repository root.

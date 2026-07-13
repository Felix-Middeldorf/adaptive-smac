# Feedback-driven depth experiment

This is a pilot test of online surrogate-depth selection on a held-out O6
condition:

- dimension: 20
- instances: 15
- trials: 1,500
- SMAC seeds: 0–4
- benchmark seed: 52
- SMAC default leaf and split settings: 3 and 3
- initial design: 10 random configurations

The feedback policy considers depths 9, 15, 20, and 30. It starts at depth 9.
Beginning at trial 200, it reassesses the depth every 100 trials:

1. The chronologically older 80% of the runhistory trains four temporary
   candidate forests.
2. The newest 20% is held out.
3. Candidates are scored by Spearman rank correlation on the holdout.
4. The smallest depth within 0.02 of the best score is preferred.
5. Increasing complexity requires a score improvement of at least 0.02.
6. A selected depth is retained for at least 200 trials.

The experiment compares the feedback policy with fixed depths 9, 15, 20, and
30 and with the earlier staged 3 → 6 → 9 → 20 policy. Random rotation is not
rerun because the previous experiments already established it as a weak
negative baseline.

Run the three submit scripts with the `adaptive-smac-synthactic-py311`
interpreter, then execute `analyze_depths.ipynb`.

# O6 RWTHGPT RF-policy comparison

This experiment evaluates O6-Multimodal dimensions 50 and 100 with ten
deterministic instances, benchmark seed 52, and 1,000 SMAC trials per run.

- Fixed baseline: SMAC default RF settings with 100 trees, six seeds per dimension.
- Initial choice: before any evaluation, RWTHGPT chooses a complete RF setting from
  the same ranges as the O1 compact policy; this selection is made independently
  for every SMAC run, and the setting then remains fixed.
- Dynamic choice: RWTHGPT starts from SMAC defaults and updates the whole RF
  setting at completed trials 100, 250, and 500 using compact run telemetry and
  all earlier decisions.

GPT-5.5 uses SMAC seeds 0–2. GPT-5.4-mini uses seeds 0–5, including 0–2.
The RWTHGPT key is read only on workers from
`~/.config/kiconnect/rwthgpt_api_key`.

The grid has 12 fixed-baseline, 18 initial-choice, and 18 dynamic-policy jobs
(48 total).

Run a non-submitting check with:

```bash
/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python submit_experiment.py --smoke-check
```

Run `submit_experiment.py` without arguments to submit all 48 jobs.

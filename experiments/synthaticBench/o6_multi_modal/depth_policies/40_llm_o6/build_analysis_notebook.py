from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "adaptive-smac-synthactic-py311",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []
cells.append(nbf.v4.new_markdown_cell("""# O6-Multimodal RF-policy analytics

This notebook compares the 100-tree fixed SMAC baseline with the two RWTHGPT
policies in dimensions 50 and 100. The initial-choice policy asks the model for
one setting before the first evaluation; the dynamic policy asks at trials 100,
250, and 500. Lower objective values are better. Incomplete jobs are reported
in the coverage table and excluded from performance rankings."""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use('seaborn-v0_8-whitegrid')
HERE = Path.cwd()
if HERE.name != '40_llm_o6':
    HERE = Path('experiments/synthaticBench/o6_multi_modal/depth_policies/40_llm_o6').resolve()
ROOT = HERE / 'results'
DIMENSIONS = (50, 100)
FIXED_SEEDS = tuple(range(6))
MODELS = {
    'gpt-5.4-mini': tuple(range(6)),
    'gpt-5.5': (0, 1, 2),
}
POLICY_SPECS = [
    ('Fixed 100 trees', 'fixed_100_trees', None, FIXED_SEEDS),
    ('Initial GPT-5.4-mini', 'initial_choice', 'gpt-5.4-mini', MODELS['gpt-5.4-mini']),
    ('Initial GPT-5.5', 'initial_choice', 'gpt-5.5', MODELS['gpt-5.5']),
    ('Dynamic GPT-5.4-mini', 'dynamic', 'gpt-5.4-mini', MODELS['gpt-5.4-mini']),
    ('Dynamic GPT-5.5', 'dynamic', 'gpt-5.5', MODELS['gpt-5.5']),
]
N_TRIALS = 1000
print(f'Results root: {ROOT}')"""))
cells.append(nbf.v4.new_markdown_cell("""## Coverage and loading

The expected grid contains 48 runs: 12 fixed, 18 initial-choice, and 18
dynamic. The loader reads the completion marker even when a trajectory is not
available yet, so this cell is also a progress monitor."""))
cells.append(nbf.v4.new_code_cell("""def model_suffix(model):
    return model.replace('.', '_').replace('-', '_') if model else None


def run_directory(dimension, kind, model, seed):
    name = kind if model is None else f'{kind}_{model_suffix(model)}'
    return ROOT / f'dimension_{dimension}' / 'benchmark_seed_52' / name / str(seed)


def failed_evaluations(run_dir):
    path = run_dir / 'runhistory.json'
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return sum(not np.isfinite(float(entry.get('cost', np.nan))) for entry in data.get('data', []))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return np.nan


expected_rows, run_rows = [], []
for dimension in DIMENSIONS:
    for label, kind, model, seeds in POLICY_SPECS:
        for seed in seeds:
            run_dir = run_directory(dimension, kind, model, seed)
            completion_path = run_dir / 'completed.json'
            trajectory_path = run_dir / 'trajectory.json'
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            payload = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else None
            status = 'complete' if payload is not None else completion.get('state', 'missing')
            row = dict(dimension=dimension, policy=label, kind=kind, model=model,
                       smac_seed=seed, status=status, trials=0,
                       failed_evaluations=failed_evaluations(run_dir), final=np.nan,
                       run_dir=str(run_dir), payload=payload)
            if payload is not None:
                row['trials'] = len(payload.get('iteration', []))
                best = payload.get('best_so_far', [])
                row['final'] = float(best[-1]) if best else np.nan
            expected_rows.append({k: row[k] for k in ('dimension', 'policy', 'smac_seed')})
            run_rows.append(row)

coverage = pd.DataFrame(run_rows)
display(coverage.drop(columns='payload').sort_values(['dimension', 'policy', 'smac_seed']))
display(coverage.groupby(['dimension', 'policy', 'status']).size().rename('runs').reset_index())
print(f'Complete trajectories: {(coverage.status == "complete").sum()} / {len(coverage)}')"""))
cells.append(nbf.v4.new_markdown_cell("""## Final-incumbent rankings

Only complete trajectories with 1,000 trials are used. For each dimension,
policies are ranked by their mean final incumbent objective across their
available SMAC seeds. The final table averages those ranks across dimensions;
it also reports how many dimensions are currently available."""))
cells.append(nbf.v4.new_code_cell("""complete = coverage[(coverage.status == 'complete') & (coverage.trials == N_TRIALS)].copy()
ranking = (complete.groupby(['dimension', 'policy'])
           .agg(average_final=('final', 'mean'), std_final=('final', 'std'), runs=('final', 'size'))
           .reset_index())
if len(ranking):
    ranking['rank'] = (ranking.groupby('dimension')['average_final']
                       .rank(method='min', ascending=True).astype(int))
    ranking = ranking.sort_values(['dimension', 'rank', 'policy'])
    display(ranking)
    average_rank = (ranking.groupby('policy')
                    .agg(average_rank=('rank', 'mean'), dimensions_ranked=('dimension', 'nunique'),
                         mean_final=('average_final', 'mean'))
                    .reset_index().sort_values(['average_rank', 'policy']))
    display(average_rank)
else:
    print('No complete 1,000-trial trajectories yet.')"""))
cells.append(nbf.v4.new_markdown_cell("""## Trajectory comparison

For every dimension, the lines show the mean best-so-far objective across
completed seeds. Shaded regions show ± one standard error. The plot updates as
the arrays finish."""))
cells.append(nbf.v4.new_code_cell("""for dimension in DIMENSIONS:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plotted = False
    for label, *_ in POLICY_SPECS:
        selected = complete[(complete.dimension == dimension) & (complete.policy == label)]
        curves = [np.asarray(row.payload['best_so_far'], dtype=float) for row in selected.itertuples()]
        if not curves or min(map(len, curves)) < 2:
            continue
        n = min(map(len, curves))
        values = np.vstack([curve[:n] for curve in curves])
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.zeros(n)
        x = np.arange(1, n + 1)
        ax.plot(x, mean, label=label, linewidth=2 if 'Dynamic' in label else 1.6)
        ax.fill_between(x, mean - se, mean + se, alpha=0.12)
        plotted = True
    ax.set(title=f'O6-Multimodal dimension {dimension}', xlabel='SMAC trial',
           ylabel='Mean best-so-far objective')
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No complete trajectories yet', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout(); plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""## LLM RF settings and token usage

The initial settings come from each run identity. Dynamic settings are read
from the callback audit, including checkpoint, chosen RF values, confidence,
and token usage."""))
cells.append(nbf.v4.new_code_cell("""decision_rows = []
for row in coverage.itertuples():
    payload = row.payload
    if payload is None:
        continue
    if row.kind == 'initial_choice':
        settings = payload.get('initial_choice', {}).get('settings', payload.get('rf_settings', {}))
        decision_rows.append(dict(dimension=row.dimension, policy=row.policy, model=row.model,
                                  smac_seed=row.smac_seed, checkpoint=0, **settings))
    elif row.kind == 'dynamic':
        decisions = payload.get('llm_policy', {}).get('decisions', {})
        for checkpoint, decision in decisions.items():
            settings = decision.get('settings', {})
            usage = decision.get('usage') or {}
            decision_rows.append(dict(dimension=row.dimension, policy=row.policy, model=row.model,
                                      smac_seed=row.smac_seed, checkpoint=int(checkpoint),
                                      actual_completed_trials=decision.get('actual_completed_trials_at_call'),
                                      input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                                      **settings))
decisions = pd.DataFrame(decision_rows)
if len(decisions):
    display(decisions.sort_values(['dimension', 'policy', 'smac_seed', 'checkpoint']))
    setting_columns = ['n_trees', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'feature_ratio']
    display(decisions.groupby(['dimension', 'policy', 'checkpoint'])[setting_columns].mean().round(3))
else:
    print('No LLM decisions have been recorded yet.')"""))
nb["cells"] = cells
nbf.write(nb, HERE / "analyze_o6_llm.ipynb")
print(HERE / "analyze_o6_llm.ipynb")

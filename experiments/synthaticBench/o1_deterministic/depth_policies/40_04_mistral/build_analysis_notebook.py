from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "adaptive-smac-synthactic-py311", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
cells = []
cells.append(nbf.v4.new_markdown_cell("""# O1 fixed-100-tree Mistral policy comparison

This notebook compares two RWTHGPT Mistral policies with the original compact
LLM policy and the 100-tree fixed-depth controls from `40_02_different_dimensions`.
Both Mistral policies fix `n_trees=100`; Mistral chooses only maximum depth,
split and leaf sizes, and feature ratio. One receives the original compact
ten-window summary, while the other receives every second completed trial with
its configuration and proposal-time SMAC diagnostics.

Lower objective values are better. The legacy experiment has no 10D compact or
100-tree fixed-depth reference data; the coverage table leaves that gap visible."""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use('seaborn-v0_8-whitegrid')
HERE = Path.cwd()
if HERE.name != '40_04_mistral':
    HERE = Path('experiments/synthaticBench/o1_deterministic/depth_policies/40_04_mistral').resolve()
NEW_ROOT = HERE / 'results'
LEGACY_ROOT = HERE.parent / '40_02_different_dimensions' / 'results'
DIMENSIONS = (10, 25, 50, 100)
SEEDS = tuple(range(5))
DEPTHS = (5, 10, 15, 20, 30)
N_TRIALS = 1000
NEW_POLICIES = {
    'Compact LLM Mistral (100 trees)': 'rwthgpt_mistral_compact_fixed_100_trees',
    'Every-second Mistral (100 trees)': 'rwthgpt_mistral_every_second_fixed_100_trees',
}
LEGACY_COMPACT = 'openai_compact_llm_rf_policy'
print(f'New results: {NEW_ROOT}')
print(f'Legacy results: {LEGACY_ROOT}')"""))
cells.append(nbf.v4.new_markdown_cell("""## Load and coverage

Only complete 1,000-trial trajectories are included in rankings and plots.
`missing` means that no trajectory exists; `running` means a completion marker
exists but the run has not yet written its trajectory."""))
cells.append(nbf.v4.new_code_cell("""def new_path(policy_name, dimension, seed):
    return NEW_ROOT / f'dimension_{dimension}' / 'benchmark_seed_40' / policy_name / str(seed)

def legacy_path(policy_name, dimension, seed):
    return LEGACY_ROOT / f'dimension_{dimension}' / 'benchmark_seed_40' / policy_name / str(seed)

specifications = [
    *[(label, name, 'mistral') for label, name in NEW_POLICIES.items()],
    ('Compact LLM original', LEGACY_COMPACT, 'legacy'),
    *[(f'Fixed depth {depth} (100 trees)', f'fixed_depth_{depth}_100_trees', 'legacy') for depth in DEPTHS],
]
runs = []
for dimension in DIMENSIONS:
    for label, name, source in specifications:
        for seed in SEEDS:
            directory = (new_path if source == 'mistral' else legacy_path)(name, dimension, seed)
            trajectory_path = directory / 'trajectory.json'
            completion_path = directory / 'completed.json'
            payload = json.loads(trajectory_path.read_text()) if trajectory_path.exists() else None
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            status = 'complete' if payload else completion.get('state', 'missing')
            best = [] if payload is None else payload.get('best_so_far', [])
            runs.append(dict(dimension=dimension, policy=label, source=source, smac_seed=seed,
                             status=status, trials=0 if payload is None else len(payload.get('iteration', [])),
                             final=np.nan if not best else float(best[-1]), payload=payload,
                             directory=str(directory)))
coverage = pd.DataFrame(runs)
display(coverage.drop(columns='payload').sort_values(['dimension', 'policy', 'smac_seed']))
display(coverage.groupby(['dimension', 'policy', 'status']).size().rename('runs').reset_index())"""))
cells.append(nbf.v4.new_markdown_cell("""## Mean optimization trajectories

Lines are mean best-so-far objective values over the completed SMAC seeds.
Shading is ± one standard error; a policy is omitted until it has a complete
trajectory."""))
cells.append(nbf.v4.new_code_cell("""complete = coverage[(coverage.status == 'complete') & (coverage.trials == N_TRIALS)].copy()
for dimension in DIMENSIONS:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    plotted = False
    for label, _, _ in specifications:
        selected = complete[(complete.dimension == dimension) & (complete.policy == label)]
        curves = [np.asarray(row.payload['best_so_far'], dtype=float) for row in selected.itertuples()]
        if not curves:
            continue
        length = min(map(len, curves))
        values = np.vstack([curve[:length] for curve in curves])
        mean = values.mean(axis=0)
        se = values.std(axis=0, ddof=1) / math.sqrt(len(values)) if len(values) > 1 else np.zeros(length)
        x = np.arange(1, length + 1)
        ax.plot(x, mean, label=label, linewidth=2.4 if 'Mistral' in label else 1.5)
        ax.fill_between(x, mean-se, mean+se, alpha=.10)
        plotted = True
    for checkpoint in (100, 250, 500): ax.axvline(checkpoint, color='0.45', linestyle=':', alpha=.45)
    ax.set(title=f'O1 deterministic objective — dimension {dimension}', xlabel='SMAC trial',
           ylabel='Mean best-so-far objective (lower is better)', xlim=(1, N_TRIALS))
    if plotted: ax.legend(ncol=2)
    else: ax.text(.5, .5, 'No complete trajectories yet', ha='center', va='center', transform=ax.transAxes)
    plt.tight_layout(); plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""## Final-incumbent ranking

Policies are ranked independently within each objective dimension by their
mean final incumbent over complete runs. The last table averages ranks only
over dimensions where a policy has data."""))
cells.append(nbf.v4.new_code_cell("""ranking = (complete.groupby(['dimension', 'policy'])
           .agg(average_final=('final', 'mean'), std_final=('final', 'std'), runs=('final', 'size'))
           .reset_index())
if len(ranking):
    ranking['rank'] = ranking.groupby('dimension')['average_final'].rank(method='min', ascending=True).astype(int)
    ranking = ranking.sort_values(['dimension', 'rank', 'policy'])
    display(ranking)
    display(ranking.groupby('policy').agg(average_rank=('rank', 'mean'), dimensions_ranked=('dimension', 'nunique'))
            .reset_index().sort_values(['average_rank', 'policy']))
else:
    print('No complete trajectories yet.')"""))
cells.append(nbf.v4.new_markdown_cell("""## Final incumbent boxplots

Each box shows the distribution of final incumbent performance across SMAC
seeds for one dimension and policy. Points are the individual seeds; lower is
better. Policies without complete trajectories are omitted automatically."""))
cells.append(nbf.v4.new_code_cell("""for dimension in DIMENSIONS:
    part = complete[complete.dimension == dimension]
    available = [label for label, _, _ in specifications if (part.policy == label).any()]
    if not available:
        print(f'No complete trajectories for dimension {dimension}.')
        continue
    values = [part.loc[part.policy == label, 'final'].to_numpy(dtype=float) for label in available]
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(available)), 5.5))
    ax.boxplot(values, tick_labels=available, showfliers=False)
    rng = np.random.default_rng(40 + dimension)
    for index, sample in enumerate(values, 1):
        jitter = rng.uniform(-0.08, 0.08, size=len(sample))
        ax.scatter(index + jitter, sample, s=38, alpha=0.85, zorder=3)
    ax.set_title(f'Final incumbent performance — dimension {dimension}')
    ax.set_ylabel('Final best objective (lower is better)')
    ax.tick_params(axis='x', rotation=28)
    plt.tight_layout(); plt.show()"""))
cells.append(nbf.v4.new_markdown_cell("""## Mistral decisions, prompt size, and retries

The decision table audits that `n_trees` remained fixed at 100. It also shows
the actual checkpoint trigger, input-token estimate, and any 15-second
rate-limit retries. The every-second policy should have substantially larger
inputs than the compact-summary policy."""))
cells.append(nbf.v4.new_code_cell("""decision_rows = []
for row in coverage[coverage.source == 'mistral'].itertuples():
    payload = row.payload
    if payload is None:
        continue
    for checkpoint, decision in payload.get('llm_policy', {}).get('decisions', {}).items():
        settings = decision.get('settings', {})
        metadata = decision.get('usage') or {}
        request_path = Path(row.directory) / 'llm_requests' / f'checkpoint_{int(checkpoint):04d}' / 'openai_response_metadata.json'
        request_metadata = json.loads(request_path.read_text()) if request_path.exists() else {}
        decision_rows.append(dict(dimension=row.dimension, policy=row.policy, smac_seed=row.smac_seed,
            checkpoint=int(checkpoint), actual_completed_trials=decision.get('actual_completed_trials_at_call'),
            n_trees=settings.get('n_trees'), max_depth=settings.get('max_depth'),
            min_samples_split=settings.get('min_samples_split'), min_samples_leaf=settings.get('min_samples_leaf'),
            feature_ratio=settings.get('feature_ratio'), confidence=decision.get('confidence'),
            prompt_token_estimate=request_metadata.get('input_token_estimate'),
            rate_limit_retries=request_metadata.get('rate_limit_retries'),
            input_tokens=metadata.get('prompt_tokens'), output_tokens=metadata.get('completion_tokens'),
            reason=decision.get('reason')))
decisions = pd.DataFrame(decision_rows)
if len(decisions):
    assert decisions.n_trees.eq(100).all(), 'A model decision changed n_trees.'
    with pd.option_context('display.max_colwidth', 240, 'display.max_rows', None):
        display(decisions.sort_values(['dimension', 'policy', 'smac_seed', 'checkpoint']))
    display(decisions.groupby(['policy', 'checkpoint'])[['prompt_token_estimate', 'rate_limit_retries', 'input_tokens', 'output_tokens']].mean().round(1))
else:
    print('No completed Mistral decisions yet.')"""))
nb["cells"] = cells
nbf.write(nb, HERE / "analyze_mistral.ipynb")
print(HERE / "analyze_mistral.ipynb")

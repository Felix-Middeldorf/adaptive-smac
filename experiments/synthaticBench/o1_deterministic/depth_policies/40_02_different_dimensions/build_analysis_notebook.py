from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "adaptive-smac-synthactic-py311", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
c = []
c.append(nbf.v4.new_markdown_cell("""# Compact LLM policy across dimensions

This notebook compares the compact LLM-selected SMAC random-forest policy with 100-tree fixed controls and ten frozen fixed policies sampled from LLM decisions. The sampled policies are evaluated across objective dimensions 2, 5, 25, 50, and 100. The earlier 10-tree fixed runs are deliberately excluded. Lower objective values are better."""))
c.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use('seaborn-v0_8-whitegrid')
HERE = Path.cwd()
if HERE.name != '40_02_different_dimensions':
    HERE = Path('experiments/synthaticBench/o1_deterministic/depth_policies/40_02_different_dimensions').resolve()
ROOT = HERE / 'results'
DIMENSIONS = (2, 5, 25, 50, 100)
SMAC_SEEDS = tuple(range(5))
DEPTHS = (5, 10, 15, 20, 30)
N_TRIALS = 1000
FIXED_POLICIES = [f'Fixed depth {d} (100 trees)' for d in DEPTHS]
SAMPLED_POLICIES = {
    'Sampled LLM d2 @100': 'fixed_llm_config_source_d2_cp100',
    'Sampled LLM d2 @500': 'fixed_llm_config_source_d2_cp500',
    'Sampled LLM d5 @100': 'fixed_llm_config_source_d5_cp100',
    'Sampled LLM d5 @500': 'fixed_llm_config_source_d5_cp500',
    'Sampled LLM d25 @100': 'fixed_llm_config_source_d25_cp100',
    'Sampled LLM d25 @500': 'fixed_llm_config_source_d25_cp500',
    'Sampled LLM d50 @100': 'fixed_llm_config_source_d50_cp100',
    'Sampled LLM d50 @500': 'fixed_llm_config_source_d50_cp500',
    'Sampled LLM d100 @100': 'fixed_llm_config_source_d100_cp100',
    'Sampled LLM d100 @500': 'fixed_llm_config_source_d100_cp500',
}
POLICIES = ['Compact LLM'] + FIXED_POLICIES + list(SAMPLED_POLICIES)
print(ROOT)"""))
c.append(nbf.v4.new_markdown_cell("""## Load and coverage

Only completed trajectories are loaded. The coverage table makes missing or incomplete array tasks visible before interpreting the comparisons."""))
c.append(nbf.v4.new_code_cell("""def path_for(dimension, policy, seed):
    if policy == 'Compact LLM':
        name = 'openai_compact_llm_rf_policy'
    elif policy in SAMPLED_POLICIES:
        name = SAMPLED_POLICIES[policy]
    else:
        depth = DEPTHS[FIXED_POLICIES.index(policy)]
        name = f'fixed_depth_{depth}_100_trees'
    return ROOT / f'dimension_{dimension}' / 'benchmark_seed_40' / name / str(seed) / 'trajectory.json'

runs = []
for dimension in DIMENSIONS:
    for policy in POLICIES:
        for seed in SMAC_SEEDS:
            path = path_for(dimension, policy, seed)
            if not path.exists():
                continue
            p = json.loads(path.read_text())
            runs.append(dict(dimension=dimension, policy=policy, smac_seed=seed,
                             iteration=np.asarray(p['iteration']), best=np.asarray(p['best_so_far']),
                             final=float(p['best_so_far'][-1]), payload=p, path=path))
coverage = pd.DataFrame([dict(dimension=r['dimension'], policy=r['policy'], smac_seed=r['smac_seed'],
                              trials=len(r['iteration']), final=r['final']) for r in runs])
expected = pd.MultiIndex.from_product([DIMENSIONS, POLICIES], names=['dimension', 'policy']).to_frame(index=False)
summary = (coverage.groupby(['dimension', 'policy']).agg(runs=('smac_seed','nunique'),
           minimum_trials=('trials','min'), maximum_trials=('trials','max')).reset_index()
           if len(coverage) else pd.DataFrame(columns=['dimension','policy','runs','minimum_trials','maximum_trials']))
display(expected.merge(summary, how='left').fillna(0))"""))
c.append(nbf.v4.new_markdown_cell("""## 1. Compact-policy trajectories

One plot per dimension shows the five SMAC seeds. Downward steps are improvements; dotted lines mark the three LLM decisions at trials 100, 250, and 500."""))
c.append(nbf.v4.new_code_cell("""for dimension in DIMENSIONS:
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in runs:
        if r['dimension'] == dimension and r['policy'] == 'Compact LLM':
            ax.step(r['iteration'], r['best'], where='post', label=f\"seed {r['smac_seed']}\")
    for checkpoint in (100, 250, 500): ax.axvline(checkpoint, color='0.4', ls=':', alpha=.5)
    ax.set(title=f'Compact LLM — dimension {dimension}', xlabel='SMAC trial', ylabel='Best-so-far objective', xlim=(1, N_TRIALS))
    ax.legend(); plt.tight_layout(); plt.show()"""))
c.append(nbf.v4.new_markdown_cell("""## 2. Compact policy versus 100-tree fixed depths

Only the new 100-tree fixed-depth runs are used here; the previous 10-tree controls are not loaded. Lines are mean best-so-far values across SMAC seeds; shading is a two-sided 95% Student-t confidence interval for the mean. It reflects seed uncertainty, not individual-run prediction uncertainty."""))
c.append(nbf.v4.new_code_cell("""T95 = {2:12.706, 3:4.303, 4:3.182, 5:2.776}
def mean_ci(curves):
    a = np.vstack(curves); mean = a.mean(0)
    half = T95[len(a)] * a.std(0, ddof=1) / math.sqrt(len(a)) if len(a)>1 else np.zeros_like(mean)
    return mean, mean-half, mean+half

for dimension in DIMENSIONS:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for policy in POLICIES:
        selected = sorted([r for r in runs if r['dimension']==dimension and r['policy']==policy], key=lambda r:r['smac_seed'])
        if not selected or min(len(r['best']) for r in selected) < N_TRIALS: continue
        mean, lo, hi = mean_ci([r['best'][:N_TRIALS] for r in selected])
        x = np.arange(1, N_TRIALS+1); ax.plot(x, mean, label=policy, lw=2.3 if policy=='Compact LLM' else 1.5)
        ax.fill_between(x, lo, hi, alpha=.1)
    ax.set(title=f'Mean trajectory and 95% CI — dimension {dimension}', xlabel='SMAC trial', ylabel='Mean best-so-far objective', xlim=(1,N_TRIALS))
    ax.legend(ncol=2); plt.tight_layout(); plt.show()"""))
c.append(nbf.v4.new_markdown_cell("""## 3. Final-incumbent comparison

Every point is one SMAC seed's best value after trial 1,000. The boxplots expose both typical performance and seed variability; the table ranks policies by their mean final value within each dimension."""))
c.append(nbf.v4.new_code_cell("""complete = coverage[coverage.trials == N_TRIALS].copy()
for dimension in DIMENSIONS:
    part = complete[complete.dimension == dimension]
    values = [part.loc[part.policy==p, 'final'].values for p in POLICIES]
    if not all(len(v) for v in values): continue
    fig, ax = plt.subplots(figsize=(11,5)); ax.boxplot(values, tick_labels=POLICIES, showfliers=False)
    rng=np.random.default_rng(40+dimension)
    for i,v in enumerate(values,1): ax.scatter(i+rng.uniform(-.06,.06,len(v)),v,s=35,zorder=3)
    ax.set(title=f'Final incumbents — dimension {dimension}', ylabel='Final best objective (lower is better)')
    ax.tick_params(axis='x', rotation=25); plt.tight_layout(); plt.show()

ranking=(complete.groupby(['dimension','policy']).final.agg(['mean','std','median','min','max']).reset_index()
         .sort_values(['dimension','mean']))
display(ranking)"""))
c.append(nbf.v4.new_markdown_cell("""## 4. Rankings across policies

For each target dimension, policies are ranked by their average final incumbent objective across complete SMAC runs. Lower average objective is better, so rank 1 is best. The final table averages each policy's rank across dimensions; sampled policies currently have one evaluation seed unless additional seeds are submitted."""))
c.append(nbf.v4.new_code_cell("""rank_by_dimension = (
    complete.groupby(['dimension', 'policy'])
    .agg(average_final=('final', 'mean'), runs=('final', 'size'))
    .reset_index()
)
rank_by_dimension['rank'] = (
    rank_by_dimension.groupby('dimension')['average_final']
    .rank(method='min', ascending=True)
    .astype('Int64')
)
rank_by_dimension = rank_by_dimension.sort_values(['dimension', 'rank', 'policy'])
display(rank_by_dimension)

average_rank = (
    rank_by_dimension.groupby('policy')
    .agg(average_rank=('rank', 'mean'), dimensions_ranked=('dimension', 'nunique'))
    .reset_index()
    .sort_values(['average_rank', 'policy'])
)
display(average_rank)"""))
c.append(nbf.v4.new_markdown_cell("""## 5. LLM choices by dimension and checkpoint

This table reports the exact RF settings returned at each decision. Frequencies reveal whether the model reacts differently to objective dimension or repeatedly collapses to the same choice. Token usage is included to make API cost auditable."""))
c.append(nbf.v4.new_code_cell("""decisions=[]
for r in runs:
    if r['policy'] != 'Compact LLM': continue
    rows=r['payload'].get('llm_policy',{}).get('decisions',{})
    for checkpoint, row in rows.items():
        chosen=row.get('settings', {}); usage=row.get('usage') or {}
        decisions.append(dict(dimension=r['dimension'], smac_seed=r['smac_seed'], checkpoint=int(checkpoint),
            actual_completed_trials=row.get('actual_completed_trials_at_call'),
            **{k:chosen.get(k) for k in ['n_trees','max_depth','min_samples_split','min_samples_leaf','feature_ratio']},
            confidence=row.get('confidence'), input_tokens=usage.get('input_tokens'),
            output_tokens=usage.get('output_tokens'), reason=row.get('reason')))
decision_df=pd.DataFrame(decisions)
display(decision_df.sort_values(['dimension','smac_seed','checkpoint']) if len(decision_df) else decision_df)
if len(decision_df):
    display(decision_df.groupby(['dimension','checkpoint','n_trees','max_depth','min_samples_split','min_samples_leaf','feature_ratio']).size().rename('count').reset_index().sort_values(['dimension','checkpoint','count'],ascending=[True,True,False]))
    display(decision_df.groupby(['dimension','checkpoint'])[['input_tokens','output_tokens']].mean().round(1).rename(columns=lambda x:'mean_'+x))"""))
c.append(nbf.v4.new_markdown_cell("""### 5a. Complete 100D decisions and reasons

This dedicated table contains all three decisions for every 100D SMAC seed. `checkpoint` is the requested threshold; `actual_completed_trials` is when the callback was able to make the API call. The five RF columns are the settings applied to the next model-training phase. The reason is shown without truncation."""))
c.append(nbf.v4.new_code_cell("""decision_columns = [
    'smac_seed', 'checkpoint', 'actual_completed_trials',
    'n_trees', 'max_depth', 'min_samples_split', 'min_samples_leaf',
    'feature_ratio', 'confidence', 'input_tokens', 'output_tokens', 'reason',
]
decisions_100d = (
    decision_df.loc[decision_df['dimension'].eq(100), decision_columns]
    .sort_values(['smac_seed', 'checkpoint'])
    .reset_index(drop=True)
)
assert len(decisions_100d) == len(SMAC_SEEDS) * 3, (
    f'Expected 15 decisions for 100D, found {len(decisions_100d)}.'
)
with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):
    display(decisions_100d.style.set_properties(
        subset=['reason'], **{'text-align': 'left', 'white-space': 'pre-wrap'}
    ))"""))
nb["cells"] = c
nbf.write(nb, HERE / "analyze_different_dimensions.ipynb")
print(HERE / "analyze_different_dimensions.ipynb")

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {"kernelspec": {"display_name": "adaptive-smac-synthactic-py311", "language": "python", "name": "python3"}}
c = []
c.append(nbf.v4.new_markdown_cell("""# Depth-only compact LLM ablation

This notebook compares the depth-only compact LLM policy with fixed depths 5, 10, 15, 20, and 30 at dimensions 25, 50, and 100. Every policy uses 100 trees, split size 2, leaf size 1, feature ratio 5/6, PCA 4, ten deterministic instances, benchmark seed 40, and five SMAC seeds. Lower values and gaps are better."""))
c.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
plt.style.use('seaborn-v0_8-whitegrid')
HERE=Path.cwd()
if HERE.name!='40_03_depth': HERE=Path('experiments/synthaticBench/o1_deterministic/depth_policies/40_03_depth').resolve()
ROOT=HERE/'results'; DIMENSIONS=(25,50,100); SEEDS=tuple(range(5)); DEPTHS=(5,10,15,20,30); N=1000
POLICIES=['Depth-only LLM']+[f'Fixed depth {d}' for d in DEPTHS]"""))
c.append(nbf.v4.new_markdown_cell("""## Coverage

This table prevents incomplete array tasks from being silently mixed into the comparison."""))
c.append(nbf.v4.new_code_cell("""def run_path(d,p,s):
    name='openai_compact_depth_only_policy' if p=='Depth-only LLM' else 'fixed_depth_'+p.rsplit(' ',1)[-1]
    return ROOT/f'dimension_{d}'/'benchmark_seed_40'/name/str(s)/'trajectory.json'
runs=[]
for d in DIMENSIONS:
 for p in POLICIES:
  for s in SEEDS:
   path=run_path(d,p,s)
   if not path.exists(): continue
   x=json.loads(path.read_text()); best=np.asarray(x['best_so_far']); f=float(x['f_min'])
   runs.append(dict(dimension=d,policy=p,seed=s,iteration=np.asarray(x['iteration']),best=best,
                    relative_gap=(best-f)/abs(f),final=best[-1],final_relative_gap=(best[-1]-f)/abs(f),payload=x,path=path))
coverage=pd.DataFrame([dict(dimension=r['dimension'],policy=r['policy'],seed=r['seed'],trials=len(r['best'])) for r in runs])
expected=pd.MultiIndex.from_product([DIMENSIONS,POLICIES],names=['dimension','policy']).to_frame(index=False)
summary=coverage.groupby(['dimension','policy']).agg(runs=('seed','nunique'),min_trials=('trials','min'),max_trials=('trials','max')).reset_index() if len(coverage) else pd.DataFrame()
display(expected.merge(summary,how='left').fillna(0))"""))
c.append(nbf.v4.new_markdown_cell("""## LLM trajectories by dimension

Each line is one SMAC seed. Dotted lines mark depth decisions at trials 100, 250, and 500."""))
c.append(nbf.v4.new_code_cell("""for d in DIMENSIONS:
 fig,ax=plt.subplots(figsize=(10,5))
 for r in runs:
  if r['dimension']==d and r['policy']=='Depth-only LLM': ax.step(r['iteration'],r['relative_gap'],where='post',label=f\"seed {r['seed']}\")
 for cp in (100,250,500): ax.axvline(cp,color='.4',ls=':',alpha=.5)
 ax.set(title=f'Depth-only LLM — {d}D',xlabel='SMAC trial',ylabel='Relative optimality gap',xlim=(1,N)); ax.legend(); plt.tight_layout(); plt.show()"""))
c.append(nbf.v4.new_markdown_cell("""## Mean trajectories and 95% confidence intervals

Lines are arithmetic means across the five matched SMAC seeds. Shading is a two-sided Student-t 95% confidence interval for the mean, not the spread of individual runs."""))
c.append(nbf.v4.new_code_cell("""T95={2:12.706,3:4.303,4:3.182,5:2.776}
def ci(curves):
 a=np.vstack(curves); m=a.mean(0); h=T95[len(a)]*a.std(0,ddof=1)/math.sqrt(len(a)) if len(a)>1 else np.zeros_like(m); return m,m-h,m+h
for d in DIMENSIONS:
 fig,ax=plt.subplots(figsize=(11,5.5))
 for p in POLICIES:
  q=sorted([r for r in runs if r['dimension']==d and r['policy']==p and len(r['best'])==N],key=lambda r:r['seed'])
  if not q: continue
  m,lo,hi=ci([r['relative_gap'] for r in q]); x=np.arange(1,N+1); ax.plot(x,m,label=p,lw=2.3 if p=='Depth-only LLM' else 1.5); ax.fill_between(x,lo,hi,alpha=.1)
 ax.set(title=f'Mean relative-gap trajectory — {d}D',xlabel='SMAC trial',ylabel='Mean relative optimality gap',xlim=(1,N)); ax.legend(ncol=2); plt.tight_layout(); plt.show()"""))
c.append(nbf.v4.new_markdown_cell("""## Final performance

Every point is one seed's final incumbent. The ranking table compares mean final gap and whole-run normalized area under the gap curve."""))
c.append(nbf.v4.new_code_cell("""finals=pd.DataFrame([dict(dimension=r['dimension'],policy=r['policy'],seed=r['seed'],final=r['final'],relative_gap=r['final_relative_gap'],normalized_auc=r['relative_gap'].mean()) for r in runs if len(r['best'])==N])
for d in DIMENSIONS:
 part=finals[finals.dimension==d]; values=[part.loc[part.policy==p,'relative_gap'].values for p in POLICIES]
 if not all(len(v) for v in values): continue
 fig,ax=plt.subplots(figsize=(11,5)); ax.boxplot(values,labels=POLICIES,showfliers=False)
 rng=np.random.default_rng(40+d)
 for i,v in enumerate(values,1): ax.scatter(i+rng.uniform(-.06,.06,len(v)),v,s=35,zorder=3)
 ax.set(title=f'Final relative gap — {d}D',ylabel='Relative optimality gap'); ax.tick_params(axis='x',rotation=25); plt.tight_layout(); plt.show()
display(finals.groupby(['dimension','policy']).agg(runs=('seed','count'),mean_gap=('relative_gap','mean'),sd_gap=('relative_gap','std'),median_gap=('relative_gap','median'),mean_normalized_auc=('normalized_auc','mean')).reset_index().sort_values(['dimension','mean_gap']))"""))
c.append(nbf.v4.new_markdown_cell("""## Selected depths

The table shows the only hyperparameter returned by the LLM at each checkpoint. The audit assertion verifies that all persisted transitions retained the immutable RF settings."""))
c.append(nbf.v4.new_code_cell("""rows=[]
for r in runs:
 if r['policy']!='Depth-only LLM': continue
 audit=r['payload'].get('llm_policy',{})
 for cp,v in audit.get('decisions',{}).items():
  st=v['settings']; assert st['n_trees']==100 and st['min_samples_split']==2 and st['min_samples_leaf']==1 and np.isclose(st['feature_ratio'],5/6)
  usage=v.get('usage') or {}
  rows.append(dict(dimension=r['dimension'],seed=r['seed'],checkpoint=int(cp),max_depth=st['max_depth'],confidence=v['confidence'],input_tokens=usage.get('input_tokens'),output_tokens=usage.get('output_tokens'),reason=v['reason']))
decisions=pd.DataFrame(rows)
display(decisions.sort_values(['dimension','seed','checkpoint']) if len(decisions) else decisions)
if len(decisions): display(decisions.groupby(['dimension','checkpoint']).max_depth.agg(['mean','std','min','max']))"""))
nb["cells"] = c
nbf.write(nb, HERE / "analyze_depth_only.ipynb")
print(HERE / "analyze_depth_only.ipynb")

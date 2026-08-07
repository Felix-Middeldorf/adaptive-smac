from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "analyze_llm_policy.ipynb"

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

cells.append(
    nbf.v4.new_markdown_cell(
        """# LLM-selected SMAC random-forest policy

This notebook compares the original raw-history OpenAI/LLM RF policy, the compact-summary OpenAI/LLM RF policy, and fixed RF-depth controls on the two O1 deterministic benchmark landscapes used by this experiment (benchmark seeds 40 and 42).

The optimization objective is minimized. Every trajectory uses the stored `best_so_far` value: the best full 10-instance objective value discovered by that trial. This is not the raw cost from one instance evaluation.

For a fair comparison, all analyses use the five SMAC seeds shared by both LLM experiments and controls: seeds 0–4. The fixed controls come from `09_big_experiment` and differ only in their fixed maximum RF depth (5, 10, 15, or 20)."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.max_colwidth', 120)


def locate_experiment_dir():
    relative = Path('experiments/synthaticBench/o1_deterministic/depth_policies/40_llm_chooses')
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        direct = candidate if candidate.name == '40_llm_chooses' else candidate / relative
        if (direct / 'smac_output').is_dir():
            return direct.resolve()
    raise FileNotFoundError('Could not locate 40_llm_chooses/smac_output from the current directory.')


HERE = locate_experiment_dir()
FIXED_ROOT = HERE.parent / '09_big_experiment' / 'smac_output' / 'fixed'
BENCHMARK_SEEDS = (40, 42)
SMAC_SEEDS = tuple(range(5))
DEPTHS = (5, 10, 15, 20)
N_TRIALS = 1000

print(f'LLM results:   {HERE / "smac_output"}')
print(f'Fixed results: {FIXED_ROOT}')"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Load and validate trajectories

The table confirms which files are included and checks that every run contains all 1,000 requested trials. A run is only admitted if its benchmark seed, SMAC seed, and stored trajectory agree with the path."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """def load_trajectory(path, benchmark_seed, smac_seed, policy):
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if int(payload['benchmark_seed']) != benchmark_seed:
        raise ValueError(f'Benchmark-seed mismatch in {path}')
    if int(payload['smac_seed']) != smac_seed:
        raise ValueError(f'SMAC-seed mismatch in {path}')
    iterations = np.asarray(payload['iteration'], dtype=int)
    best = np.asarray(payload['best_so_far'], dtype=float)
    if len(iterations) != len(best):
        raise ValueError(f'Unequal iteration and best_so_far lengths in {path}')
    return {
        'benchmark_seed': benchmark_seed,
        'smac_seed': smac_seed,
        'policy': policy,
        'iteration': iterations,
        'best_so_far': best,
        'payload': payload,
        'path': path,
    }


runs = []
for benchmark_seed in BENCHMARK_SEEDS:
    for smac_seed in SMAC_SEEDS:
        llm_path = (HERE / 'smac_output' / f'benchmark_seed_{benchmark_seed}' /
                    'openai_llm_rf_policy' / str(smac_seed) / 'trajectory.json')
        runs.append(load_trajectory(llm_path, benchmark_seed, smac_seed, 'Original LLM'))
        compact_path = (HERE / 'smac_output' / f'benchmark_seed_{benchmark_seed}' /
                        'openai_compact_llm_rf_policy' / str(smac_seed) / 'trajectory.json')
        runs.append(load_trajectory(compact_path, benchmark_seed, smac_seed, 'Compact LLM'))
        for depth in DEPTHS:
            fixed_path = (FIXED_ROOT / f'benchmark_seed_{benchmark_seed}' /
                          f'fixed_depth_{depth}' / str(smac_seed) / 'trajectory.json')
            runs.append(load_trajectory(fixed_path, benchmark_seed, smac_seed, f'Fixed depth {depth}'))

coverage = pd.DataFrame([
    {
        'benchmark_seed': r['benchmark_seed'],
        'policy': r['policy'],
        'smac_seed': r['smac_seed'],
        'trials': len(r['iteration']),
        'final_best_cost': r['best_so_far'][-1],
    }
    for r in runs
])

assert len(runs) == len(BENCHMARK_SEEDS) * len(SMAC_SEEDS) * (2 + len(DEPTHS))
assert coverage['trials'].eq(N_TRIALS).all(), 'At least one trajectory is incomplete.'
display(coverage.groupby(['benchmark_seed', 'policy'], sort=False).agg(
    runs=('smac_seed', 'nunique'),
    minimum_trials=('trials', 'min'),
    maximum_trials=('trials', 'max'),
).reset_index())"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1. Original LLM-policy trajectories by benchmark seed

Each plot fixes one benchmark landscape and shows one original raw-history LLM-policy line per SMAC seed. A downward step means that SMAC discovered a configuration with a better full 10-instance objective value. Differences between lines show sensitivity to SMAC's random seed."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """for benchmark_seed in BENCHMARK_SEEDS:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    selected = [r for r in runs if r['benchmark_seed'] == benchmark_seed and r['policy'] == 'Original LLM']
    for run in selected:
        ax.step(run['iteration'], run['best_so_far'], where='post',
                label=f\"SMAC seed {run['smac_seed']}\", linewidth=1.7)
    for checkpoint in (100, 250, 500):
        ax.axvline(checkpoint, color='0.4', linestyle=':', alpha=0.45)
    ax.set(title=f'Original LLM policy — benchmark seed {benchmark_seed}',
           xlabel='SMAC trial number', ylabel='Best-so-far cost (lower is better)',
           xlim=(1, N_TRIALS))
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 1b. Compact LLM-policy trajectories by benchmark seed

These are the exact counterparts of the preceding plots for the compact-summary LLM policy. Each figure shows all five SMAC seeds for one benchmark landscape, with the LLM checkpoints marked at trials 100, 250, and 500."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """for benchmark_seed in BENCHMARK_SEEDS:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    selected = [r for r in runs if r['benchmark_seed'] == benchmark_seed and r['policy'] == 'Compact LLM']
    for run in selected:
        ax.step(run['iteration'], run['best_so_far'], where='post',
                label=f\"SMAC seed {run['smac_seed']}\", linewidth=1.7)
    for checkpoint in (100, 250, 500):
        ax.axvline(checkpoint, color='0.4', linestyle=':', alpha=0.45)
    ax.set(title=f'Compact LLM policy — benchmark seed {benchmark_seed}',
           xlabel='SMAC trial number', ylabel='Best-so-far cost (lower is better)',
           xlim=(1, N_TRIALS))
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 2. Original and compact LLM policies versus fixed depths, with confidence intervals

For each benchmark landscape, the solid line is the arithmetic mean best-so-far cost at every trial across the same five SMAC seeds (0–4). The shaded region is a two-sided 95% Student-t confidence interval for that mean. With only five seeds, these intervals can be wide; they describe uncertainty in the mean across SMAC seeds, not the spread of individual runs.

The vertical dotted lines mark both LLM policies' decision checkpoints. Fixed policies do not change at those lines."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """# Two-sided Student-t 97.5% critical values indexed by sample size.
T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
             6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_and_ci(curves):
    matrix = np.vstack(curves)
    n = matrix.shape[0]
    mean = matrix.mean(axis=0)
    if n < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    half_width = T_CRIT_95.get(n, 1.96) * matrix.std(axis=0, ddof=1) / math.sqrt(n)
    return mean, mean - half_width, mean + half_width


POLICY_ORDER = ['Original LLM', 'Compact LLM'] + [f'Fixed depth {depth}' for depth in DEPTHS]
COLORS = dict(zip(POLICY_ORDER, ['black', '#B279A2', '#4C78A8', '#F58518', '#54A24B', '#E45756']))

for benchmark_seed in BENCHMARK_SEEDS:
    fig, ax = plt.subplots(figsize=(12, 6))
    for policy in POLICY_ORDER:
        selected = sorted(
            (r for r in runs if r['benchmark_seed'] == benchmark_seed and r['policy'] == policy),
            key=lambda r: r['smac_seed'],
        )
        mean, lower, upper = mean_and_ci([r['best_so_far'] for r in selected])
        x = selected[0]['iteration']
        width = 2.5 if policy in ('Original LLM', 'Compact LLM') else 1.7
        ax.plot(x, mean, label=policy, color=COLORS[policy], linewidth=width)
        ax.fill_between(x, lower, upper, color=COLORS[policy], alpha=0.12)
    for checkpoint in (100, 250, 500):
        ax.axvline(checkpoint, color='0.4', linestyle=':', alpha=0.4)
    ax.set(title=f'Mean trajectory and 95% CI — benchmark seed {benchmark_seed}',
           xlabel='SMAC trial number', ylabel='Mean best-so-far cost (lower is better)',
           xlim=(1, N_TRIALS))
    ax.legend(ncol=2)
    plt.tight_layout()
    plt.show()"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 3. Final-incumbent comparison

Each boxplot summarizes the five final best-so-far costs at trial 1,000. Every point is one SMAC seed and is shown explicitly because five samples are too few for a box alone to communicate the data. Lower values are better. The accompanying table reports the mean, standard deviation, median, minimum, and maximum."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """finals = coverage.copy()
finals['policy'] = pd.Categorical(finals['policy'], POLICY_ORDER, ordered=True)

for benchmark_seed in BENCHMARK_SEEDS:
    part = finals[finals['benchmark_seed'] == benchmark_seed].sort_values('policy')
    values = [part.loc[part['policy'] == policy, 'final_best_cost'].to_numpy() for policy in POLICY_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    boxes = ax.boxplot(values, tick_labels=POLICY_ORDER, patch_artist=True, showfliers=False)
    for box, policy in zip(boxes['boxes'], POLICY_ORDER):
        box.set_facecolor(COLORS[policy])
        box.set_alpha(0.35)
    rng = np.random.default_rng(benchmark_seed)
    for position, (policy, vals) in enumerate(zip(POLICY_ORDER, values), start=1):
        jitter = rng.uniform(-0.055, 0.055, len(vals))
        ax.scatter(position + jitter, vals, color=COLORS[policy], edgecolor='white',
                   linewidth=0.5, s=42, zorder=3)
    ax.set(title=f'Final incumbent at trial {N_TRIALS} — benchmark seed {benchmark_seed}',
           xlabel='SMAC RF policy', ylabel='Final best-so-far cost (lower is better)')
    ax.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    plt.show()

summary = (finals.groupby(['benchmark_seed', 'policy'], observed=True)['final_best_cost']
           .agg(['count', 'mean', 'std', 'median', 'min', 'max']).reset_index())
display(summary.style.format({column: '{:,.3f}' for column in ['mean', 'std', 'median', 'min', 'max']}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## 4. Hyperparameters selected at each stage

Rows below show the RF settings actually active during each trial interval for both LLM policies. Trials 1–100 use SMAC's configured default RF settings. The response obtained at checkpoint 100 applies to trials 101–250, the checkpoint-250 response applies to trials 251–500, and the checkpoint-500 response applies to trials 501–1,000.

`feature_ratio` is the fraction of configuration dimensions considered at a split. `confidence` is the LLM's self-reported confidence, not a statistically calibrated probability."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """STAGES = [
    ('Trials 1–100 (default)', None),
    ('Trials 101–250', '100'),
    ('Trials 251–500', '250'),
    ('Trials 501–1000', '500'),
]

selection_rows = []
for run in (r for r in runs if r['policy'] in ('Original LLM', 'Compact LLM')):
    payload = run['payload']
    decisions = payload['llm_policy']['decisions']
    for stage, checkpoint in STAGES:
        if checkpoint is None:
            settings = payload['default_rf_settings']
            confidence = np.nan
            reason = 'SMAC RF defaults used before the first LLM call.'
        else:
            decision = decisions[checkpoint]
            settings = decision['settings']
            confidence = decision.get('confidence', np.nan)
            reason = decision.get('reason', '')
        selection_rows.append({
            'benchmark_seed': run['benchmark_seed'],
            'smac_seed': run['smac_seed'],
            'policy': run['policy'],
            'stage': stage,
            'n_trees': settings['n_trees'],
            'max_depth': settings['max_depth'],
            'min_samples_split': settings['min_samples_split'],
            'min_samples_leaf': settings['min_samples_leaf'],
            'feature_ratio': settings['feature_ratio'],
            'confidence': confidence,
            'reason': reason,
        })

selections = pd.DataFrame(selection_rows)
stage_order = [stage for stage, _ in STAGES]
selections['stage'] = pd.Categorical(selections['stage'], stage_order, ordered=True)
display(selections.sort_values(['benchmark_seed', 'policy', 'smac_seed', 'stage']).style.format({
    'feature_ratio': '{:.3f}', 'confidence': lambda value: '' if pd.isna(value) else f'{value:.2f}'
}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """### Selection frequencies

This compact table counts how often each complete RF setting was selected in a stage across the five SMAC seeds, separately for each LLM policy and benchmark landscape. It makes convergence on the same setting—and deviations from it—easy to see."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """setting_columns = ['n_trees', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'feature_ratio']
frequencies = (selections.groupby(['benchmark_seed', 'policy', 'stage', *setting_columns], observed=True)
               .size().rename('number_of_smac_seeds').reset_index()
               .sort_values(['benchmark_seed', 'policy', 'stage', 'number_of_smac_seeds'],
                            ascending=[True, True, True, False]))
display(frequencies.style.format({'feature_ratio': '{:.3f}'}))"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """### Final ranking and paired differences

The last table ranks policies by mean final cost for each benchmark. The two `mean_cost_minus_*` columns use matched SMAC seeds. Because lower is better, a positive value favors the named LLM baseline and a negative value favors the policy in that row. The paired-win columns count how many of five seeds the row policy beats each LLM baseline."""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """comparison_rows = []
for benchmark_seed in BENCHMARK_SEEDS:
    wide = (finals[finals['benchmark_seed'] == benchmark_seed]
            .pivot(index='smac_seed', columns='policy', values='final_best_cost'))
    for policy in POLICY_ORDER:
        values = wide[policy]
        comparison_rows.append({
            'benchmark_seed': benchmark_seed,
            'policy': policy,
            'mean_final_cost': values.mean(),
            'median_final_cost': values.median(),
            'mean_cost_minus_original_llm': (values - wide['Original LLM']).mean(),
            'mean_cost_minus_compact_llm': (values - wide['Compact LLM']).mean(),
            'paired_wins_over_original_llm': np.nan if policy == 'Original LLM' else int((values < wide['Original LLM']).sum()),
            'paired_wins_over_compact_llm': np.nan if policy == 'Compact LLM' else int((values < wide['Compact LLM']).sum()),
        })

ranking = pd.DataFrame(comparison_rows)
ranking['rank_by_mean'] = ranking.groupby('benchmark_seed')['mean_final_cost'].rank(method='min')
ranking = ranking.sort_values(['benchmark_seed', 'rank_by_mean'])
display(ranking.style.format({
    'mean_final_cost': '{:,.3f}',
    'median_final_cost': '{:,.3f}',
    'mean_cost_minus_original_llm': '{:+,.3f}',
    'mean_cost_minus_compact_llm': '{:+,.3f}',
    'paired_wins_over_original_llm': lambda value: '' if pd.isna(value) else f'{int(value)}/5',
    'paired_wins_over_compact_llm': lambda value: '' if pd.isna(value) else f'{int(value)}/5',
    'rank_by_mean': '{:.0f}',
}))"""
    )
)

nb["cells"] = cells
nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT}")

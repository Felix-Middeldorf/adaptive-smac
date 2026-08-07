"""Generate the two reproducible RF-schedule analytics notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = (
    (
        ROOT
        / "clasp_queens_surrogate"
        / "05_bruteforce_rf_schedules"
        / "analyze_schedules.ipynb",
        "Clasp Queens",
        "clasp_queens",
    ),
    (
        ROOT
        / "clasp_weighted-sequence_surrogate"
        / "05_bruteforce_rf_schedules"
        / "analyze_schedules.ipynb",
        "Clasp Weighted-Sequence",
        "clasp_weighted",
    ),
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def make_notebook(display_name: str, benchmark_key: str):
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.9"}
    notebook.cells = [
        markdown(
            f"""
            # {display_name}: brute-force RF schedule analytics

            This notebook compares all 50 three-phase RF schedules, the five
            deterministic fixed-depth controls from `03_fixed_deterministic`
            requested for the comparison, and the five matched PCA=4
            fixed-depth controls from `04_fixed_deterministic_pca4`.

            Every reported performance is the incumbent configuration's mean
            PAR10 over **all training instances**, evaluated with surrogate
            quantile seed 0. Lower PAR10 is better. Rankings therefore do not
            use SMAC's partial-instance incumbent cost.

            The three schedule phases cover trials 0–499, 500–1,999, and
            2,000–4,999. All schedules use 100 trees, minimum leaf size 1,
            PCA=4, and deterministic target values.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Markdown, display

            HERE = Path.cwd().resolve()
            if not (HERE / "schedule_catalog.json").is_file():
                HERE = (
                    Path.cwd()
                    / "experiments/aclib"
                    / "{'clasp_queens_surrogate' if benchmark_key == 'clasp_queens' else 'clasp_weighted-sequence_surrogate'}"
                    / "05_bruteforce_rf_schedules"
                ).resolve()
            ACLIB_ROOT = HERE.parents[1]
            if str(ACLIB_ROOT) not in sys.path:
                sys.path.insert(0, str(ACLIB_ROOT))

            from rf_schedule_notebook_analytics import (
                build_notebook_data,
                findings_markdown,
                plot_checkpoint_trajectories,
                plot_early_vs_final,
                plot_overall_ranking,
                plot_phase_effects,
                plot_rank_stability,
                plot_runtime_performance,
                plot_top_five_by_seed,
                top_five_by_seed,
                top_five_overall,
                top_policy_definitions,
            )

            BENCHMARK_KEY = "{benchmark_key}"
            DISPLAY_NAME = "{display_name}"
            plt.rcParams.update({{"figure.dpi": 120, "axes.grid": False}})
            print("Experiment:", HERE)
            """
        ),
        markdown(
            """
            ## Data loading and audit

            The schedule checkpoint values were already independently
            validated on all training instances. The first execution may
            validate missing fixed-depth incumbents and cache those values in
            the corresponding fixed experiment's `analytics_cache` directory.

            The audit table confirms the number of distinct policies and
            seed runs included in each policy class.
            """
        ),
        code(
            """
            data = build_notebook_data(HERE, BENCHMARK_KEY)
            audit = (
                data.policy_rows.groupby("policy_type")
                .agg(
                    policies=("policy_id", "nunique"),
                    policy_seed_rows=("policy_id", "size"),
                    seeds=("smac_seed", "nunique"),
                    minimum_final_trials=("par10_5000", "count"),
                )
                .reset_index()
            )
            display(audit)
            assert data.policy_rows["policy_id"].nunique() == 60
            assert len(data.policy_rows) == 180
            """
        ),
        markdown(
            """
            ## Top five policies for each SMAC seed

            Each seed is ranked independently by final full-training PAR10 at
            trial 5,000. This reveals whether one schedule wins consistently
            or whether the preferred schedule depends strongly on SMAC's
            optimization seed. Bars include schedules and both fixed-depth
            control families.
            """
        ),
        code(
            """
            seed_top5 = top_five_by_seed(data)
            display(seed_top5)
            plot_top_five_by_seed(data)
            plt.show()
            """
        ),
        markdown(
            """
            ## Top five policies across all SMAC seeds

            The primary cross-seed ranking uses mean final full-training
            PAR10. Standard deviation, worst-seed performance, mean seed rank,
            and worst seed rank show how much confidence to place in the mean.
            The plot displays the top 15 so near-ties and fixed baselines
            remain visible.
            """
        ),
        code(
            """
            overall_top5 = top_five_overall(data)
            display(overall_top5)
            plot_overall_ranking(data, top_n=15)
            plt.show()
            """
        ),
        markdown(
            """
            ## RF definitions of the leading policies

            This table expands each leading policy into its depth,
            minimum-split size, and feature ratio in the three phases. Fixed
            controls repeat the same setting in every phase. It makes repeated
            motifs among successful schedules directly visible.
            """
        ),
        code(
            """
            definitions = top_policy_definitions(data, top_n=12)
            display(definitions)
            """
        ),
        markdown(
            """
            ## Comparison with fixed-depth controls

            For every schedule and seed, the table compares its final value
            with the best fixed PCA=4 depth for that same seed. Negative deltas
            favor the schedule. Wins count how many of the three seeds the
            schedule beats. The no-PCA controls from `03_fixed_deterministic`
            are also reported, but that delta combines policy and PCA effects.
            """
        ),
        code(
            """
            fixed_summary = (
                data.aggregate_rankings[
                    data.aggregate_rankings["policy_type"].str.startswith("fixed")
                ]
                .sort_values(["policy_type", "final_mean"])
                [
                    [
                        "policy_label",
                        "policy_type",
                        "overall_rank",
                        "final_mean",
                        "final_std",
                        "final_worst",
                        "mean_seed_rank",
                    ]
                ]
            )
            display(fixed_summary)
            display(data.baseline_comparison.head(15))
            """
        ),
        markdown(
            """
            ## How the leading policies improve over time

            Lines show mean incumbent PAR10 at trials 500, 2,000, and 5,000;
            shaded regions span the minimum to maximum across the three SMAC
            seeds. A curve that improves mainly after a switch indicates that
            later-phase settings may be useful, while a flat curve suggests
            most useful progress happened early.
            """
        ),
        code(
            """
            plot_checkpoint_trajectories(data, top_n=10)
            plt.show()
            """
        ),
        markdown(
            """
            ## Cross-seed rank stability

            The heatmap gives the rank of each overall leading policy within
            every seed. Uniformly low ranks indicate a robust policy; a mix of
            excellent and poor ranks warns that a good mean is driven by one
            favorable seed.
            """
        ),
        code(
            """
            plot_rank_stability(data, top_n=20)
            plt.show()
            """
        ),
        markdown(
            """
            ## Early performance versus final performance

            This scatter plot tests whether success by trial 500 predicts the
            final result. Points below the general trend are late improvers;
            points that start well but finish poorly may use unsuitable
            settings after a checkpoint.
            """
        ),
        code(
            """
            plot_early_vs_final(data)
            plt.show()
            """
        ),
        markdown(
            """
            ## Marginal effects of phase settings

            For phase 0, bars show PAR10 at trial 500 and lower is better. For
            phases 1 and 2, bars show improvement during that phase and higher
            is better. Error bars are standard deviations over all
            schedule-seed runs carrying the setting.

            These are descriptive associations, not isolated causal effects:
            the 50 schedules are diverse rather than a complete factorial
            design, so settings can be correlated across phases.
            """
        ),
        code(
            """
            phase_table = data.phase_effects.sort_values(
                ["phase", "factor", "mean"],
                ascending=[True, True, True],
            )
            display(phase_table)
            plot_phase_effects(data)
            plt.show()
            """
        ),
        markdown(
            """
            ## Runtime–performance trade-off

            The plot shows whether better policies systematically require more
            wall time. Runtime is averaged across seeds. Cluster placement
            also affects wall time, so this is useful for screening expensive
            settings but should not be interpreted as a pure RF-cost
            measurement.
            """
        ),
        code(
            """
            plot_runtime_performance(data)
            plt.show()
            """
        ),
        markdown(
            """
            ## Automatically calculated findings

            These statements summarize the current cached results and update
            automatically if the experiment data are replaced.
            """
        ),
        code(
            """
            display(Markdown(findings_markdown(data)))
            """
        ),
    ]
    return notebook


def main() -> None:
    for path, display_name, benchmark_key in NOTEBOOKS:
        notebook = make_notebook(display_name, benchmark_key)
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()

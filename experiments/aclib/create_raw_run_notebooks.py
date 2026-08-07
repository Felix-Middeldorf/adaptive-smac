"""Generate one training/test incumbent notebook per ACLib raw-run folder."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
BENCHMARKS = (
    ("clasp_queens_surrogate", "clasp_queens", "Clasp Queens"),
    (
        "clasp_weighted-sequence_surrogate",
        "clasp_weighted",
        "Clasp Weighted-Sequence",
    ),
    ("cplex_rcw_surrogate", "cplex_rcw", "CPLEX RCW"),
    (
        "lingeling_circuitfuzz_surrogate",
        "lingeling_circuitfuzz",
        "Lingeling CircuitFuzz",
    ),
    ("lpg_zenotravel_surrogate", "lpg_zenotravel", "LPG Zenotravel"),
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def make_notebook(folder: str, key: str, display_name: str):
    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "ACLib Python 3.9",
        "language": "python",
        "name": "aclib-py39",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.9"}
    notebook.cells = [
        md(
            f"""
            # {display_name}: raw SMAC incumbent trajectories

            This notebook reconstructs the incumbent that SMAC recorded at
            every incumbent-change trial for all depths, PCA settings, and
            SMAC seeds in `30_raw_run`.

            Each incumbent is independently evaluated twice with surrogate
            quantile seed 0: once over **all training instances** and once over
            **all held-out test instances**. The figures are separated by
            evaluation split so optimization performance and held-out
            generalization are not conflated. Lower PAR10 is better.

            Solid lines denote no PCA and dashed lines denote PCA=4. Colors
            identify RF depth. Runs that did not reach 5,000 trials stop at
            their actual native runhistory count. Every graph is shown first
            on a linear scale and then again on a logarithmic performance
            scale; the final train/test scatter uses logarithmic axes for both
            performance dimensions.
            """
        ),
        code(
            f"""
            from pathlib import Path
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import display

            HERE = Path.cwd().resolve()
            if not (HERE / "results").is_dir():
                HERE = (
                    Path.cwd()
                    / "experiments/aclib"
                    / "{folder}"
                    / "30_raw_run"
                ).resolve()
            ACLIB_ROOT = HERE.parents[1]
            if str(ACLIB_ROOT) not in sys.path:
                sys.path.insert(0, str(ACLIB_ROOT))

            from raw_run_analytics import (
                build_raw_run_analysis,
                plot_final_train_test,
                plot_incumbents_across_seeds,
                plot_incumbents_per_seed,
            )

            BENCHMARK_KEY = "{key}"
            DISPLAY_NAME = "{display_name}"
            plt.rcParams.update({{"figure.dpi": 120, "axes.grid": False}})
            print("Experiment:", HERE)
            """
        ),
        md(
            """
            ## Load, validate, and audit the current snapshot

            The validator caches each unique incumbent configuration, so
            rerunning the notebook only evaluates newly observed incumbents.
            The first audit table reports the current native trial count and
            number of incumbent changes for every run. `complete=False`
            indicates a timed-out or currently unfinished 5,000-trial run.
            """
        ),
        code(
            """
            analysis = build_raw_run_analysis(HERE, BENCHMARK_KEY)
            display(analysis.run_summary.reset_index(drop=True))
            print("Unique incumbent configurations:", analysis.trajectories["configuration_fingerprint"].nunique())
            print("Training cache:", analysis.cache_files["training"])
            print("Test cache:", analysis.cache_files["test"])
            """
        ),
        md(
            """
            ## Native evaluation coverage per run

            Each row below is one depth/PCA/SMAC-seed run.
            `evaluated_configurations` counts distinct configuration IDs with
            at least one completed native runhistory record; it is therefore
            not the number of acquisition candidates considered internally.
            `recorded_trials` counts all completed configuration-instance
            evaluations in the runhistory.

            The last-incumbent columns use the final configuration in SMAC's
            native intensifier trajectory. They report how many completed
            trials belong to that configuration and how many distinct
            training instances those trials cover. The coverage percentage is
            relative to the benchmark's complete training-instance set. In
            these deterministic runs one trial normally corresponds to one
            distinct instance, but both counts are retained so repetitions or
            partial coverage remain visible.
            """
        ),
        code(
            """
            coverage_columns = [
                "pca_mode", "depth", "smac_seed", "complete",
                "recorded_trials", "evaluated_configurations",
                "final_incumbent_config_id", "final_incumbent_trials",
                "final_incumbent_instances", "available_training_instances",
                "final_incumbent_instance_coverage_percent",
            ]
            evaluation_coverage = analysis.run_summary[coverage_columns].copy()
            evaluation_coverage["final_incumbent_instance_coverage_percent"] = (
                evaluation_coverage[
                    "final_incumbent_instance_coverage_percent"
                ].round(2)
            )
            display(evaluation_coverage.reset_index(drop=True))
            """
        ),
        md(
            """
            ## Training-instance incumbent trajectories: one plot per seed

            These plots answer how the incumbent selected during each SMAC run
            performs when re-evaluated over every training instance. A step
            occurs only when SMAC's recorded incumbent changes. Independent
            full-training performance can rise at a step because SMAC makes
            decisions from incrementally evaluated instance subsets. The
            second set of figures repeats the comparison with a logarithmic
            y-axis, making relative improvements near low PAR10 values easier
            to distinguish.
            """
        ),
        code(
            """
            training_seed_figures = plot_incumbents_per_seed(analysis, "training")
            for figure in training_seed_figures:
                display(figure)
                plt.close(figure)

            training_seed_log_figures = plot_incumbents_per_seed(
                analysis, "training", yscale="log"
            )
            for figure in training_seed_log_figures:
                display(figure)
                plt.close(figure)
            """
        ),
        md(
            """
            ## Training-instance trajectories aggregated over both seeds

            Each line is the mean of the two seed trajectories and its light
            band is their minimum-to-maximum range. With only two seeds this is
            deliberately a range, not a confidence interval. Every policy is
            shown only up to the smaller finished-trial count of its two runs,
            preventing a single surviving seed from determining the tail. The
            logarithmic version below contains the identical trajectories and
            seed ranges with only the y-axis transformation changed.
            """
        ),
        code(
            """
            training_all_seeds = plot_incumbents_across_seeds(analysis, "training")
            display(training_all_seeds)
            plt.close(training_all_seeds)

            training_all_seeds_log = plot_incumbents_across_seeds(
                analysis, "training", yscale="log"
            )
            display(training_all_seeds_log)
            plt.close(training_all_seeds_log)
            """
        ),
        md(
            """
            ## Test-instance incumbent trajectories: one plot per seed

            These are the same historical incumbents evaluated on every
            held-out test instance. Test instances never influenced SMAC.
            Comparing these figures with the training plots reveals whether
            apparent optimization progress transfers out of sample. Linear
            figures are followed by the same data on logarithmic y-axes.
            """
        ),
        code(
            """
            test_seed_figures = plot_incumbents_per_seed(analysis, "test")
            for figure in test_seed_figures:
                display(figure)
                plt.close(figure)

            test_seed_log_figures = plot_incumbents_per_seed(
                analysis, "test", yscale="log"
            )
            for figure in test_seed_log_figures:
                display(figure)
                plt.close(figure)
            """
        ),
        md(
            """
            ## Test-instance trajectories aggregated over both seeds

            Lines are seed means and bands are seed ranges, restricted to the
            common finished-trial horizon for each depth/PCA policy. This is
            the held-out counterpart of the aggregated training plot. The
            second graph repeats it with logarithmic performance scaling.
            """
        ),
        code(
            """
            test_all_seeds = plot_incumbents_across_seeds(analysis, "test")
            display(test_all_seeds)
            plt.close(test_all_seeds)

            test_all_seeds_log = plot_incumbents_across_seeds(
                analysis, "test", yscale="log"
            )
            display(test_all_seeds_log)
            plt.close(test_all_seeds_log)
            """
        ),
        md(
            """
            ## Final incumbent: training versus test performance

            Each point is one run's current final incumbent. The diagonal
            denotes equal training and test PAR10. Distance from the diagonal
            is a descriptive generalization gap, but train and test instance
            sets can also differ intrinsically in difficulty. The second
            scatter repeats the points with logarithmic training and test
            axes.
            """
        ),
        code(
            """
            final_columns = [
                "pca_mode", "depth", "smac_seed", "finished_trials",
                "training_mean_par10", "test_mean_par10",
                "test_minus_training_par10",
                "training_timeouts", "test_timeouts",
            ]
            final_table = analysis.final_incumbents[final_columns].sort_values(
                ["test_mean_par10", "training_mean_par10"]
            )
            display(final_table.reset_index(drop=True))
            final_scatter = plot_final_train_test(analysis)
            display(final_scatter)
            plt.close(final_scatter)

            final_log_scatter = plot_final_train_test(analysis, scale="log")
            display(final_log_scatter)
            plt.close(final_log_scatter)
            """
        ),
    ]
    return notebook


def main() -> None:
    for folder, key, display_name in BENCHMARKS:
        path = ROOT / folder / "30_raw_run" / "analyze_results.ipynb"
        nbf.write(make_notebook(folder, key, display_name), path)
        print(path)


if __name__ == "__main__":
    main()

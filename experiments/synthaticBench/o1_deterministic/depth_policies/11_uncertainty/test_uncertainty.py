from __future__ import annotations

from o1_uncertainty_runner import (
    BENCHMARK_SEEDS,
    DEPTHS,
    DIMENSION,
    N_INSTANCES,
    N_TRIALS,
    PROBLEM_CONFIG,
    PROPOSAL_SCHEMA_VERSION,
    SMAC_SEEDS,
    is_acquisition_selected_origin,
    make_instance_map,
    proposal_diagnostics_are_valid,
)
from submit_uncertainty import MAX_SUBMITTED_JOBS, experiment_jobs


def test_requested_experiment_matrix() -> None:
    assert DEPTHS == (5, 10, 15, 20, 25, 30)
    assert BENCHMARK_SEEDS == (40, 42)
    assert SMAC_SEEDS == (0, 1, 2)
    assert DIMENSION == 12
    assert N_INSTANCES == 12
    assert N_TRIALS == 2_000


def test_job_matrix_is_complete_and_unique() -> None:
    jobs = experiment_jobs()
    assert len(jobs) == 6 * 2 * 3 == 36
    assert len(set(jobs)) == len(jobs)
    assert len(jobs) <= MAX_SUBMITTED_JOBS
    assert {
        (job.benchmark_seed, job.smac_seed, job.depth)
        for job in jobs
    } == {
        (benchmark_seed, smac_seed, depth)
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
        for depth in DEPTHS
    }


def test_shared_instances_and_problem_config() -> None:
    assert make_instance_map() == make_instance_map()
    assert len(make_instance_map()) == N_INSTANCES
    assert PROBLEM_CONFIG.is_file()


def test_acquisition_origin_classification() -> None:
    assert is_acquisition_selected_origin(
        "Acquisition Function Maximizer: Local Search"
    )
    assert is_acquisition_selected_origin(
        "Acquisition Function Maximizer: Random Search (sorted)"
    )
    assert not is_acquisition_selected_origin("Initial Design: Sobol")
    assert not is_acquisition_selected_origin(
        "Random Search (max retries, no candidates)"
    )
    assert not is_acquisition_selected_origin(None)


def test_diagnostics_validator_requires_all_prediction_fields() -> None:
    selected_record = {
        "selected_by_acquisition": True,
        "model_max_depth": 10,
        "predicted_cost_mean_marginalized": 1.0,
        "predicted_variance_marginalized": 2.0,
        "predicted_std_marginalized": 2.0 ** 0.5,
        "acquisition_value": 0.25,
        "acquisition_eta": 0.5,
    }
    diagnostics = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "max_depth": 10,
        "total_configuration_selector_proposals": 1,
        "acquisition_selected_proposals": 1,
        "proposals": [selected_record],
    }
    assert proposal_diagnostics_are_valid(diagnostics, 10)
    selected_record["predicted_variance_marginalized"] = None
    assert not proposal_diagnostics_are_valid(diagnostics, 10)


if __name__ == "__main__":
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)} tests passed.")

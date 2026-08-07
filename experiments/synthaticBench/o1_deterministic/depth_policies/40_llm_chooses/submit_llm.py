#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Validate or submit the direct-OpenAI O1 RF-policy jobs."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import submitit
from smac.runhistory.enumerations import StatusType

from o1_llm_runner import (
    BENCHMARK_SEEDS,
    CHECKPOINTS,
    DEFAULT_SETTINGS,
    DIAGNOSTIC_SAMPLE_SIZE,
    DIMENSION,
    HERE,
    LLMRFPolicyCallback,
    LOCAL_SMAC_ROOT,
    N_INSTANCES,
    N_TRIALS,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    OUTPUT_ROOT,
    PYTHONHASHSEED,
    SMAC_SEEDS,
    TELEMETRY_FILENAME,
    append_jsonl,
    configuration_fingerprint,
    evenly_spaced_trial_numbers,
    local_smac_metadata,
    run_llm_policy,
)


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 4 * 60
MEM_GB = 4
API_KEY_FILE = Path("/home/io632776/.config/openai/smac_api_key")
PYTHON_ENV = Path(
    "/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python"
)


@dataclass(frozen=True)
class LLMPolicyJob:
    benchmark_seed: int
    smac_seed: int

    def __call__(self):
        if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
            raise RuntimeError("PYTHONHASHSEED is not fixed.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is unavailable in the job.")
        return run_llm_policy(self.benchmark_seed, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[LLMPolicyJob, ...]:
    jobs = tuple(
        LLMPolicyJob(benchmark_seed, smac_seed)
        for benchmark_seed in BENCHMARK_SEEDS
        for smac_seed in SMAC_SEEDS
    )
    if len(jobs) != 10 or len(set(jobs)) != 10:
        raise RuntimeError("Expected ten unique benchmark/SMAC-seed jobs.")
    return jobs


def validate_api_key_file() -> None:
    if not API_KEY_FILE.is_file():
        raise RuntimeError(f"API key file does not exist: {API_KEY_FILE}.")
    mode = API_KEY_FILE.stat().st_mode & 0o777
    if mode != 0o600:
        raise RuntimeError(
            f"API key file permissions are {mode:o}; expected 600."
        )
    value = API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not value.startswith("sk-"):
        raise RuntimeError("API key file has an unexpected prefix.")


def slurm_parameters() -> dict[str, Any]:
    validate_api_key_file()
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": 10,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": "SynthACtic_O1_40_OpenAI_RF",
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'test -r "{API_KEY_FILE}"',
            f'export OPENAI_API_KEY="$(< "{API_KEY_FILE}")"',
            (
                "export PYTHONPATH="
                f'"{LOCAL_SMAC_ROOT}:{HERE}:${{PYTHONPATH:-}}"'
            ),
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(list_jobs: bool = False) -> None:
    print(f"Jobs: {len(experiment_jobs())}")
    print(f"Benchmark seeds: {BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Dimensions/instances: {DIMENSION}/{N_INSTANCES}")
    print(f"Trials per job: {N_TRIALS}")
    print(f"LLM checkpoints: {CHECKPOINTS}")
    print(f"Runhistory rows per request: {DIAGNOSTIC_SAMPLE_SIZE}")
    print(f"Initial RF defaults: {DEFAULT_SETTINGS.to_dict()}")
    print(f"OpenAI model: {OPENAI_MODEL}")
    print(f"Reasoning effort: {OPENAI_REASONING_EFFORT}")
    print(f"Secure key file: {API_KEY_FILE} (contents not printed)")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Time/memory: {TIMEOUT_MIN} minutes / {MEM_GB} GB")
    print(f"Python: {PYTHON_ENV}")
    print(f"Output root: {OUTPUT_ROOT}")
    if list_jobs:
        for index, job in enumerate(experiment_jobs()):
            print(
                f"job={index:02d} benchmark_seed={job.benchmark_seed} "
                f"smac_seed={job.smac_seed}"
            )


def smoke_check() -> None:
    assert CHECKPOINTS == (100, 250, 500)
    assert N_TRIALS == 1_000
    assert BENCHMARK_SEEDS == (40, 42)
    assert SMAC_SEEDS == tuple(range(5))
    assert DIMENSION == N_INSTANCES == 10
    assert OPENAI_MODEL == "gpt-5.6-terra"
    assert OPENAI_REASONING_EFFORT == "medium"
    assert evenly_spaced_trial_numbers(100) == list(range(1, 101))
    assert evenly_spaced_trial_numbers(500) == list(range(5, 501, 5))
    assert len(set(evenly_spaced_trial_numbers(250))) == 100
    parameters = slurm_parameters()
    assert parameters["slurm_account"] == "thes2388"
    assert parameters["slurm_array_parallelism"] == 10
    assert parameters["timeout_min"] == 240
    assert parameters["mem_gb"] == 4
    metadata = local_smac_metadata()
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["module"]

    class _Config(dict):
        origin = "Acquisition Function Maximizer: Local Search"

    class _Tree:
        def __init__(self, depth: int) -> None:
            self.depth = depth

        def get_depth(self) -> int:
            return self.depth

    class _Model:
        def __init__(self) -> None:
            self._rf_opts = {
                "n_estimators": 10,
                "max_depth": 20,
                "min_samples_split": 3,
                "min_samples_leaf": 3,
            }
            self._ratio_features = 5.0 / 6.0
            self._rf = None

        def train(self, X, y):
            del y
            depth = self._rf_opts["max_depth"]
            self._rf = SimpleNamespace(
                estimators_=[_Tree(depth - 1), _Tree(depth)],
            )
            return self

        def predict_marginalized(self, array):
            count = len(array)
            return np.full((count, 1), 9.0), np.full((count, 1), 4.0)

    class _History:
        def __init__(self, finished: int, config: _Config) -> None:
            self.finished = finished
            self.config = config
            self.rows = []
            for index in range(500):
                key = SimpleNamespace(config_id=1, instance=f"i{index % 10}")
                value = SimpleNamespace(
                    status=StatusType.SUCCESS,
                    cost=10.0 + index / 100.0,
                    time=0.01,
                    starttime=float(index),
                    endtime=float(index) + 0.01,
                )
                self.rows.append((key, value))

        def __len__(self) -> int:
            return self.finished

        def items(self):
            return self.rows[: self.finished]

        def get_config(self, config_id: int):
            assert config_id == 1
            return self.config

    choices = iter(
        (
            (50, 10, 2, 1, 0.5),
            (100, 15, 4, 3, 1.0),
            (50, 30, 8, 5, 0.3),
        )
    )
    prompts = []

    def fake_api(prompt: str):
        prompts.append(prompt)
        trees, depth, split, leaf, ratio = next(choices)
        return {
            "n_trees": trees,
            "max_depth": depth,
            "min_samples_split": split,
            "min_samples_leaf": leaf,
            "feature_ratio": ratio,
            "confidence": 0.75,
            "reason": "Structured smoke-test decision.",
        }, {
            "response_id": "smoke-test",
            "model_returned": "smoke-test",
            "elapsed_seconds": 0.0,
            "usage": {},
        }

    model = _Model()
    config = _Config({f"x_{index}": 0.0 for index in range(10)})
    history = _History(0, config)
    with tempfile.TemporaryDirectory(prefix="o1-openai-policy-smoke-") as tmp:
        callback = LLMRFPolicyCallback(
            output_path=Path(tmp),
            model=model,
            decision_provider=fake_api,
            overwrite=True,
        )
        model.train(X=[[0.0] * 10], y=[1.0])
        append_jsonl(
            Path(tmp) / TELEMETRY_FILENAME,
            {
                "event_type": "proposal",
                "proposal_index": 1,
                "completed_trials_before_proposal": 0,
                "configuration_fingerprint": configuration_fingerprint(config),
                "origin": "Acquisition Function Maximizer: Local Search",
                "model_is_trained": True,
                "expected_improvement": 2.5,
                "predicted_marginal_mean": 9.0,
                "predicted_marginal_variance": 4.0,
                "rf_settings": DEFAULT_SETTINGS.to_dict(),
                "telemetry_error": None,
            },
        )
        for checkpoint in CHECKPOINTS:
            history.finished = checkpoint
            callback.on_next_configurations_start(
                SimpleNamespace(_runhistory=history)
            )
            model.train(X=[[0.0] * 10], y=[1.0])
        audit = callback.audit()
        assert len(audit["decisions"]) == 3
        assert len(prompts) == 3
        assert callback.next_settings.max_depth == 30
        checkpoint_500 = Path(tmp) / "llm_requests/checkpoint_0500/llm_input.json"
        data = __import__("json").loads(checkpoint_500.read_text())
        rows = data["sampled_runhistory"]
        assert len(rows) == 100
        assert [row["trial_number"] for row in rows] == list(range(5, 501, 5))
        assert rows[0]["expected_improvement"] == 2.5
        assert rows[0]["absolute_proxy_error"] == 1.0
        assert rows[0]["relative_proxy_error"] == 0.1
        assert rows[0]["rf_prediction_variance"] == 4.0
        assert rows[0]["configuration_values"] == dict(config)
        assert "anonymous_configuration_id" not in rows[0]
        assert "target_evaluation_seconds" not in rows[0]
        assert "OPENAI_API_KEY" not in "".join(prompts)
        assert configuration_fingerprint(config) not in "".join(prompts)
    print("PASS: 2 benchmark seeds x 5 SMAC seeds = 10 jobs.")
    print("PASS: 1,000 trials and LLM checkpoints 100/250/500.")
    print("PASS: each request contains exactly 100 evenly spaced trial rows.")
    print("PASS: EI, prediction, variance, and proxy errors align by configuration.")
    print("PASS: strict decisions update all five RF hyperparameters.")
    print("PASS: secure API-key file, local SMAC, project thes2388.")


def submit_jobs() -> None:
    print_summary()
    executor = submitit.AutoExecutor(
        folder=str(HERE / "submitit_logs"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(**slurm_parameters())
    submitted = []
    with executor.batch():
        for specification in experiment_jobs():
            submitted.append(
                (specification, executor.submit(specification))
            )
    print(f"Submitted {len(submitted)} direct-OpenAI policy jobs.")
    for specification, job in submitted:
        print(
            f"benchmark_seed={specification.benchmark_seed}, "
            f"smac_seed={specification.smac_seed}: {job.job_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-jobs", action="store_true")
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()
    if args.smoke_check:
        smoke_check()
    elif args.dry_run or args.list_jobs:
        print_summary(list_jobs=args.list_jobs)
    else:
        submit_jobs()


if __name__ == "__main__":
    main()

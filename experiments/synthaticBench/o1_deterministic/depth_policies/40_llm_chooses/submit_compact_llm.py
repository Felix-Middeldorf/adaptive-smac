#!/home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python
"""Validate or submit the compact direct-OpenAI O1 RF-policy jobs."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import submitit
from smac.runhistory.enumerations import StatusType

import o1_llm_runner as base
from o1_compact_llm_runner import (
    CompactLLMRFPolicyCallback,
    CompactRFSettings,
    DECISION_SCHEMA,
    POLICY_NAME,
    SUMMARY_WINDOWS,
    run_compact_llm_policy,
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
class CompactLLMPolicyJob:
    benchmark_seed: int
    smac_seed: int

    def __call__(self):
        if os.environ.get("PYTHONHASHSEED") != base.PYTHONHASHSEED:
            raise RuntimeError("PYTHONHASHSEED is not fixed.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is unavailable in the job.")
        return run_compact_llm_policy(self.benchmark_seed, self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[CompactLLMPolicyJob, ...]:
    jobs = tuple(
        CompactLLMPolicyJob(benchmark_seed, smac_seed)
        for benchmark_seed in base.BENCHMARK_SEEDS
        for smac_seed in base.SMAC_SEEDS
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
        "slurm_job_name": "SynthACtic_O1_40_Compact_OpenAI_RF",
        "slurm_setup": [
            f"export PYTHONHASHSEED={base.PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f'test -r "{API_KEY_FILE}"',
            f'export OPENAI_API_KEY="$(< "{API_KEY_FILE}")"',
            (
                "export PYTHONPATH="
                f'"{base.LOCAL_SMAC_ROOT}:{base.HERE}:${{PYTHONPATH:-}}"'
            ),
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(list_jobs: bool = False) -> None:
    print(f"Jobs: {len(experiment_jobs())}")
    print(f"Benchmark seeds: {base.BENCHMARK_SEEDS}")
    print(f"SMAC seeds: {base.SMAC_SEEDS}")
    print(f"Dimensions/instances: {base.DIMENSION}/{base.N_INSTANCES}")
    print(f"Trials per job: {base.N_TRIALS}")
    print(f"LLM checkpoints: {base.CHECKPOINTS}")
    print(f"Compact trial windows per request: {SUMMARY_WINDOWS}")
    print("Raw configuration rows per request: 0")
    print(f"Initial RF defaults: {base.DEFAULT_SETTINGS.to_dict()}")
    print("Allowed trees/depth/split/leaf/ratio: 1-100/1-30/2-10/1-10/(0,1]")
    print(f"OpenAI model: {base.OPENAI_MODEL}")
    print(f"Reasoning effort: {base.OPENAI_REASONING_EFFORT}")
    print(f"Secure key file: {API_KEY_FILE} (contents not printed)")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Time/memory: {TIMEOUT_MIN} minutes / {MEM_GB} GB")
    print(f"Python: {PYTHON_ENV}")
    print(f"Output policy: {POLICY_NAME}")
    if list_jobs:
        for index, job in enumerate(experiment_jobs()):
            print(
                f"job={index:02d} benchmark_seed={job.benchmark_seed} "
                f"smac_seed={job.smac_seed}"
            )


def smoke_check() -> None:
    assert base.CHECKPOINTS == (100, 250, 500)
    assert base.N_TRIALS == 1_000
    assert base.BENCHMARK_SEEDS == (40, 42)
    assert base.SMAC_SEEDS == tuple(range(5))
    assert base.DIMENSION == base.N_INSTANCES == 10
    assert base.OPENAI_MODEL == "gpt-5.6-terra"
    assert base.OPENAI_REASONING_EFFORT == "medium"
    assert DECISION_SCHEMA["properties"]["n_trees"]["maximum"] == 100
    assert DECISION_SCHEMA["properties"]["max_depth"]["maximum"] == 30
    assert DECISION_SCHEMA["properties"]["min_samples_split"]["maximum"] == 10
    assert DECISION_SCHEMA["properties"]["min_samples_leaf"]["maximum"] == 10
    assert DECISION_SCHEMA["properties"]["feature_ratio"]["maximum"] == 1.0
    CompactRFSettings(73, 27, 9, 7, 0.67)
    parameters = slurm_parameters()
    assert parameters["slurm_account"] == "thes2388"
    assert parameters["slurm_array_parallelism"] == 10
    assert parameters["timeout_min"] == 240
    assert parameters["mem_gb"] == 4
    metadata = base.local_smac_metadata()
    assert str(base.LOCAL_SMAC_ROOT.resolve()) in metadata["module"]

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
                estimators_=[_Tree(max(1, depth - 1)), _Tree(depth)],
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
            (73, 27, 9, 7, 0.67),
            (41, 13, 6, 4, 0.42),
            (99, 29, 10, 8, 0.91),
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
            "reason": "Structured compact-summary smoke-test decision.",
        }, {
            "response_id": "smoke-test",
            "model_returned": "smoke-test",
            "elapsed_seconds": 0.0,
            "usage": {},
        }

    model = _Model()
    config = _Config({f"x_{index}": float(index) for index in range(10)})
    history = _History(0, config)
    with tempfile.TemporaryDirectory(prefix="o1-compact-policy-smoke-") as tmp:
        tmp_path = Path(tmp)
        callback = CompactLLMRFPolicyCallback(
            output_path=tmp_path,
            model=model,
            decision_provider=fake_api,
            overwrite=True,
        )
        model.train(X=[[0.0] * 10], y=[1.0])
        base.append_jsonl(
            tmp_path / base.TELEMETRY_FILENAME,
            {
                "event_type": "proposal",
                "proposal_index": 1,
                "completed_trials_before_proposal": 0,
                "configuration_fingerprint": base.configuration_fingerprint(
                    config
                ),
                "origin": "Acquisition Function Maximizer: Local Search",
                "model_is_trained": True,
                "expected_improvement": 2.5,
                "predicted_marginal_mean": 9.0,
                "predicted_marginal_variance": 4.0,
                "rf_settings": base.DEFAULT_SETTINGS.to_dict(),
                "telemetry_error": None,
            },
        )
        for checkpoint in base.CHECKPOINTS:
            history.finished = checkpoint
            callback.on_next_configurations_start(
                SimpleNamespace(_runhistory=history)
            )
            model.train(X=[[0.0] * 10], y=[1.0])
        audit = callback.audit()
        assert len(audit["decisions"]) == 3
        assert len(prompts) == 3
        assert callback.next_settings == CompactRFSettings(
            99, 29, 10, 8, 0.91
        )
        data = json.loads(
            (
                tmp_path
                / "llm_requests/checkpoint_0500/llm_input.json"
            ).read_text()
        )
        assert len(data["trial_windows"]) == SUMMARY_WINDOWS
        search_space = data["optimization_search_space"]
        assert search_space["total_dimensions"] == 10
        assert len(search_space["parameters"]) == 10
        assert search_space["parameters"][0] == {
            "name": "x_0",
            "lower_bound": -100.0,
            "upper_bound": 100.0,
        }
        assert search_space["parameters"][-1]["name"] == "x_9"
        assert "sampled_runhistory" not in data
        assert "configuration_values" not in "".join(prompts)
        assert '"compact_constraint":"-100 <= x_i <= 100 for i=0,...,9"' in prompts[-1]
        assert len(prompts[-1]) < 20_000
        print(
            "Measured checkpoint-500 compact prompt: "
            f"{len(prompts[-1]):,} characters, approximately "
            f"{len(prompts[-1]) / 4:,.0f} tokens by the chars/4 heuristic."
        )

    with tempfile.TemporaryDirectory(prefix="o1-compact-e2e-") as tmp:
        result = run_compact_llm_policy(
            40,
            0,
            n_trials=2,
            output_root=Path(tmp),
            decision_provider=fake_api,
        )
        assert result["finished_trials"] == 2
        assert result["policy"] == POLICY_NAME
        assert result["summary_mode"] == "ten_window_compact_aggregates"

    integration_prompts = []

    def integration_api(prompt: str):
        integration_prompts.append(prompt)
        return {
            "n_trees": 67,
            "max_depth": 23,
            "min_samples_split": 7,
            "min_samples_leaf": 6,
            "feature_ratio": 0.61,
            "confidence": 0.8,
            "reason": "Real-runhistory compact integration decision.",
        }, {
            "response_id": "integration-test",
            "model_returned": "integration-test",
            "elapsed_seconds": 0.0,
            "usage": {},
        }

    with tempfile.TemporaryDirectory(prefix="o1-compact-checkpoint-") as tmp:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            result = run_compact_llm_policy(
                40,
                0,
                n_trials=101,
                output_root=Path(tmp),
                decision_provider=integration_api,
            )
        assert result["finished_trials"] == 101
        assert len(integration_prompts) == 1
        assert len(integration_prompts[0]) < 20_000
        assert result["llm_policy"]["decisions"]["100"]["settings"] == {
            "n_trees": 67,
            "max_depth": 23,
            "min_samples_split": 7,
            "min_samples_leaf": 6,
            "feature_ratio": 0.61,
        }
        print(
            "Measured real-runhistory checkpoint-100 compact prompt: "
            f"{len(integration_prompts[0]):,} characters, approximately "
            f"{len(integration_prompts[0]) / 4:,.0f} tokens."
        )

    print("PASS: 2 benchmark seeds x 5 SMAC seeds = 10 jobs.")
    print("PASS: 1,000 trials and LLM checkpoints 100/250/500.")
    print("PASS: compact ten-window summaries contain no raw configuration rows.")
    print("PASS: a real 101-trial SMAC run triggered one compact fake-API decision.")
    print("PASS: expanded bounded structured decisions update all RF settings.")
    print("PASS: secure API-key file, local SMAC, project thes2388.")


def submit_jobs() -> None:
    print_summary()
    executor = submitit.AutoExecutor(
        folder=str(base.HERE / "submitit_logs_compact"),
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
    print(f"Submitted {len(submitted)} compact direct-OpenAI policy jobs.")
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

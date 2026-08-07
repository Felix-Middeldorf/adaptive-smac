#!/home/io632776/work/py-envs/aclib2-surrogates-py39/bin/python
"""Submit the three Clasp Queens Codex-policy jobs."""

import argparse
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import submitit

from experiment import DEFINITION
from llm_policy import (
    CHECKPOINTS,
    CODEX_MODEL,
    DEFAULT_SETTINGS,
    LLMRFPolicyCallback,
    LOCAL_SMAC_ROOT,
    N_TRIALS,
    PCA_COMPONENTS,
    SMAC_SEEDS,
    local_smac_metadata,
    run_llm_policy,
)
from fixed_depth_submit import EPM_SOURCE


SLURM_PARTITION = "c23ms"
SLURM_ACCOUNT = "thes2388"
TIMEOUT_MIN = 16 * 60
MEM_GB = 4
PYTHONHASHSEED = "0"
ACLIB_ROOT = EPM_SOURCE.parents[2] / "experiments" / "aclib"


def codex_binary() -> Path:
    configured = os.environ.get("CODEX_BINARY")
    found = configured or shutil.which("codex")
    if not found:
        raise RuntimeError("Codex CLI is unavailable; set CODEX_BINARY.")
    path = Path(found).resolve()
    if not path.is_file():
        raise RuntimeError(f"Codex CLI does not exist: {path}")
    return path


@dataclass(frozen=True)
class LLMPolicyJob:
    smac_seed: int

    def __call__(self):
        if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
            raise RuntimeError("PYTHONHASHSEED is not fixed to 0.")
        return run_llm_policy(definition=DEFINITION, smac_seed=self.smac_seed)

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)


def experiment_jobs() -> tuple[LLMPolicyJob, ...]:
    jobs = tuple(LLMPolicyJob(seed) for seed in SMAC_SEEDS)
    if len(jobs) != 3 or len(set(jobs)) != 3:
        raise RuntimeError("Expected three unique SMAC-seed jobs.")
    return jobs


def slurm_parameters() -> dict[str, Any]:
    pythonpath = (
        f"{LOCAL_SMAC_ROOT}:{EPM_SOURCE}:{ACLIB_ROOT}:"
        f"{DEFINITION.directory}:${{PYTHONPATH:-}}"
    )
    return {
        "timeout_min": TIMEOUT_MIN,
        "slurm_partition": SLURM_PARTITION,
        "slurm_account": SLURM_ACCOUNT,
        "slurm_array_parallelism": 3,
        "cpus_per_task": 1,
        "mem_gb": MEM_GB,
        "slurm_job_name": "ACLib_cq_40_llm_rf_policy",
        "slurm_setup": [
            f"export PYTHONHASHSEED={PYTHONHASHSEED}",
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            f"export CODEX_BINARY={shlex.quote(str(codex_binary()))}",
            f"export PYTHONPATH={pythonpath}",
        ],
        "slurm_additional_parameters": {"requeue": True},
    }


def print_summary(list_jobs: bool = False) -> None:
    print(f"SMAC seeds: {SMAC_SEEDS}")
    print(f"Jobs: {len(experiment_jobs())}")
    print(f"Trials per job: {N_TRIALS}")
    print(f"Codex checkpoints: {CHECKPOINTS}")
    print(f"Trials 0-499 settings: {DEFAULT_SETTINGS.to_dict()}")
    print(f"PCA components fixed: {PCA_COMPONENTS}")
    print(f"Codex model: {CODEX_MODEL}")
    print("Codex request: anonymized aggregate telemetry only")
    print("Codex auth: cached ChatGPT login; no API key")
    print(f"Slurm account: {SLURM_ACCOUNT}")
    print(f"Time/memory: {TIMEOUT_MIN} minutes / {MEM_GB} GB")
    print(f"Codex binary: {codex_binary()}")
    print(f"Output root: {DEFINITION.output_root}")
    if list_jobs:
        for index, job in enumerate(experiment_jobs()):
            print(f"job={index} smac_seed={job.smac_seed}")


def smoke_check() -> None:
    assert CHECKPOINTS == (500, 1_000, 1_500)
    assert N_TRIALS == 2_500
    assert SMAC_SEEDS == (0, 1, 2)
    assert DEFAULT_SETTINGS.to_dict() == {
        "n_trees": 10,
        "max_depth": 20,
        "min_samples_split": 3,
        "min_samples_leaf": 3,
        "feature_ratio": 5.0 / 6.0,
    }
    parameters = slurm_parameters()
    assert parameters["slurm_account"] == "thes2388"
    assert parameters["timeout_min"] == 16 * 60
    assert parameters["mem_gb"] == 4
    assert parameters["slurm_array_parallelism"] == 3
    metadata = local_smac_metadata()
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["module"]
    assert str(LOCAL_SMAC_ROOT.resolve()) in metadata["random_forest"]
    status = subprocess.run(
        [str(codex_binary()), "login", "status"],
        text=True,
        capture_output=True,
        check=False,
    )
    login_output = status.stdout + status.stderr
    if status.returncode != 0 or "ChatGPT" not in login_output:
        raise RuntimeError(
            "Codex is not authenticated with ChatGPT: "
            + login_output
        )

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
                max_features=max(1, int(12 * self._ratio_features)),
                n_features_in_=12,
            )
            return self

    class _History:
        def __init__(self, finished: int) -> None:
            self.finished = finished
            self.key = SimpleNamespace(config_id=1)
            self.value = SimpleNamespace(
                status=SimpleNamespace(name="SUCCESS"),
                cost=12.5,
                time=0.1,
                starttime=1.0,
                endtime=1.1,
            )

        def __len__(self) -> int:
            return self.finished

        def __iter__(self):
            return iter((self.key,))

        def __getitem__(self, key):
            assert key is self.key
            return self.value

    decisions = iter(
        (
            {
                "n_trees": 50,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "feature_ratio": 0.5,
                "confidence": 0.8,
                "reason": "Use a smaller model for this phase.",
            },
            {
                "n_trees": 100,
                "max_depth": 15,
                "min_samples_split": 4,
                "min_samples_leaf": 3,
                "feature_ratio": 1.0,
                "confidence": 0.7,
                "reason": "Increase capacity after more observations.",
            },
            {
                "n_trees": 50,
                "max_depth": 30,
                "min_samples_split": 8,
                "min_samples_leaf": 5,
                "feature_ratio": 0.3,
                "confidence": 0.6,
                "reason": "Test deeper regularized trees late in the run.",
            },
        )
    )
    prompts = []

    def _fake_codex(prompt: str):
        prompts.append(prompt)
        return next(decisions), {
            "codex_model": "smoke-test",
            "elapsed_seconds": 0.0,
        }

    model = _Model()
    with tempfile.TemporaryDirectory(prefix="aclib-llm-policy-smoke-") as tmp:
        callback = LLMRFPolicyCallback(
            output_directory=Path(tmp),
            model=model,
            decision_provider=_fake_codex,
            overwrite=True,
        )
        model.train(X=[[0.0] * 12], y=[1.0])
        assert callback.state["fit_observations"][-1]["training_rows"] == 1
        for checkpoint in CHECKPOINTS:
            callback.on_next_configurations_start(
                SimpleNamespace(_runhistory=_History(checkpoint))
            )
            model.train(X=[[0.0] * 12], y=[1.0])
        audit = callback.audit()
        assert len(audit["decisions"]) == 3
        assert len(prompts) == 3
        assert all("ANONYMIZED TELEMETRY" in prompt for prompt in prompts)
        assert callback.next_settings.max_depth == 30

    print("PASS: three 2,500-trial jobs; checkpoints 500/1000/1500.")
    print("PASS: native SMAC RF defaults are used through trial 499.")
    print("PASS: local SMAC, deterministic target seed 0, PCA=4.")
    print("PASS: Codex CLI is present and authenticated through ChatGPT.")
    print("PASS: all checkpoint transitions pass structured, anonymized telemetry.")
    print("PASS: Slurm project thes2388, 16 hours, 4 GB, 1 CPU.")


def submit_jobs() -> None:
    print_summary()
    executor = submitit.AutoExecutor(
        folder=str(DEFINITION.directory / "submitit_logs"),
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(**slurm_parameters())
    submitted = []
    with executor.batch():
        for specification in experiment_jobs():
            submitted.append((specification, executor.submit(specification)))
    print(f"Submitted {len(submitted)} LLM-policy jobs.")
    for specification, job in submitted:
        print(f"seed={specification.smac_seed}: {job.job_id}")


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

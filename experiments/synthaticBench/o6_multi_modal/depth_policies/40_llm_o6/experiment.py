"""RWTHGPT RF-policy study for the O6-Multimodal benchmark."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai
from carps.utils.running import make_problem
from carps.utils.trials import TrialInfo
from omegaconf import OmegaConf
from smac.initial_design import RandomInitialDesign


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
SHARED_POLICY_CODE = (
    REPOSITORY_ROOT
    / "experiments/synthaticBench/o1_deterministic/depth_policies/40_llm_chooses"
)
if str(SHARED_POLICY_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_POLICY_CODE))

import o1_compact_llm_runner as compact
import o1_llm_runner as base


BENCHMARK_SEED = 52
DIMENSIONS = (50, 100)
N_INSTANCES = 10
N_TRIALS = 1_000
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
PCA_COMPONENTS = 4
N_INITIAL_CONFIGS = 10
RANDOM_DESIGN_PROBABILITY = 0.0
GPT55_SEEDS = (0, 1, 2)
GPT54_SEEDS = (0, 1, 2, 3, 4, 5)
FIXED_SEEDS = GPT54_SEEDS
RWTHGPT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
RWTHGPT_API_KEY_FILE = Path.home() / ".config/kiconnect/rwthgpt_api_key"
MAX_INPUT_TOKENS = 200_000
EXPERIMENT_VERSION = 2
POLICY_VERSION = 1
OUTPUT_ROOT = HERE / "results"
PROBLEM_CONFIG = (
    REPOSITORY_ROOT
    / "external/SynthACticBench/synthacticbench/configs/problem/"
    "SynthACticBench/O6-Multimodal.yaml"
)


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    smac_seeds: tuple[int, ...]


MODELS = (
    ModelSpec("gpt-5.4-mini", GPT54_SEEDS),
    ModelSpec("gpt-5.5", GPT55_SEEDS),
)
MODELS_BY_ID = {model.identifier: model for model in MODELS}

RFSettings = compact.CompactRFSettings
DEFAULT_SETTINGS = RFSettings(
    n_trees=10,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=3,
    feature_ratio=5.0 / 6.0,
)
FIXED_100_TREE_SETTINGS = RFSettings(
    n_trees=100,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=3,
    feature_ratio=5.0 / 6.0,
)


def dimension_root(dimension: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return Path(output_root) / f"dimension_{dimension}"


def policy_name(kind: str, model: str | None = None) -> str:
    if kind == "fixed_100_trees":
        return kind
    if model is None:
        raise ValueError("An LLM policy requires a model identifier.")
    return f"{kind}_{model.replace('.', '_').replace('-', '_')}"


def output_directory(
    dimension: int,
    kind: str,
    smac_seed: int,
    model: str | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    return (
        dimension_root(dimension, output_root)
        / f"benchmark_seed_{BENCHMARK_SEED}"
        / policy_name(kind, model)
        / str(smac_seed)
    )


def configuration_space_summary(dimension: int) -> dict[str, Any]:
    return {
        "total_dimensions": dimension,
        "parameter_type": "continuous real-valued",
        "parameters": {
            "count": dimension,
            "name_pattern": "x_0 through x_{dimension_minus_1}",
            "lower_bound": -600.0,
            "upper_bound": 600.0,
            "default": 1.0,
        },
        "hierarchy": "none; every parameter is always active",
        "conditionals": False,
        "forbidden_combinations": False,
        "deterministic_objective": True,
        "objective_direction": "minimize",
    }


def _json_from_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("RWTHGPT response must be a JSON object.")
    return payload


def _model_dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


class RWTHGPTClient:
    """Minimal OpenAI-compatible client for the RWTHGPT endpoint."""

    def __init__(self, model: str) -> None:
        if model not in MODELS_BY_ID:
            raise ValueError(f"Unsupported model: {model}.")
        self.model = model

    def invoke(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        estimated_tokens = math.ceil(len(prompt) / 4)
        if estimated_tokens > MAX_INPUT_TOKENS:
            raise RuntimeError(
                f"Prompt estimate {estimated_tokens} exceeds {MAX_INPUT_TOKENS}."
            )
        try:
            api_key = RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(
                f"Could not read RWTHGPT key from {RWTHGPT_API_KEY_FILE}."
            ) from error
        if not api_key:
            raise RuntimeError("RWTHGPT API-key file is empty.")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=RWTHGPT_BASE_URL,
            timeout=180.0,
            # Retry explicitly below so a 429 does not fail a whole SMAC run.
            max_retries=0,
        )
        started = time.perf_counter()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are selecting hyperparameters for a random-forest "
                    "surrogate in SMAC. Return only the required JSON object."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        for attempt in range(8):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                break
            except openai.RateLimitError:
                if attempt == 7:
                    raise
                delay_seconds = min(120.0, 5.0 * (2**attempt)) + random.uniform(0.0, 1.0)
                print(
                    f"RWTHGPT rate limited for {self.model}; retrying in "
                    f"{delay_seconds:.1f} seconds (attempt {attempt + 2}/8)."
                )
                time.sleep(delay_seconds)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("RWTHGPT returned an empty completion.")
        return _json_from_content(content), {
            "response_id": response.id,
            "model_requested": self.model,
            "model_returned": response.model,
            "elapsed_seconds": time.perf_counter() - started,
            "usage": _model_dump(response.usage),
            "input_token_estimate": estimated_tokens,
            "api_base_url": RWTHGPT_BASE_URL,
        }


def validate_decision(payload: dict[str, Any]) -> tuple[RFSettings, dict[str, Any]]:
    return compact.validate_decision(payload)


def _prompt(instruction: str, data: dict[str, Any]) -> str:
    return f"""{instruction}

Choose the next random-forest settings only from these ranges:
- n_trees: integer in [1, 100]
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: real number in (0, 1]

Return exactly one JSON object with these keys and no others:
{{"n_trees": integer, "max_depth": integer, "min_samples_split": integer,
  "min_samples_leaf": integer, "feature_ratio": number,
  "confidence": number from 0 to 1, "reason": concise string}}

DATA
{json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)}
"""


def initial_choice_prompt(dimension: int) -> str:
    return _prompt(
        """Before any evaluations, select one fixed random-forest surrogate
configuration for the entire upcoming SMAC run. No objective observations are
available. Use only the known configuration-space information below. The
objective is deterministic and lower values are better.""",
        {
            "selection_mode": "before_first_evaluation_fixed_for_entire_run",
            "configuration_space": configuration_space_summary(dimension),
            "n_instances": N_INSTANCES,
            "n_trials": N_TRIALS,
            "initial_random_configurations": N_INITIAL_CONFIGS,
        },
    )


def dynamic_prompt(summary: dict[str, Any]) -> str:
    return _prompt(
        """You are adjusting the random-forest surrogate for the remaining
phase of a SMAC algorithm-configuration run. Lower objective values are
better. The supplied data aggregate all completed trials into windows and
include acquisition diagnostics, prediction diagnostics, RF-fit diagnostics,
the current settings, and any earlier decisions. Choose settings expected to
improve the remaining optimization run.""",
        summary,
    )


class O6DynamicCallback(base.LLMRFPolicyCallback):
    def __init__(self, *, model_name: str, dimension: int, **kwargs: Any) -> None:
        self.model_name = model_name
        self.dimension = dimension
        super().__init__(
            settings_class=RFSettings,
            decision_validator=validate_decision,
            prompt_builder=dynamic_prompt,
            policy_version=POLICY_VERSION,
            **kwargs,
        )

    def _summary(
        self, checkpoint: int, trigger_trial: int, runhistory: Any
    ) -> dict[str, Any]:
        summary = compact.compact_summary(
            checkpoint=checkpoint,
            trigger_trial=trigger_trial,
            runhistory=runhistory,
            telemetry_path=self.telemetry_path,
            current_settings=self.next_settings,
            decisions=self.state["decisions"],
            fit_observations=self.state["fit_observations"],
            objective_dimension=self.dimension,
        )
        summary["configuration_space"] = configuration_space_summary(self.dimension)
        summary["initial_random_configurations"] = N_INITIAL_CONFIGS
        summary["llm_model"] = self.model_name
        summary["context_limit_input_tokens"] = MAX_INPUT_TOKENS
        summary.pop("optimization_search_space", None)
        return summary

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        audit = super().audit(n_trials=n_trials)
        audit["model"] = self.model_name
        audit["api"] = "RWTHGPT OpenAI-compatible chat completions"
        audit["api_key_source"] = str(RWTHGPT_API_KEY_FILE)
        return audit


def _validate_common(dimension: int, smac_seed: int) -> None:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}.")
    if smac_seed not in FIXED_SEEDS:
        raise ValueError(f"smac_seed must be one of {FIXED_SEEDS}.")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(f"Expected PYTHONHASHSEED={PYTHONHASHSEED}.")


def _make_problem(dimension: int) -> tuple[Any, dict[str, float]]:
    problem_cfg = OmegaConf.load(PROBLEM_CONFIG)
    problem_cfg.problem.function.seed = BENCHMARK_SEED
    problem_cfg.problem.function.dim = dimension
    problem_cfg.task.dimensions = dimension
    problem_cfg.task.search_space_n_floats = dimension
    problem = make_problem(problem_cfg)
    instance_map = base.make_instance_map(N_INSTANCES, INSTANCE_SEED)
    problem.set_instances(instance_map)
    return problem, instance_map


def _result(
    *,
    identity: dict[str, Any],
    policy: str,
    smac: Any,
    incumbent: Any,
    problem: Any,
    instance_map: dict[str, float],
    walltime: float,
    dynamic_callback: O6DynamicCallback | None = None,
) -> dict[str, Any]:
    trials = base.ordered_trials(smac.runhistory)
    costs = [float(np.asarray(value.cost).reshape(-1)[0]) for _, value in trials]
    objective_values = [
        float(np.asarray(value.cost).reshape(-1)[0]) - instance_map[key.instance]
        for key, value in trials
    ]
    f_min = float(problem.f_min)
    regret = [value - f_min for value in objective_values]
    trials_per_config = Counter(int(key.config_id) for key, _ in trials)
    result = {
        **identity,
        "benchmark": "SynthACticBench",
        "problem": "O6-Multimodal",
        "policy": policy,
        "instance_map": instance_map,
        "finished_trials": len(trials),
        "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)),
        "iteration": list(range(1, len(trials) + 1)),
        "cost": costs,
        "objective_value": objective_values,
        "f_min": f_min,
        "regret": regret,
        "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": np.minimum.accumulate(objective_values).astype(float).tolist(),
        "trials_per_config": {
            str(key): value for key, value in sorted(trials_per_config.items())
        },
        "walltime_seconds_this_process": walltime,
    }
    if dynamic_callback is not None:
        result["llm_policy"] = dynamic_callback.audit(N_TRIALS)
    return result


def _run(
    *,
    dimension: int,
    smac_seed: int,
    kind: str,
    settings: RFSettings,
    identity: dict[str, Any],
    model_name: str | None = None,
    initial_choice: dict[str, Any] | None = None,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_path = output_directory(dimension, kind, smac_seed, model_name, output_root)
    completion_path = output_path / "completed.json"
    trajectory_path = output_path / "trajectory.json"
    if completion_path.exists() and trajectory_path.exists():
        completion = base._read_json(completion_path)
        if completion.get("state") == "complete" and completion.get("identity") == identity:
            print(f"Skipping complete run {output_path}.")
            return base._read_json(trajectory_path)

    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and base._read_json(identity_path) != identity:
        raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(identity_path, identity)
    resume = base._resume_state_is_valid(output_path)
    problem, instance_map = _make_problem(dimension)

    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        cost = np.asarray(
            problem.evaluate(TrialInfo(config=config, instance=instance, seed=seed)).cost,
            dtype=float,
        ).reshape(-1)
        if cost.size != 1:
            raise ValueError(f"Expected one O6 objective value, got {cost}")
        return float(cost[0])

    scenario_root = dimension_root(dimension, output_root) / f"benchmark_seed_{BENCHMARK_SEED}"
    policy = policy_name(kind, model_name)
    scenario = base.Scenario(
        name=policy,
        output_directory=scenario_root,
        configspace=problem.configspace,
        deterministic=True,
        instances=list(instance_map),
        n_trials=N_TRIALS,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != output_path:
        raise RuntimeError(f"Unexpected output {scenario.output_directory}; expected {output_path}.")
    model = base.ACFacade.get_model(
        scenario=scenario,
        n_trees=settings.n_trees,
        ratio_features=settings.feature_ratio,
        min_samples_split=settings.min_samples_split,
        min_samples_leaf=settings.min_samples_leaf,
        max_depth=settings.max_depth,
        pca_components=PCA_COMPONENTS,
    )
    initial_design = RandomInitialDesign(
        scenario=scenario, n_configs=N_INITIAL_CONFIGS, seed=smac_seed
    )
    random_design = base.ACFacade.get_random_design(
        scenario=scenario, probability=RANDOM_DESIGN_PROBABILITY
    )
    dynamic_callback = None
    if kind == "dynamic":
        assert model_name is not None
        dynamic_callback = O6DynamicCallback(
            output_path=output_path,
            model=model,
            decision_provider=RWTHGPTClient(model_name).invoke,
            overwrite=not resume,
            initial_settings=settings,
            model_name=model_name,
            dimension=dimension,
        )
    smac = base.ACFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        initial_design=initial_design,
        random_design=random_design,
        callbacks=[] if dynamic_callback is None else [dynamic_callback],
        overwrite=not resume,
    )
    base.atomic_write_json(
        output_path / "run_metadata.json",
        {
            **identity,
            "output_directory": str(output_path),
            "configuration_space": configuration_space_summary(dimension),
            "instance_map": instance_map,
            "initial_choice": initial_choice,
            "local_smac_root": str(base.LOCAL_SMAC_ROOT),
            "api_key_persisted_in_output": False,
        },
    )
    base.atomic_write_json(completion_path, {"state": "running", "identity": identity})
    started = time.time()
    incumbent = smac.optimize()
    walltime = time.time() - started
    result = _result(
        identity=identity,
        policy=policy,
        smac=smac,
        incumbent=incumbent,
        problem=problem,
        instance_map=instance_map,
        walltime=walltime,
        dynamic_callback=dynamic_callback,
    )
    base.atomic_write_json(trajectory_path, result)
    base.atomic_write_json(completion_path, {"state": "complete", "identity": identity})
    print(f"Completed dimension={dimension}, policy={policy}, smac_seed={smac_seed}.")
    return result


def run_fixed_100_trees(dimension: int, smac_seed: int) -> dict[str, Any]:
    _validate_common(dimension, smac_seed)
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "o6_fixed_100_trees",
        "benchmark_seed": BENCHMARK_SEED,
        "dimension": dimension,
        "smac_seed": smac_seed,
        "n_instances": N_INSTANCES,
        "n_trials": N_TRIALS,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": PYTHONHASHSEED,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "pca_components": PCA_COMPONENTS,
        "rf_settings": FIXED_100_TREE_SETTINGS.to_dict(),
    }
    return _run(
        dimension=dimension,
        smac_seed=smac_seed,
        kind="fixed_100_trees",
        settings=FIXED_100_TREE_SETTINGS,
        identity=identity,
    )


def _initial_choice(
    dimension: int, smac_seed: int, model_name: str, output_path: Path
) -> tuple[RFSettings, dict[str, Any]]:
    choice_directory = output_path / "initial_choice"
    validated_path = choice_directory / "validated_decision.json"
    metadata_path = choice_directory / "rwthgpt_response_metadata.json"
    if validated_path.exists():
        settings, normalized = validate_decision(base._read_json(validated_path))
        metadata = base._read_json(metadata_path) if metadata_path.exists() else {}
        return settings, {"settings": settings.to_dict(), "decision": normalized, "metadata": metadata}
    prompt = initial_choice_prompt(dimension)
    choice_directory.mkdir(parents=True, exist_ok=True)
    (choice_directory / "prompt.txt").write_text(prompt, encoding="utf-8")
    raw, metadata = RWTHGPTClient(model_name).invoke(prompt)
    base.atomic_write_json(choice_directory / "structured_output.json", raw)
    settings, normalized = validate_decision(raw)
    base.atomic_write_json(validated_path, normalized)
    base.atomic_write_json(metadata_path, metadata)
    return settings, {"settings": settings.to_dict(), "decision": normalized, "metadata": metadata}


def run_initial_choice(dimension: int, smac_seed: int, model_name: str) -> dict[str, Any]:
    _validate_common(dimension, smac_seed)
    if model_name not in MODELS_BY_ID or smac_seed not in MODELS_BY_ID[model_name].smac_seeds:
        raise ValueError(f"smac_seed={smac_seed} is not assigned to {model_name}.")
    output_path = output_directory(dimension, "initial_choice", smac_seed, model_name)
    settings, choice = _initial_choice(dimension, smac_seed, model_name, output_path)
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "o6_llm_initial_choice",
        "benchmark_seed": BENCHMARK_SEED,
        "dimension": dimension,
        "smac_seed": smac_seed,
        "n_instances": N_INSTANCES,
        "n_trials": N_TRIALS,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": PYTHONHASHSEED,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "pca_components": PCA_COMPONENTS,
        "rwthgpt_model": model_name,
        "rf_settings": settings.to_dict(),
    }
    return _run(
        dimension=dimension,
        smac_seed=smac_seed,
        kind="initial_choice",
        model_name=model_name,
        settings=settings,
        identity=identity,
        initial_choice=choice,
    )


def run_dynamic(dimension: int, smac_seed: int, model_name: str) -> dict[str, Any]:
    _validate_common(dimension, smac_seed)
    if model_name not in MODELS_BY_ID or smac_seed not in MODELS_BY_ID[model_name].smac_seeds:
        raise ValueError(f"smac_seed={smac_seed} is not assigned to {model_name}.")
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment": "o6_llm_dynamic",
        "benchmark_seed": BENCHMARK_SEED,
        "dimension": dimension,
        "smac_seed": smac_seed,
        "n_instances": N_INSTANCES,
        "n_trials": N_TRIALS,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": PYTHONHASHSEED,
        "initial_design": "random",
        "n_initial_configs": N_INITIAL_CONFIGS,
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "pca_components": PCA_COMPONENTS,
        "rwthgpt_model": model_name,
        "checkpoints": list(base.CHECKPOINTS),
        "initial_rf_settings": DEFAULT_SETTINGS.to_dict(),
        "rf_setting_ranges": {key: list(value) for key, value in compact.RANGES.items()},
    }
    return _run(
        dimension=dimension,
        smac_seed=smac_seed,
        kind="dynamic",
        model_name=model_name,
        settings=DEFAULT_SETTINGS,
        identity=identity,
    )


def fixed_jobs() -> tuple[tuple[int, int], ...]:
    return tuple((dimension, seed) for dimension in DIMENSIONS for seed in FIXED_SEEDS)


def llm_jobs(kind: str) -> tuple[tuple[int, int, str], ...]:
    if kind not in {"initial_choice", "dynamic"}:
        raise ValueError(f"Unsupported LLM job kind: {kind}.")
    return tuple(
        (dimension, seed, model.identifier)
        for dimension in DIMENSIONS
        for model in MODELS
        for seed in model.smac_seeds
    )

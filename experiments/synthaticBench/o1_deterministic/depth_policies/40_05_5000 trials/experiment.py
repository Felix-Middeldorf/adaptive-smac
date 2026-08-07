"""5000-trial deterministic-O1 RF-depth and adaptive-policy study.

The module deliberately reuses the local SMAC runner used by the preceding
O1 studies.  All policy decisions are persisted next to the SMAC output, so a
requeued Submitit worker resumes without issuing a duplicate API request.
"""

from __future__ import annotations

import functools
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai


HERE = Path(__file__).resolve().parent
SHARED_POLICY_CODE = HERE.parent / "40_llm_chooses"
if str(SHARED_POLICY_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_POLICY_CODE))

import o1_compact_llm_runner as compact
import o1_llm_runner as base


BENCHMARK_SEED = 40
DIMENSIONS = (10, 25, 50, 100)
SMAC_SEEDS = tuple(range(10))
N_INSTANCES = 10
N_TRIALS = 5_000
INSTANCE_SEED = 0
PYTHONHASHSEED = "12345"
FIXED_N_TREES = 100
FIXED_DEPTHS = (5, 10, 20, 30, 20_000)
CHECKPOINTS = (100, 250, 500, 1_000, 1_500, 2_000, 2_500, 3_000, 3_500, 4_000, 4_500)
HOLDOUT_CHECKPOINTS = tuple(range(500, N_TRIALS, 500))

MISTRAL_MODEL = "mistralai-mistral-small-4-119b"
RWTHGPT_BASE_URL = "https://chat.kiconnect.nrw/api/v1"
RWTHGPT_API_KEY_FILE = Path.home() / ".config/kiconnect/rwthgpt_api_key"
MAX_INPUT_TOKENS = 200_000
RATE_LIMIT_WAIT_SECONDS = 15

FIXED_POLICY_PREFIX = "fixed_depth"
MISTRAL_POLICY_NAME = "rwthgpt_mistral_compact_fixed_100_trees_5000"
HOLDOUT_POLICY_NAME = "chronological_holdout_rf_selection_5000"
EXPERIMENT_VERSION = 1
POLICY_VERSION = 1
OUTPUT_ROOT = HERE / "results"

# The shared runner was designed for the earlier 1,000-trial, five-seed
# experiment.  Its run machinery is parameterised, while these two globals
# define when its callback fires and which seeds it accepts.
base.CHECKPOINTS = CHECKPOINTS
base.SMAC_SEEDS = SMAC_SEEDS


@dataclass(frozen=True)
class RFSettings:
    n_trees: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    feature_ratio: float

    def __post_init__(self) -> None:
        if self.n_trees != FIXED_N_TREES:
            raise ValueError(f"n_trees must be fixed to {FIXED_N_TREES}.")
        if not 1 <= self.max_depth <= 20_000:
            raise ValueError("max_depth must be in [1, 20000].")
        if not 2 <= self.min_samples_split <= 10:
            raise ValueError("min_samples_split must be in [2, 10].")
        if not 1 <= self.min_samples_leaf <= 10:
            raise ValueError("min_samples_leaf must be in [1, 10].")
        if not math.isfinite(self.feature_ratio) or not 0 < self.feature_ratio <= 1:
            raise ValueError("feature_ratio must be in (0, 1].")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "n_trees": self.n_trees,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "feature_ratio": self.feature_ratio,
        }

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RFSettings":
        return cls(
            n_trees=int(value["n_trees"]),
            max_depth=int(value["max_depth"]),
            min_samples_split=int(value["min_samples_split"]),
            min_samples_leaf=int(value["min_samples_leaf"]),
            feature_ratio=float(value["feature_ratio"]),
        )


LLM_INITIAL_SETTINGS = RFSettings(100, 20, 3, 3, 5.0 / 6.0)
HOLDOUT_INITIAL_SETTINGS = RFSettings(100, 20_000, 2, 1, 5.0 / 6.0)


def dimension_root(dimension: int, output_root: Path = OUTPUT_ROOT) -> Path:
    return Path(output_root) / f"dimension_{dimension}"


def fixed_policy_name(depth: int) -> str:
    return f"{FIXED_POLICY_PREFIX}_{depth}_100_trees_split_2_leaf_1"


def _model_dump(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def _json_from_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("RWTHGPT response must be a JSON object.")
    return parsed


RAW_DECISION_FIELDS = {
    "max_depth", "min_samples_split", "min_samples_leaf", "feature_ratio",
    "confidence", "reason",
}
CACHED_DECISION_FIELDS = RAW_DECISION_FIELDS | {"n_trees"}


def validate_mistral_decision(payload: dict[str, Any]) -> tuple[RFSettings, dict[str, Any]]:
    fields = set(payload)
    if fields == CACHED_DECISION_FIELDS:
        if int(payload["n_trees"]) != FIXED_N_TREES:
            raise ValueError(f"Cached n_trees must remain {FIXED_N_TREES}.")
    elif fields != RAW_DECISION_FIELDS:
        raise ValueError("Mistral decision has missing or additional fields.")
    settings = RFSettings(
        n_trees=FIXED_N_TREES,
        max_depth=int(payload["max_depth"]),
        min_samples_split=int(payload["min_samples_split"]),
        min_samples_leaf=int(payload["min_samples_leaf"]),
        feature_ratio=float(payload["feature_ratio"]),
    )
    # Mistral may only select from the historical compact-policy range.
    if settings.max_depth > 30:
        raise ValueError("Mistral max_depth must be in [1, 30].")
    confidence = float(payload["confidence"])
    reason = str(payload["reason"]).strip()
    if not 0 <= confidence <= 1 or not reason or len(reason) > 500:
        raise ValueError("Invalid confidence or reason.")
    return settings, {**settings.to_dict(), "confidence": confidence, "reason": reason}


class RWTHGPTMistralClient:
    """Retry only the transient malformed/429 responses inside each job."""

    def invoke(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        estimate = math.ceil(len(prompt) / 4)
        if estimate > MAX_INPUT_TOKENS:
            raise RuntimeError(f"Prompt estimate {estimate} exceeds {MAX_INPUT_TOKENS}.")
        key = RWTHGPT_API_KEY_FILE.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"RWTHGPT key file is empty: {RWTHGPT_API_KEY_FILE}")
        client = openai.OpenAI(api_key=key, base_url=RWTHGPT_BASE_URL, timeout=300.0, max_retries=0)
        messages = [
            {"role": "system", "content": "You select SMAC RF hyperparameters. Return only the requested JSON."},
            {"role": "user", "content": prompt},
        ]
        started = time.perf_counter()
        retries = 0
        while True:
            try:
                response = client.chat.completions.create(model=MISTRAL_MODEL, messages=messages)
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty completion")
                payload = _json_from_content(content)
                validate_mistral_decision(payload)
                return payload, {
                    "response_id": response.id,
                    "model_requested": MISTRAL_MODEL,
                    "model_returned": response.model,
                    "elapsed_seconds": time.perf_counter() - started,
                    "usage": _model_dump(response.usage),
                    "input_token_estimate": estimate,
                    "rate_limit_retries": retries,
                    "api_base_url": RWTHGPT_BASE_URL,
                }
            except openai.RateLimitError:
                message = "RWTHGPT returned 429"
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                message = f"RWTHGPT invalid completion: {error}"
            print(f"{message}; retrying in {RATE_LIMIT_WAIT_SECONDS}s (attempt {retries + 1}).")
            retries += 1
            time.sleep(RATE_LIMIT_WAIT_SECONDS)


def _short(value: float | int | None) -> float | int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(f"{float(value):.7g}")


def _capacity_and_calibration(
    *, fit_observations: list[dict[str, Any]], diagnostics: list[dict[str, Any]]
) -> dict[str, Any]:
    """Capacity of the latest fitted forests and calibration of online predictions."""
    recent = fit_observations[-50:]
    def median_field(name: str) -> float | None:
        values = [float(item[name]) for item in recent if item.get(name) is not None]
        return _short(np.median(values)) if values else None

    capped = [float(item["fraction_trees_exactly_at_depth_cap"]) for item in recent
              if item.get("fraction_trees_exactly_at_depth_cap") is not None]
    matched: list[tuple[float, float]] = []
    for row in diagnostics:
        error, variance = row.get("absolute_proxy_error"), row.get("prediction_variance")
        if error is None or variance is None or float(variance) <= 0:
            continue
        standard_deviation = math.sqrt(float(variance))
        if math.isfinite(float(error)) and math.isfinite(standard_deviation) and standard_deviation > 0:
            matched.append((abs(float(error)), standard_deviation))
    standardized = np.asarray([error / std for error, std in matched], dtype=float)
    correlation = None
    if len(matched) >= 3:
        errors, stds = np.asarray(matched, dtype=float).T
        if np.std(errors) > 0 and np.std(stds) > 0:
            correlation = _short(np.corrcoef(errors, stds)[0, 1])
    return {
        "fit_window_count": len(recent),
        "fraction_trees_exactly_at_depth_cap": _short(float(np.mean(capped))) if capped else None,
        "tree_depth_q10": median_field("tree_depth_q10"),
        "tree_depth_median": median_field("tree_depth_median"),
        "tree_depth_q90": median_field("tree_depth_q90"),
        "matched_online_error_count": int(len(matched)),
        "median_standardized_error": _short(float(np.median(standardized))) if standardized.size else None,
        "q90_standardized_error": _short(float(np.quantile(standardized, .9))) if standardized.size else None,
        "coverage_within_1_std": _short(float(np.mean(standardized <= 1))) if standardized.size else None,
        "coverage_within_2_std": _short(float(np.mean(standardized <= 2))) if standardized.size else None,
        "error_std_correlation": correlation,
    }


def _mistral_prompt(summary: dict[str, Any]) -> str:
    return f"""Choose random-forest surrogate hyperparameters for the next phase of a
SMAC algorithm-configuration run. Lower objective values are better. The
number of trees is fixed to {FIXED_N_TREES} and must not be proposed or changed.
The input aggregates all completed trials into ten chronological windows; it
also states the continuous objective-space dimensionality and bounds.

Choose only from:
- max_depth: integer in [1, 30]
- min_samples_split: integer in [2, 10]
- min_samples_leaf: integer in [1, 10]
- feature_ratio: number in (0, 1]

Return exactly one JSON object with these keys and no others:
{{"max_depth": integer, "min_samples_split": integer,
  "min_samples_leaf": integer, "feature_ratio": number,
  "confidence": number from 0 to 1, "reason": concise string}}

The capacity_and_calibration block is important: a high fraction at the depth
cap means trees are genuinely constrained, whereas deep but uncapped trees do
not. Standardized error is absolute proposal-time proxy error divided by the
predicted standard deviation. Coverage is the fraction within one/two predicted
standard deviations. error_std_correlation relates absolute error to predicted
standard deviation: low or negative values can indicate confidently wrong
predictions. Use these with progress, EI, and the current settings.

DATA
{json.dumps(summary, sort_keys=True, separators=(',', ':'), allow_nan=False)}
"""


class ExtendedMistralCallback(base.LLMRFPolicyCallback):
    def __init__(self, *, dimension: int, **kwargs: Any) -> None:
        self.dimension = dimension
        super().__init__(
            settings_class=RFSettings,
            decision_validator=validate_mistral_decision,
            prompt_builder=_mistral_prompt,
            policy_version=POLICY_VERSION,
            **kwargs,
        )

    def _install_controlled_train(self) -> None:
        original_train = self.model.train

        @functools.wraps(original_train)
        def controlled_train(*args: Any, **kwargs: Any) -> Any:
            settings = self.next_settings
            self._apply_settings(settings)
            data = args[0] if args else kwargs.get("X")
            started = time.perf_counter()
            result = original_train(*args, **kwargs)
            depths = np.asarray([tree.get_depth() for tree in self.model._rf.estimators_], dtype=float)
            observation = {
                "fit_index": len(self.state["fit_observations"]),
                "training_rows": -1 if data is None else int(len(data)),
                "fit_duration_seconds": time.perf_counter() - started,
                "settings": settings.to_dict(),
                "actual_tree_depth_mean": float(np.mean(depths)),
                "actual_tree_depth_min": int(np.min(depths)),
                "actual_tree_depth_max": int(np.max(depths)),
                "depth_utilization": float(np.mean(depths)) / settings.max_depth,
                "tree_depth_q10": float(np.quantile(depths, .1)),
                "tree_depth_median": float(np.median(depths)),
                "tree_depth_q90": float(np.quantile(depths, .9)),
                "fraction_trees_exactly_at_depth_cap": float(np.mean(depths == settings.max_depth)),
            }
            self.state["fit_observations"].append(observation)
            self._save_state()
            return result
        self.model.train = controlled_train

    def _summary(self, checkpoint: int, trigger_trial: int, runhistory: Any) -> dict[str, Any]:
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
        diagnostics = compact._trial_diagnostics(runhistory, self.telemetry_path, checkpoint)
        summary["fixed_rf_hyperparameters"] = {"n_trees": FIXED_N_TREES}
        summary["allowed_next_settings"].pop("n_trees")
        summary["capacity_and_calibration"] = _capacity_and_calibration(
            fit_observations=self.state["fit_observations"], diagnostics=diagnostics
        )
        return summary

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        audit = super().audit(n_trials)
        audit.update({"model": MISTRAL_MODEL, "api": "RWTHGPT", "fixed_n_trees": FIXED_N_TREES})
        return audit


class FixedSettingsCallback(base.LLMRFPolicyCallback):
    """The common callback installation/persistence path without API calls."""
    def __init__(self, *, label: str, **kwargs: Any) -> None:
        self.label = label
        super().__init__(
            settings_class=RFSettings,
            decision_validator=lambda value: (_ for _ in ()).throw(RuntimeError("fixed policy has no decisions")),
            prompt_builder=lambda value: "",
            policy_version=POLICY_VERSION,
            **kwargs,
        )

    def on_next_configurations_start(self, config_selector: Any) -> None:
        return None

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        return {"policy_type": "fixed_rf", "label": self.label, "settings": self.next_settings.to_dict(), "rf_fit_count": len(self.state["fit_observations"])}


class ChronologicalHoldoutCallback(FixedSettingsCallback):
    """Select leaf size/feature ratio by a chronological 80/20 holdout."""
    CANDIDATES = tuple(RFSettings(100, 20_000, 2, leaf, ratio) for leaf in (1, 2, 3) for ratio in (.75, .9, 1.0))

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(label="chronological_80_20_holdout", **kwargs)

    def _install_controlled_train(self) -> None:
        original_train = self.model.train
        self._uncontrolled_train = original_train

        @functools.wraps(original_train)
        def controlled_train(*args: Any, **kwargs: Any) -> Any:
            x = np.asarray(args[0] if args else kwargs["X"])
            y = np.asarray(args[1] if len(args) > 1 else kwargs["Y"])
            return self._fit_selected_model(x, y)

        self.model.train = controlled_train

    def _fit_selected_model(self, x: np.ndarray, y: np.ndarray) -> Any:
        """Fit the live surrogate immediately after a checkpoint choice."""
        self._apply_settings(self.next_settings)
        started = time.perf_counter()
        result = self._uncontrolled_train(x, y)
        depths = np.asarray(
            [tree.get_depth() for tree in self.model._rf.estimators_], dtype=float
        )
        self.state["fit_observations"].append(
            {
                "fit_index": len(self.state["fit_observations"]),
                "training_rows": int(len(x)),
                "fit_duration_seconds": time.perf_counter() - started,
                "settings": self.next_settings.to_dict(),
                "actual_tree_depth_mean": float(np.mean(depths)),
                "actual_tree_depth_min": int(np.min(depths)),
                "actual_tree_depth_max": int(np.max(depths)),
                "depth_utilization": float(np.mean(depths)) / self.next_settings.max_depth,
            }
        )
        self._save_state()
        return result

    def _select_candidate(
        self, x: np.ndarray, y: np.ndarray, checkpoint: int, trigger_trial: int
    ) -> None:
        key = str(checkpoint)
        if key in self.state["decisions"]:
            self.next_settings = RFSettings.from_mapping(self.state["decisions"][key]["settings"])
            return
        split = max(1, min(len(x) - 1, int(math.floor(.8 * len(x)))))
        scores: list[dict[str, Any]] = []
        for candidate_settings in self.CANDIDATES:
            candidate = base.ACFacade.get_model(
                scenario=self._scenario,
                n_trees=candidate_settings.n_trees,
                ratio_features=candidate_settings.feature_ratio,
                min_samples_split=candidate_settings.min_samples_split,
                min_samples_leaf=candidate_settings.min_samples_leaf,
                max_depth=candidate_settings.max_depth,
                pca_components=base.PCA_COMPONENTS,
            )
            candidate.train(x[:split], y[:split])
            predicted, _ = candidate.predict(x[split:])
            score = float(np.mean((np.asarray(predicted).reshape(-1) - y[split:].reshape(-1)) ** 2))
            scores.append({"settings": candidate_settings.to_dict(), "validation_mse": score})
        winner = min(scores, key=lambda item: (item["validation_mse"], item["settings"]["min_samples_leaf"], item["settings"]["feature_ratio"]))
        self.next_settings = RFSettings.from_mapping(winner["settings"])
        decision = {
            "checkpoint": checkpoint,
            "actual_completed_trials_at_selection": trigger_trial,
            "training_rows": int(len(x)),
            "earliest_training_rows": int(split),
            "latest_validation_rows": int(len(x) - split),
            "candidate_validation_mse": scores,
            "settings": self.next_settings.to_dict(),
        }
        self.state["decisions"][key] = decision
        self.state["current_settings"] = self.next_settings.to_dict()
        self.state["transitions"].append({"completed_trials": checkpoint, "source": "chronological_80_20_holdout", "checkpoint": checkpoint, "settings": self.next_settings.to_dict()})
        base.append_jsonl(self.events_path, {"event_type": "holdout_selection", **decision})
        self._save_state()
        print(f"[HoldoutRF] checkpoint={checkpoint}; selected {self.next_settings.to_dict()}")

    def on_next_configurations_start(self, config_selector: Any) -> None:
        """Choose once SMAC normally returns to train/propose after a checkpoint."""
        completed = int(getattr(config_selector._runhistory, "finished", len(config_selector._runhistory)))
        encoder = config_selector._runhistory_encoder
        if encoder is None:
            raise RuntimeError("SMAC has no runhistory encoder.")
        for checkpoint in HOLDOUT_CHECKPOINTS:
            if checkpoint <= completed and str(checkpoint) not in self.state["decisions"]:
                # No extra fit is performed here.  The ordinary SMAC training
                # call immediately following this callback consumes these new
                # settings; it may occur a few trials after the nominal checkpoint.
                x, y = (np.asarray(item) for item in encoder.transform())
                self._select_candidate(x, y, checkpoint, completed)

    def audit(self, n_trials: int = N_TRIALS) -> dict[str, Any]:
        missing = [cp for cp in HOLDOUT_CHECKPOINTS if cp < n_trials and str(cp) not in self.state["decisions"]]
        if missing:
            raise RuntimeError(f"Completed holdout run lacks selections {missing}.")
        return {"policy_type": "chronological_80_20_holdout", "checkpoints": list(HOLDOUT_CHECKPOINTS), "candidates": [candidate.to_dict() for candidate in self.CANDIDATES], "decisions": self.state["decisions"], "rf_fit_count": len(self.state["fit_observations"])}


def _run(
    dimension: int, smac_seed: int, *, policy_name: str, initial_settings: RFSettings,
    callback_type: type[Any], callback_extra: dict[str, Any] | None = None,
    decision_provider: Callable[[str], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if dimension not in DIMENSIONS or smac_seed not in SMAC_SEEDS:
        raise ValueError(f"dimension={dimension}, smac_seed={smac_seed} outside experiment design")
    if callback_type is ChronologicalHoldoutCallback:
        # See _holdout_run below; this branch is intentionally not used.
        raise RuntimeError("Use _holdout_run for chronological holdout policy.")
    provider = decision_provider or (RWTHGPTMistralClient().invoke if callback_type is ExtendedMistralCallback else lambda _: ({}, {}))
    return base.run_llm_policy(
        BENCHMARK_SEED, smac_seed, n_trials=N_TRIALS, output_root=dimension_root(dimension),
        decision_provider=provider, policy_name=policy_name, experiment_version=EXPERIMENT_VERSION,
        callback_factory=functools.partial(callback_type, **(callback_extra or {})),
        identity_extra={"study": "40_05_5000_trials", "fixed_n_trees": FIXED_N_TREES, "rf_policy_type": policy_name},
        dimension=dimension, n_instances=N_INSTANCES, instance_seed=INSTANCE_SEED, initial_settings=initial_settings,
    )


def run_fixed(dimension: int, smac_seed: int, depth: int) -> dict[str, Any]:
    if depth not in FIXED_DEPTHS:
        raise ValueError(f"depth must be one of {FIXED_DEPTHS}")
    return _run(dimension, smac_seed, policy_name=fixed_policy_name(depth), initial_settings=RFSettings(100, depth, 2, 1, 5.0 / 6.0), callback_type=FixedSettingsCallback, callback_extra={"label": f"fixed depth {depth}"})


def run_mistral_compact(dimension: int, smac_seed: int) -> dict[str, Any]:
    return _run(dimension, smac_seed, policy_name=MISTRAL_POLICY_NAME, initial_settings=LLM_INITIAL_SETTINGS, callback_type=ExtendedMistralCallback, callback_extra={"dimension": dimension})


def run_holdout(dimension: int, smac_seed: int) -> dict[str, Any]:
    """Run the chronological holdout policy, including requeue-safe state."""
    if dimension not in DIMENSIONS or smac_seed not in SMAC_SEEDS:
        raise ValueError(f"dimension={dimension}, smac_seed={smac_seed} outside experiment design")
    if os.environ.get("PYTHONHASHSEED") != PYTHONHASHSEED:
        raise RuntimeError(f"Expected PYTHONHASHSEED={PYTHONHASHSEED}.")
    output_root = dimension_root(dimension).resolve()
    output_path = base.output_directory(output_root, BENCHMARK_SEED, smac_seed, HOLDOUT_POLICY_NAME)
    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "study": "40_05_5000_trials",
        "problem": "O1-DeterministicObjective",
        "benchmark_seed": BENCHMARK_SEED,
        "smac_seed": smac_seed,
        "instance_seed": INSTANCE_SEED,
        "pythonhashseed": PYTHONHASHSEED,
        "dimension": dimension,
        "n_instances": N_INSTANCES,
        "n_trials": N_TRIALS,
        "deterministic": True,
        "checkpoints": list(HOLDOUT_CHECKPOINTS),
        "default_rf_settings": HOLDOUT_INITIAL_SETTINGS.to_dict(),
        "fixed_n_trees": FIXED_N_TREES,
        "policy_type": "chronological_80_20_holdout",
        "candidate_grid": [candidate.to_dict() for candidate in ChronologicalHoldoutCallback.CANDIDATES],
    }
    completion_path = output_path / "completed.json"
    trajectory_path = output_path / "trajectory.json"
    if completion_path.exists():
        completion = base._read_json(completion_path)
        if completion.get("state") == "complete" and completion.get("identity") == identity and trajectory_path.exists():
            print(f"Skipping complete run {output_path}.")
            return base._read_json(trajectory_path)
    identity_path = output_path / "run_identity.json"
    if identity_path.exists() and base._read_json(identity_path) != identity:
        raise RuntimeError(f"Existing identity differs in {output_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    base.atomic_write_json(identity_path, identity)
    resume = base._resume_state_is_valid(output_path)

    problem_cfg = base.OmegaConf.load(base.PROBLEM_CONFIG)
    problem_cfg.problem.function.wrapped_bench.seed = BENCHMARK_SEED
    problem_cfg.problem.function.wrapped_bench.dim = dimension
    problem_cfg.task.dimensions = dimension
    problem_cfg.task.search_space_n_floats = dimension
    problem = base.make_problem(problem_cfg)
    instance_map = base.make_instance_map(N_INSTANCES, INSTANCE_SEED)
    problem.set_instances(instance_map)
    def target_function(config: Any, instance: str, seed: int = 0) -> float:
        return float(problem.evaluate(base.TrialInfo(config=config, instance=instance, seed=seed)).cost)
    scenario = base.Scenario(
        name=HOLDOUT_POLICY_NAME,
        output_directory=output_root / f"benchmark_seed_{BENCHMARK_SEED}",
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
        scenario=scenario, n_trees=100, ratio_features=HOLDOUT_INITIAL_SETTINGS.feature_ratio,
        min_samples_split=2, min_samples_leaf=1, max_depth=20_000,
        pca_components=base.PCA_COMPONENTS,
    )
    callback = ChronologicalHoldoutCallback(
        output_path=output_path, model=model, decision_provider=lambda _: ({}, {}),
        overwrite=not resume, initial_settings=HOLDOUT_INITIAL_SETTINGS,
    )
    callback._scenario = scenario
    random_design = base.ACFacade.get_random_design(scenario=scenario, probability=base.RANDOM_DESIGN_PROBABILITY)
    smac = base.ACFacade(
        scenario=scenario, target_function=target_function, model=model,
        random_design=random_design, callbacks=[callback], overwrite=not resume,
    )
    base.atomic_write_json(output_path / "run_metadata.json", {**identity, "output_directory": str(output_path), "instance_map": instance_map, "api_key_persisted_in_output": False})
    base.atomic_write_json(completion_path, {"state": "running", "identity": identity})
    started = time.time()
    incumbent = smac.optimize()
    trials = base.ordered_trials(smac.runhistory)
    costs = [float(value.cost) for _, value in trials]
    objective_values = [float(value.cost) - instance_map[key.instance] for key, value in trials]
    regret = [value - float(problem.f_min) for value in objective_values]
    result = {
        **identity, "benchmark": "SynthACticBench", "policy": HOLDOUT_POLICY_NAME,
        "instance_map": instance_map, "finished_trials": len(trials), "incumbent": dict(incumbent),
        "incumbent_cost": float(smac.runhistory.get_cost(incumbent)), "iteration": list(range(1, len(trials) + 1)),
        "cost": costs, "objective_value": objective_values, "f_min": float(problem.f_min),
        "regret": regret, "best_regret": np.minimum.accumulate(regret).astype(float).tolist(),
        "best_so_far": np.minimum.accumulate(objective_values).astype(float).tolist(),
        "walltime_seconds_this_process": time.time() - started, "llm_policy": callback.audit(N_TRIALS),
    }
    base.atomic_write_json(trajectory_path, result)
    base.atomic_write_json(completion_path, {"state": "complete", "identity": identity})
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def jobs() -> tuple[tuple[str, int, int, int | None], ...]:
    fixed = tuple(("fixed", dimension, seed, depth) for dimension in DIMENSIONS for depth in FIXED_DEPTHS for seed in SMAC_SEEDS)
    mistral = tuple(("mistral", dimension, seed, None) for dimension in DIMENSIONS for seed in SMAC_SEEDS)
    holdout = tuple(("holdout", dimension, seed, None) for dimension in DIMENSIONS for seed in SMAC_SEEDS)
    return fixed + mistral + holdout

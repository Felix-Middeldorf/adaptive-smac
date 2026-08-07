"""Run deterministic ACLib experiments with fixed-checkpoint RF schedules."""

from __future__ import annotations

import functools
import inspect
import json
import os
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ConfigSpace import Configuration

from fixed_depth_experiment import (
    LOCAL_SMAC_ROOT,
    AlgorithmConfigurationFacade,
    RandomForest,
    RandomInitialDesign,
    Scenario,
    _acquire_run_lock,
    _atomic_json,
    _package_version,
    _serialize_trajectory,
    load_initial_choice,
    resolve_initial_configuration,
)
from smac.callback import Callback
from rf_schedule_catalog import N_SCHEDULES, RFSchedule, RFSettings, get_schedule
from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    asset_metadata,
    get_benchmark_spec,
    load_benchmark_data,
)
from surrogate_telemetry import (
    TELEMETRY_FILENAME,
    TELEMETRY_SCHEMA_VERSION,
    TELEMETRY_SUMMARY_FILENAME,
    SurrogateTelemetryCallback,
    TelemetryEI,
)


N_TRIALS = 5_000
SMAC_SEEDS = (0, 1, 2)
N_TREES = 100
MIN_SAMPLES_LEAF = 1
PCA_COMPONENTS = 4
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 1
SCHEDULE_EVENTS_FILENAME = "rf_schedule_events.jsonl"
SCHEDULE_STATE_FILENAME = "rf_schedule_state.json"


@dataclass(frozen=True)
class RFScheduleExperimentDefinition:
    benchmark_key: str
    initials: str
    directory: Path
    initial_directory: Path

    @property
    def output_root(self) -> Path:
        return self.directory / "results"

    @property
    def initial_choice_file(self) -> Path:
        return self.initial_directory / "initial_config.json"


def run_directory(
    output_root: Path,
    schedule_index: int,
    smac_seed: int,
) -> Path:
    return output_root / f"schedule_{schedule_index:02d}" / str(smac_seed)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resume_state_is_valid(path: Path) -> bool:
    required = (
        path / "scenario.json",
        path / "configspace.json",
        path / "runhistory.json",
        path / "intensifier.json",
        path / "optimization.json",
    )
    existing = [item for item in required if item.exists()]
    if not existing:
        return False
    if len(existing) != len(required):
        missing = [item.name for item in required if not item.exists()]
        raise RuntimeError(
            f"Partial SMAC checkpoint in {path}; missing {missing}."
        )
    for item in required:
        _read_json(item)
    return True


class FixedCheckpointRFScheduleCallback(Callback):
    """Apply one RF setting per trial phase to every subsequent model fit."""

    def __init__(
        self,
        *,
        schedule: RFSchedule,
        output_directory: Path,
        model: RandomForest,
        overwrite: bool,
    ) -> None:
        super().__init__()
        self.schedule = schedule
        self.output_directory = output_directory
        self.model = model
        self.events_path = output_directory / SCHEDULE_EVENTS_FILENAME
        self.state_path = output_directory / SCHEDULE_STATE_FILENAME
        self.next_settings = schedule.settings_for_trial(0)

        if overwrite:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            self.events_path.write_text("", encoding="utf-8")
            self.state = {
                "schedule_index": schedule.index,
                "active_phase": None,
                "last_fitted_phase": None,
                "last_fitted_settings": None,
                "transitions": [],
            }
            self._save_state()
        elif self.state_path.exists():
            self.state = _read_json(self.state_path)
            if int(self.state["schedule_index"]) != schedule.index:
                raise RuntimeError("Persisted RF schedule index does not match.")
        else:
            raise RuntimeError(
                f"Resume requested without {self.state_path.name}."
            )

        self._next_event_index = (
            sum(1 for _ in self.events_path.open("r", encoding="utf-8"))
            if self.events_path.exists()
            else 0
        )
        self._install_controlled_train()

    def _save_state(self) -> None:
        _atomic_json(self.state_path, self.state)

    def _append_event(self, payload: dict[str, Any]) -> None:
        event = {
            "event_index": self._next_event_index,
            "schedule_index": self.schedule.index,
            **payload,
        }
        serialized = json.dumps(
            event,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_event_index += 1

    def _apply_settings(self, settings: RFSettings) -> None:
        self.model._rf_opts["max_depth"] = settings.depth
        self.model._rf_opts[
            "min_samples_split"
        ] = settings.min_samples_split
        self.model._rf_opts["min_samples_leaf"] = MIN_SAMPLES_LEAF
        self.model._ratio_features = settings.feature_ratio

    def _install_controlled_train(self) -> None:
        original_train = self.model.train

        @functools.wraps(original_train)
        def controlled_train(*args: Any, **kwargs: Any) -> Any:
            settings = self.next_settings
            phase = self.state["active_phase"]
            if phase is None:
                phase = self.schedule.phase_index(0)
            self._apply_settings(settings)
            result = original_train(*args, **kwargs)
            fitted_forest = self.model._rf
            self.state["last_fitted_phase"] = phase
            self.state["last_fitted_settings"] = {
                **settings.to_dict(),
                "actual_max_features": int(
                    fitted_forest.max_features
                ),
                "transformed_feature_count": int(
                    fitted_forest.n_features_in_
                ),
            }
            self._save_state()
            return result

        self.model.train = controlled_train

    def on_next_configurations_start(self, config_selector: Any) -> None:
        completed_trials = len(config_selector._runhistory)
        phase = self.schedule.phase_index(completed_trials)
        settings = self.schedule.phases[phase]
        self.next_settings = settings
        if self.state["active_phase"] == phase:
            return
        transition = {
            "event_type": "rf_schedule_transition",
            "completed_trials": completed_trials,
            "phase": phase,
            "settings": settings.to_dict(),
        }
        self.state["active_phase"] = phase
        self.state["transitions"].append(transition)
        self._append_event(transition)
        self._save_state()
        print(
            f"[RFSchedule] schedule={self.schedule.index:02d} "
            f"trials={completed_trials} phase={phase} "
            f"settings={settings.to_dict()}"
        )


def run_rf_schedule(
    *,
    definition: RFScheduleExperimentDefinition,
    schedule_index: int,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not 0 <= schedule_index < N_SCHEDULES:
        raise ValueError(f"schedule_index must be in [0, {N_SCHEDULES - 1}].")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")

    schedule = get_schedule(schedule_index)
    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    choice, choice_payload = load_initial_choice(
        definition.initial_choice_file
    )
    initial_config = resolve_initial_configuration(data, choice)
    if output_root is None:
        output_root = definition.output_root
    output_root = Path(output_root).resolve()
    output_path = run_directory(output_root, schedule_index, smac_seed)

    identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": spec.key,
        "display_name": spec.display_name,
        "initials": definition.initials,
        "schedule": schedule.to_dict(),
        "smac_seed": smac_seed,
        "n_trials": n_trials,
        "deterministic": True,
        "target_quantile_seed": 0,
        "n_training_instances": len(data.training_instances),
        "test_instances_used": 0,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "initial_configuration_choice": choice_payload,
        "smac_model": {
            "n_trees": N_TREES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "pca_components": PCA_COMPONENTS,
        },
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
        "phase_activation": (
            "The phase changes at the first RF retraining checkpoint whose "
            "completed target-trial count is at least 500 or 2000."
        ),
        "local_smac_root": str(LOCAL_SMAC_ROOT),
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "assets": asset_metadata(spec),
    }

    with closing(_acquire_run_lock(output_path)):
        identity_path = output_path / "run_identity.json"
        completion_path = output_path / "completed.json"
        summary_path = output_path / "summary.json"
        if completion_path.exists() and not overwrite:
            completion = _read_json(completion_path)
            if (
                completion.get("state") == "complete"
                and completion.get("identity") == identity
                and summary_path.exists()
            ):
                print(f"Complete schedule run found; skipping {output_path}")
                return _read_json(summary_path)

        if identity_path.exists() and not overwrite:
            if _read_json(identity_path) != identity:
                raise RuntimeError(
                    f"Existing run identity differs in {output_path}."
                )
        _atomic_json(identity_path, identity)

        resume = False if overwrite else _resume_state_is_valid(output_path)
        training_features = {
            instance: data.features[instance]
            for instance in data.training_instances
        }
        scenario = Scenario(
            configspace=data.configspace,
            name=schedule.name,
            output_directory=output_root,
            deterministic=True,
            objectives="PAR10",
            crash_cost=spec.timeout_cost,
            n_trials=n_trials,
            use_default_config=choice.kind == "default",
            instances=list(data.training_instances),
            instance_features=training_features,
            seed=smac_seed,
            n_workers=1,
        )
        if scenario.output_directory != output_path:
            raise RuntimeError(
                f"Unexpected output path {scenario.output_directory}; "
                f"expected {output_path}."
            )

        if choice.kind == "default":
            initial_design = (
                AlgorithmConfigurationFacade.get_initial_design(scenario)
            )
        else:
            initial_design = RandomInitialDesign(
                scenario,
                n_configs=0,
                additional_configs=[initial_config],
            )

        first = schedule.phases[0]
        model = AlgorithmConfigurationFacade.get_model(
            scenario=scenario,
            n_trees=N_TREES,
            ratio_features=first.feature_ratio,
            max_depth=first.depth,
            min_samples_split=first.min_samples_split,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            pca_components=PCA_COMPONENTS,
        )
        random_design = AlgorithmConfigurationFacade.get_random_design(
            scenario,
            probability=RANDOM_DESIGN_PROBABILITY,
        )
        acquisition_function = TelemetryEI.from_smac_ei(
            AlgorithmConfigurationFacade.get_acquisition_function(scenario)
        )
        benchmark = ACLibSurrogateBenchmark(spec)

        def deterministic_target(
            config: Configuration,
            instance: str,
            seed: int = 0,
        ) -> tuple[float, dict[str, Any]]:
            del seed
            cost, info = benchmark.evaluate(config, instance, seed=0)
            info["fixed_target_quantile_seed"] = 0
            return cost, info

        schedule_callback = FixedCheckpointRFScheduleCallback(
            schedule=schedule,
            output_directory=output_path,
            model=model,
            overwrite=not resume,
        )
        telemetry = SurrogateTelemetryCallback(
            output_directory=output_path,
            model=model,
            acquisition_function=acquisition_function,
            overwrite=not resume,
        )
        facade = AlgorithmConfigurationFacade(
            scenario=scenario,
            target_function=deterministic_target,
            model=model,
            acquisition_function=acquisition_function,
            initial_design=initial_design,
            random_design=random_design,
            callbacks=[schedule_callback, telemetry],
            overwrite=not resume,
        )

        rf_source = Path(inspect.getfile(RandomForest)).resolve()
        if LOCAL_SMAC_ROOT.resolve() not in rf_source.parents:
            raise RuntimeError(f"Expected local SMAC RF, found {rf_source}.")
        metadata = {
            **identity,
            "output_directory": str(output_path),
            "training_instances": list(data.training_instances),
            "training_only": True,
            "local_random_forest_source": str(rf_source),
            "telemetry": {
                "configuration_file": str(
                    output_path / TELEMETRY_FILENAME
                ),
                "summary_file": str(
                    output_path / TELEMETRY_SUMMARY_FILENAME
                ),
                "schedule_events_file": str(
                    output_path / SCHEDULE_EVENTS_FILENAME
                ),
                "prediction": (
                    "RF mean, variance, and standard deviation marginalized "
                    "over every training instance at proposal time"
                ),
                "rf_settings": (
                    "Every proposal records depth, split/leaf sizes, feature "
                    "ratio, integer max_features, PCA components, and tree depths"
                ),
            },
            "versions": {
                "python": sys.version,
                "smac_distribution": _package_version("smac"),
                "ConfigSpace": _package_version("ConfigSpace"),
                "epm": _package_version("epm"),
                "pyrfr": _package_version("pyrfr"),
                "numpy": _package_version("numpy"),
            },
        }
        _atomic_json(output_path / "run_metadata.json", metadata)
        _atomic_json(
            completion_path,
            {"state": "running", "identity": identity},
        )

        started = time.time()
        incumbent = facade.optimize()
        walltime = time.time() - started
        telemetry_summary = telemetry.audit(facade.runhistory)
        _atomic_json(
            output_path / TELEMETRY_SUMMARY_FILENAME,
            telemetry_summary,
        )
        _atomic_json(
            output_path / "trajectory.json",
            _serialize_trajectory(facade),
        )
        incumbent_cost = float(
            facade.runhistory.average_cost(incumbent, normalize=False)
        )
        summary = {
            **identity,
            "output_directory": str(output_path),
            "finished_trials": int(facade.runhistory.finished),
            "submitted_trials": int(facade.runhistory.submitted),
            "configurations": len(facade.runhistory._config_ids),
            "incumbent": dict(incumbent),
            "incumbent_cost_on_evaluated_instance_keys": incumbent_cost,
            "final_schedule_state": schedule_callback.state,
            "target_evaluations_this_process": benchmark.evaluation_count,
            "target_timeouts_this_process": benchmark.timeout_count,
            "walltime_seconds_this_process": walltime,
            "configuration_telemetry": telemetry_summary,
        }
        _atomic_json(summary_path, summary)
        _atomic_json(
            completion_path,
            {"state": "complete", "identity": identity},
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary

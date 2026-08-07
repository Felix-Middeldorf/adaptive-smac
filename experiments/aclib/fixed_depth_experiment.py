"""Shared runner for the ACLib fixed-depth experiments."""

from __future__ import annotations

import copy
import fcntl
import importlib.metadata
import inspect
import json
import os
import socket
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ConfigSpace import Configuration


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"

# Direct runs and Submitit workers must import the modified local checkout.
local_smac = str(LOCAL_SMAC_ROOT)
if local_smac in sys.path:
    sys.path.remove(local_smac)
sys.path.insert(0, local_smac)

import smac
from smac import AlgorithmConfigurationFacade, Scenario
from smac.initial_design.random_design import RandomInitialDesign
from smac.model.random_forest.random_forest import RandomForest
from smac.runhistory import RunHistory

from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    BenchmarkData,
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


DEPTHS = (5, 10, 15, 20, 30)
SMAC_SEEDS = (0, 1, 2)
N_TRIALS = 5_000
N_TREES = 100
MIN_SAMPLES_SPLIT = 1
MIN_SAMPLES_LEAF = 1
PCA_COMPONENTS = None
RANDOM_DESIGN_PROBABILITY = 0.0
EXPERIMENT_VERSION = 2


@dataclass(frozen=True)
class ExperimentDefinition:
    benchmark_key: str
    initials: str
    directory: Path
    initial_directory: Path | None = None
    deterministic: bool | None = None
    pca_components: int | None = PCA_COMPONENTS

    def __post_init__(self) -> None:
        if self.pca_components is not None and self.pca_components < 1:
            raise ValueError("pca_components must be positive or None.")

    @property
    def output_root(self) -> Path:
        return self.directory / "results"

    @property
    def initial_choice_file(self) -> Path:
        directory = (
            self.directory
            if self.initial_directory is None
            else self.initial_directory
        )
        return directory / "initial_config.json"


@dataclass(frozen=True)
class InitialChoice:
    kind: str
    sampling_seed: int | None = None
    sampling_batch_size: int | None = None
    sample_index: int | None = None


def _assert_local_smac() -> Path:
    source = Path(inspect.getfile(RandomForest)).resolve()
    if LOCAL_SMAC_ROOT.resolve() not in source.parents:
        raise RuntimeError(f"Expected local SMAC from {LOCAL_SMAC_ROOT}, got {source}.")
    return source


LOCAL_RANDOM_FOREST_SOURCE = _assert_local_smac()


def load_initial_choice(path: Path) -> tuple[InitialChoice, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = str(payload["kind"])
    if kind == "default":
        choice = InitialChoice(kind=kind)
    elif kind == "sampled":
        choice = InitialChoice(
            kind=kind,
            sampling_seed=int(payload["sampling_seed"]),
            sampling_batch_size=int(payload["sampling_batch_size"]),
            sample_index=int(payload["sample_index"]),
        )
        assert choice.sampling_batch_size is not None
        assert choice.sample_index is not None
        if choice.sampling_batch_size < 1:
            raise ValueError("sampling_batch_size must be positive.")
        if not 0 <= choice.sample_index < choice.sampling_batch_size:
            raise ValueError("sample_index must lie inside the sampled batch.")
    else:
        raise ValueError(f"Unknown initial configuration choice {kind!r}.")
    return choice, payload


def resolve_initial_configuration(
    data: BenchmarkData,
    choice: InitialChoice,
) -> Configuration:
    if choice.kind == "default":
        config = data.configspace.get_default_configuration()
        config.origin = "Initial Design: ACLib default"
        return config

    assert choice.sampling_seed is not None
    assert choice.sampling_batch_size is not None
    assert choice.sample_index is not None
    sampling_space = copy.deepcopy(data.configspace)
    sampling_space.seed(choice.sampling_seed)
    sampled = sampling_space.sample_configuration(size=choice.sampling_batch_size)
    if isinstance(sampled, Configuration):
        sampled = [sampled]
    selected = sampled[choice.sample_index]
    config = Configuration(data.configspace, values=dict(selected))
    config.origin = (
        f"Initial Design: sampled seed={choice.sampling_seed}, "
        f"batch={choice.sampling_batch_size}, index={choice.sample_index}"
    )
    return config


def run_directory(output_root: Path, depth: int, smac_seed: int) -> Path:
    return output_root / f"depth_{depth}" / str(smac_seed)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _acquire_run_lock(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    lock_path = path / ".run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.seek(0)
        owner = handle.read().strip() or "owner metadata unavailable"
        handle.close()
        raise RuntimeError(
            f"Another process is already running this SMAC output: {path}. "
            f"Lock owner: {owner}"
        ) from error
    handle.seek(0)
    handle.truncate()
    json.dump(
        {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": time.time(),
        },
        handle,
        sort_keys=True,
    )
    handle.write("\n")
    handle.flush()
    return handle


def _completion_is_valid(path: Path, expected: dict[str, Any]) -> bool:
    required = (
        path / "scenario.json",
        path / "configspace.json",
        path / "runhistory.json",
        path / "intensifier.json",
        path / "optimization.json",
        path / "run_metadata.json",
        path / TELEMETRY_FILENAME,
        path / TELEMETRY_SUMMARY_FILENAME,
        path / "incumbent.json",
        path / "trajectory.json",
        path / "summary.json",
    )
    completion_file = path / "completed.json"
    if not completion_file.is_file() or not all(item.is_file() for item in required):
        return False
    try:
        completed = json.loads(completion_file.read_text(encoding="utf-8"))
        telemetry_summary = json.loads(
            (path / TELEMETRY_SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        run_metadata = json.loads(
            (path / "run_metadata.json").read_text(encoding="utf-8")
        )
        summary = json.loads(
            (path / "summary.json").read_text(encoding="utf-8")
        )
        scenario = Scenario.load(path)
        runhistory = RunHistory()
        runhistory.load(
            path / "runhistory.json",
            configspace=scenario.configspace,
        )
        telemetry = SurrogateTelemetryCallback(
            output_directory=path,
            model=None,
            acquisition_function=None,
        )
        audited_summary = telemetry.audit(runhistory)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False

    identity_matches = all(
        summary.get(key) == value
        for key, value in expected.items()
        if key not in {"state", "asset_signature"}
    )
    metadata_identity_matches = all(
        run_metadata.get(key) == value
        for key, value in expected.items()
        if key not in {"state", "asset_signature"}
    )
    return (
        completed == expected
        and identity_matches
        and metadata_identity_matches
        and run_metadata.get("assets") == expected["asset_signature"]
        and telemetry_summary == audited_summary
        and summary.get("configuration_telemetry") == audited_summary
    )


def _smac_state_exists(path: Path) -> bool:
    return all(
        (path / filename).is_file()
        for filename in (
            "scenario.json",
            "configspace.json",
            "runhistory.json",
            "intensifier.json",
            "optimization.json",
        )
    )


def _assert_resume_identity(
    path: Path,
    run_identity: dict[str, Any],
    asset_signature: dict[str, Any],
) -> None:
    """Reject incompatible or structurally incomplete partial output."""
    metadata_file = path / "run_metadata.json"
    completion_file = path / "completed.json"
    identity_payloads: list[tuple[Path, dict[str, Any], str]] = []

    for identity_file, asset_key in (
        (metadata_file, "assets"),
        (completion_file, "asset_signature"),
    ):
        if not identity_file.is_file():
            continue
        try:
            payload = json.loads(identity_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid existing run identity: {identity_file}."
            ) from error
        identity_payloads.append((identity_file, payload, asset_key))

    material_files = (
        "scenario.json",
        "configspace.json",
        "runhistory.json",
        "intensifier.json",
        "optimization.json",
        "run_metadata.json",
        "completed.json",
        TELEMETRY_FILENAME,
        TELEMETRY_SUMMARY_FILENAME,
        "summary.json",
    )
    existing_material = [
        path / filename
        for filename in material_files
        if (path / filename).exists()
    ]
    if existing_material and not identity_payloads:
        raise RuntimeError(
            "Existing experiment artifacts have no usable run identity: "
            + ", ".join(str(item) for item in existing_material)
            + ". "
            "Use a new output root or invoke the runner once with --overwrite."
        )

    for identity_file, previous, asset_key in identity_payloads:
        differences = {
            key: {"expected": value, "found": previous.get(key)}
            for key, value in run_identity.items()
            if previous.get(key) != value
        }
        if previous.get(asset_key) != asset_signature:
            differences["assets"] = {
                "expected": asset_signature,
                "found": previous.get(asset_key),
            }
        if differences:
            raise RuntimeError(
                f"Existing run identity {identity_file} is incompatible: "
                + json.dumps(differences, sort_keys=True)
                + ". Use a new output root or invoke the runner once with "
                "--overwrite."
            )

    smac_state_files = (
        path / "scenario.json",
        path / "configspace.json",
        path / "runhistory.json",
        path / "intensifier.json",
        path / "optimization.json",
    )
    existing_smac_state = [
        state_file for state_file in smac_state_files if state_file.exists()
    ]
    if existing_smac_state and len(existing_smac_state) != len(
        smac_state_files
    ):
        missing = [
            str(state_file)
            for state_file in smac_state_files
            if not state_file.exists()
        ]
        raise RuntimeError(
            "Existing SMAC state is incomplete; missing "
            + ", ".join(missing)
            + ". Resume cannot be made reliable. Invoke the runner once with "
            "--overwrite."
        )


def _scenario_without_meta(scenario: Scenario) -> dict[str, Any]:
    """Return persistent scenario fields while ignoring facade component metadata."""
    serialized = Scenario.make_serializable(scenario)
    serialized.pop("_meta", None)
    return serialized


def _serialize_trajectory(
    facade: AlgorithmConfigurationFacade,
) -> list[dict[str, Any]]:
    return [
        {
            "config_ids": [int(config_id) for config_id in item.config_ids],
            "costs": [float(cost) for cost in item.costs],
            "trial": int(item.trial),
            "walltime": float(item.walltime),
        }
        for item in facade.intensifier.trajectory
    ]


def _run_locked(
    *,
    definition: ExperimentDefinition,
    depth: int,
    smac_seed: int,
    n_trials: int,
    output_root: Path,
    output_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    spec = get_benchmark_spec(definition.benchmark_key)
    choice, choice_payload = load_initial_choice(definition.initial_choice_file)
    deterministic = (
        spec.deterministic
        if definition.deterministic is None
        else bool(definition.deterministic)
    )
    run_identity = {
        "experiment_version": EXPERIMENT_VERSION,
        "benchmark": spec.key,
        "initials": definition.initials,
        "depth": int(depth),
        "smac_seed": int(smac_seed),
        "n_trials": int(n_trials),
        "n_training_instances": spec.expected_training_instances,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "initial_configuration_choice": {
            "kind": choice.kind,
            "sampling_seed": choice.sampling_seed,
            "sampling_batch_size": choice.sampling_batch_size,
            "sample_index": choice.sample_index,
        },
        "smac_model": {
            "n_trees": N_TREES,
            "max_depth": int(depth),
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "pca_components": definition.pca_components,
        },
        "random_design_probability": RANDOM_DESIGN_PROBABILITY,
    }
    if definition.deterministic is not None:
        run_identity.update(
            {
                "deterministic": deterministic,
                "target_quantile_seed": 0,
            }
        )
    completion = {
        "state": "complete",
        **run_identity,
        "asset_signature": asset_metadata(spec),
    }
    if not overwrite and _completion_is_valid(output_path, completion):
        print(f"Complete run found; skipping {output_path}")
        return json.loads((output_path / "summary.json").read_text(encoding="utf-8"))

    if not overwrite:
        _assert_resume_identity(
            output_path,
            run_identity,
            completion["asset_signature"],
        )
    _atomic_json(output_path / "completed.json", {**completion, "state": "running"})
    started = time.time()

    data = load_benchmark_data(spec)
    training_instances = data.training_instances
    training_features = {
        instance: data.features[instance]
        for instance in training_instances
    }
    initial_config = resolve_initial_configuration(data, choice)
    scenario = Scenario(
        configspace=data.configspace,
        name=f"depth_{depth}",
        output_directory=output_root,
        deterministic=deterministic,
        objectives="PAR10",
        crash_cost=spec.timeout_cost,
        n_trials=n_trials,
        use_default_config=choice.kind == "default",
        instances=list(training_instances),
        instance_features=training_features,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != output_path:
        raise RuntimeError(
            f"Unexpected SMAC output path {scenario.output_directory}; expected {output_path}."
        )

    if choice.kind == "default":
        initial_design = AlgorithmConfigurationFacade.get_initial_design(scenario)
    else:
        initial_design = RandomInitialDesign(
            scenario,
            n_configs=0,
            additional_configs=[initial_config],
        )

    model = AlgorithmConfigurationFacade.get_model(
        scenario=scenario,
        n_trees=N_TREES,
        max_depth=depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        pca_components=definition.pca_components,
    )
    random_design = AlgorithmConfigurationFacade.get_random_design(
        scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    default_acquisition_function = (
        AlgorithmConfigurationFacade.get_acquisition_function(scenario)
    )
    acquisition_function = TelemetryEI.from_smac_ei(
        default_acquisition_function
    )

    run_metadata = {
        **run_identity,
        "display_name": spec.display_name,
        "scenario_name": spec.scenario_name,
        "output_directory": str(output_path),
        "training_instances": list(training_instances),
        "test_instances_reserved_and_unused": len(data.test_instances),
        "training_only": True,
        "cutoff": spec.cutoff,
        "timeout_cost": spec.timeout_cost,
        "deterministic": deterministic,
        "target_quantile_seed": (
            0 if deterministic else "SMAC evaluation seed"
        ),
        "selected_initial_configuration": dict(initial_config),
        "initial_configuration_assessment": choice_payload.get("assessment"),
        "adaptive_capping": False,
        "facade": "AlgorithmConfigurationFacade",
        "acquisition_function": acquisition_function.meta,
        "telemetry": {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "configuration_file": str(output_path / TELEMETRY_FILENAME),
            "summary_file": str(output_path / TELEMETRY_SUMMARY_FILENAME),
            "format": "append-only JSON Lines event stream",
            "proposal_event": (
                "One durable event per unique configuration, captured before "
                "the configuration is yielded for evaluation"
            ),
            "first_completion_event": (
                "Links the proposal fingerprint to its SMAC runhistory config "
                "ID and first completed target trial"
            ),
            "prediction": (
                "SMAC RF mean, variance, and standard deviation marginalized "
                "over all training instances at proposal time"
            ),
            "acquisition": (
                "Expected Improvement and its RF moments retained from the "
                "exact current acquisition evaluation (with a one-prediction "
                "closed-form fallback)"
            ),
            "tree_depths": "get_depth() for every fitted SMAC RF tree",
        },
        "local_smac_root": str(LOCAL_SMAC_ROOT),
        "local_random_forest_source": str(LOCAL_RANDOM_FOREST_SOURCE),
        "assets": completion["asset_signature"],
        "versions": {
            "python": sys.version,
            "smac_distribution": _package_version("smac"),
            "ConfigSpace": _package_version("ConfigSpace"),
            "epm": _package_version("epm"),
            "pyrfr": _package_version("pyrfr"),
            "numpy": _package_version("numpy"),
        },
    }
    _atomic_json(output_path / "run_metadata.json", run_metadata)

    if not overwrite and _smac_state_exists(output_path):
        previous_scenario = Scenario.load(output_path)
        if _scenario_without_meta(scenario) != _scenario_without_meta(
            previous_scenario
        ):
            raise RuntimeError(
                "Existing SMAC state is incompatible with this run. Use a new "
                "output root or invoke the runner once with --overwrite."
            )

    print(
        f"Loading {spec.display_name} for depth={depth}, "
        f"SMAC seed={smac_seed} ..."
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

    target_function = (
        deterministic_target if deterministic else benchmark.target
    )
    telemetry = SurrogateTelemetryCallback(
        output_directory=output_path,
        model=model,
        acquisition_function=acquisition_function,
        overwrite=overwrite,
    )
    facade = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=target_function,
        model=model,
        acquisition_function=acquisition_function,
        initial_design=initial_design,
        random_design=random_design,
        callbacks=[telemetry],
        overwrite=overwrite,
    )
    incumbent = facade.optimize()
    incumbent_cost = float(
        facade.runhistory.average_cost(incumbent, normalize=False)
    )
    _atomic_json(
        output_path / "incumbent.json",
        {
            "config_id": int(facade.runhistory.get_config_id(incumbent)),
            "configuration": dict(incumbent),
            "training_cost_on_evaluated_instance_seed_keys": incumbent_cost,
        },
    )
    _atomic_json(output_path / "trajectory.json", _serialize_trajectory(facade))
    telemetry_summary = telemetry.audit(facade.runhistory)
    _atomic_json(
        output_path / TELEMETRY_SUMMARY_FILENAME,
        telemetry_summary,
    )

    summary = {
        **run_identity,
        "output_directory": str(output_path),
        "model_type": benchmark.model_type,
        "finished_trials": int(facade.runhistory.finished),
        "submitted_trials": int(facade.runhistory.submitted),
        "incumbent_cost": incumbent_cost,
        "target_evaluations_this_process": int(benchmark.evaluation_count),
        "target_timeouts_this_process": int(benchmark.timeout_count),
        "configuration_telemetry": telemetry_summary,
        "walltime_seconds_this_process": float(time.time() - started),
    }
    _atomic_json(output_path / "summary.json", summary)
    _atomic_json(output_path / "completed.json", completion)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_fixed_depth(
    *,
    definition: ExperimentDefinition,
    depth: int,
    smac_seed: int,
    n_trials: int = N_TRIALS,
    output_root: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")

    if output_root is None:
        output_root = definition.output_root
    output_root = Path(output_root).resolve()
    output_path = run_directory(output_root, depth, smac_seed)
    with closing(_acquire_run_lock(output_path)):
        return _run_locked(
            definition=definition,
            depth=depth,
            smac_seed=smac_seed,
            n_trials=n_trials,
            output_root=output_root,
            output_path=output_path,
            overwrite=overwrite,
        )


def local_smac_metadata() -> dict[str, str]:
    return {
        "module": str(Path(smac.__file__).resolve()),
        "random_forest": str(LOCAL_RANDOM_FOREST_SOURCE),
    }

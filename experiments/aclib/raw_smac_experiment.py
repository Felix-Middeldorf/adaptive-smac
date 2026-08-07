"""Minimal native-SMAC runner for ACLib fixed-depth timing controls."""

from __future__ import annotations

import copy
import inspect
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ConfigSpace import Configuration


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
LOCAL_SMAC_ROOT = REPOSITORY_ROOT / "external" / "SMAC3"

# Workers must use the local checkout containing the experiment's SMAC changes.
local_smac = str(LOCAL_SMAC_ROOT)
if local_smac in sys.path:
    sys.path.remove(local_smac)
sys.path.insert(0, local_smac)

from smac import AlgorithmConfigurationFacade, Scenario
from smac.initial_design.random_design import RandomInitialDesign
from smac.model.random_forest.random_forest import RandomForest

from surrogate_benchmark import (
    ACLibSurrogateBenchmark,
    BenchmarkData,
    get_benchmark_spec,
    load_benchmark_data,
    par10_cost,
)


DEPTHS = (5, 10, 20, 30)
SMAC_SEEDS = (0, 1)
N_TRIALS = 5_000
N_TREES = 100
MIN_SAMPLES_SPLIT = 1
MIN_SAMPLES_LEAF = 1
PCA_COMPONENTS = (None, 4)
RANDOM_DESIGN_PROBABILITY = 0.0
DETERMINISTIC = True
SURROGATE_QUANTILE_SEED = 0


@dataclass(frozen=True)
class RawExperimentDefinition:
    benchmark_key: str
    initials: str
    directory: Path
    initial_choice_file: Path
    training_instance_limit: int | None = None

    @property
    def output_root(self) -> Path:
        return self.directory / "results"


@dataclass(frozen=True)
class InitialChoice:
    kind: str
    sampling_seed: int | None = None
    sampling_batch_size: int | None = None
    sample_index: int | None = None


def _assert_local_smac() -> Path:
    source = Path(inspect.getfile(RandomForest)).resolve()
    if LOCAL_SMAC_ROOT.resolve() not in source.parents:
        raise RuntimeError(
            f"Expected local SMAC from {LOCAL_SMAC_ROOT}, got {source}."
        )
    return source


LOCAL_RANDOM_FOREST_SOURCE = _assert_local_smac()


def load_initial_choice(path: Path) -> InitialChoice:
    payload = json.loads(path.read_text(encoding="utf-8"))
    kind = str(payload["kind"])
    if kind == "default":
        return InitialChoice(kind=kind)
    if kind != "sampled":
        raise ValueError(f"Unknown initial configuration kind {kind!r}.")
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
        raise ValueError("sample_index lies outside the sampled batch.")
    return choice


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
    sampled = sampling_space.sample_configuration(
        size=choice.sampling_batch_size
    )
    if isinstance(sampled, Configuration):
        sampled = [sampled]
    selected = Configuration(data.configspace, values=dict(sampled[choice.sample_index]))
    selected.origin = (
        f"Initial Design: sampled seed={choice.sampling_seed}, "
        f"batch={choice.sampling_batch_size}, index={choice.sample_index}"
    )
    return selected


def native_output_directory(
    definition: RawExperimentDefinition,
    depth: int,
    smac_seed: int,
    pca_components: int | None,
) -> Path:
    pca_name = "pca_none" if pca_components is None else f"pca_{pca_components}"
    return (
        definition.output_root
        / pca_name
        / f"depth_{depth}"
        / str(smac_seed)
    )


def build_components(
    definition: RawExperimentDefinition,
    *,
    depth: int,
    smac_seed: int,
    pca_components: int | None,
    n_trials: int = N_TRIALS,
) -> tuple[
    Any,
    BenchmarkData,
    Scenario,
    Configuration,
    Any,
    Any,
    Any,
]:
    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}.")
    if smac_seed not in SMAC_SEEDS:
        raise ValueError(f"smac_seed must be one of {SMAC_SEEDS}.")
    if pca_components not in PCA_COMPONENTS:
        raise ValueError(
            f"pca_components must be one of {PCA_COMPONENTS}."
        )
    if n_trials < 1:
        raise ValueError("n_trials must be positive.")

    spec = get_benchmark_spec(definition.benchmark_key)
    data = load_benchmark_data(spec)
    if (
        definition.training_instance_limit is not None
        and definition.training_instance_limit < 1
    ):
        raise ValueError("training_instance_limit must be positive.")
    training_instances = data.training_instances[
        : definition.training_instance_limit
    ]
    if not training_instances:
        raise ValueError("The selected training-instance set is empty.")
    choice = load_initial_choice(definition.initial_choice_file)
    initial_config = resolve_initial_configuration(data, choice)
    training_features = {
        instance: data.features[instance]
        for instance in training_instances
    }
    scenario = Scenario(
        configspace=data.configspace,
        name=f"depth_{depth}",
        output_directory=(
            definition.output_root
            / (
                "pca_none"
                if pca_components is None
                else f"pca_{pca_components}"
            )
        ),
        deterministic=DETERMINISTIC,
        objectives="PAR10",
        crash_cost=spec.timeout_cost,
        n_trials=n_trials,
        use_default_config=choice.kind == "default",
        instances=list(training_instances),
        instance_features=training_features,
        seed=smac_seed,
        n_workers=1,
    )
    if scenario.output_directory != native_output_directory(
        definition,
        depth,
        smac_seed,
        pca_components,
    ):
        raise RuntimeError(
            f"Unexpected native SMAC output path {scenario.output_directory}."
        )

    if choice.kind == "default":
        initial_design = AlgorithmConfigurationFacade.get_initial_design(
            scenario
        )
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
        pca_components=pca_components,
    )
    random_design = AlgorithmConfigurationFacade.get_random_design(
        scenario,
        probability=RANDOM_DESIGN_PROBABILITY,
    )
    return (
        spec,
        data,
        scenario,
        initial_config,
        initial_design,
        model,
        random_design,
    )


def run_raw_smac(
    *,
    definition: RawExperimentDefinition,
    depth: int,
    smac_seed: int,
    pca_components: int | None,
    n_trials: int = N_TRIALS,
    overwrite: bool = False,
) -> dict[str, Any]:
    (
        spec,
        _data,
        scenario,
        _initial_config,
        initial_design,
        model,
        random_design,
    ) = build_components(
        definition,
        depth=depth,
        smac_seed=smac_seed,
        pca_components=pca_components,
        n_trials=n_trials,
    )
    benchmark = ACLibSurrogateBenchmark(spec)

    # Minimal target path: input conversion, EPM prediction, and PAR10
    # conversion only. It intentionally skips ACLibSurrogateBenchmark.evaluate
    # because that method creates an information dictionary and updates
    # diagnostic counters that this timing control does not consume.
    def target(
        config: Configuration,
        instance: str,
        seed: int = 0,
    ) -> float:
        prediction_array, status_array = benchmark.surrogate.predict(
            X=benchmark._model_input(config, instance),
            quality=False,
            cutoff=spec.cutoff,
            quantile_seed=SURROGATE_QUANTILE_SEED,
        )
        prediction = float(np.asarray(prediction_array).reshape(-1)[0])
        status = str(np.asarray(status_array).reshape(-1)[0])
        return par10_cost(
            prediction,
            status,
            spec.timeout_cost,
        )

    facade = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=target,
        model=model,
        initial_design=initial_design,
        random_design=random_design,
        callbacks=[],
        overwrite=overwrite,
    )
    started = time.monotonic()
    incumbent = facade.optimize()
    elapsed = time.monotonic() - started
    result = {
        "benchmark": spec.key,
        "depth": depth,
        "smac_seed": smac_seed,
        "pca_components": pca_components,
        "finished_trials": int(facade.runhistory.finished),
        "walltime_seconds": elapsed,
        "incumbent_cost": float(
            facade.runhistory.average_cost(incumbent, normalize=False)
        ),
        "native_output_directory": str(scenario.output_directory),
    }
    # This is printed to the Submitit log only; no custom result files are
    # written inside the SMAC output directory.
    print(json.dumps(result, sort_keys=True))
    return result


def local_smac_metadata() -> dict[str, str]:
    import smac

    return {
        "module": str(Path(smac.__file__).resolve()),
        "random_forest": str(LOCAL_RANDOM_FOREST_SOURCE),
    }

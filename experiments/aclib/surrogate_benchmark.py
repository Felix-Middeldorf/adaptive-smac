"""Shared in-process adapters for the downloaded ACLib surrogate benchmarks."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.read_and_write import pcs
from ConfigSpace.util import fix_types, impute_inactive_values
from epm.experiment_utils.config_space_utils import encode_config_as_array_with_true_values
from epm.surrogates.surrogate_model import SurrogateModel


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
ACLIB_ROOT = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "aclib2"


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    scenario_name: str
    model_name: str
    instance_set: tuple[str, ...]
    cutoff: float
    deterministic: bool
    expected_hyperparameters: int
    expected_conditions: int
    expected_forbiddens: int
    expected_features: int
    expected_training_instances: int
    expected_test_instances: int

    @property
    def timeout_cost(self) -> float:
        return 10.0 * self.cutoff

    @property
    def model_directory(self) -> Path:
        return ACLIB_ROOT / "target_algorithms" / "surrogate" / self.model_name

    @property
    def instance_directory(self) -> Path:
        return ACLIB_ROOT.joinpath("instances", *self.instance_set)

    @property
    def file_stem(self) -> str:
        return f"{self.model_name}.par10.random"

    @property
    def configspace_file(self) -> Path:
        return self.model_directory / f"config_space.{self.file_stem}.pcs"

    @property
    def feature_file(self) -> Path:
        return self.model_directory / f"inst_feat_dict.{self.file_stem}.json"

    @property
    def model_file(self) -> Path:
        return self.model_directory / f"pyrfr_model.{self.file_stem}.bin"

    @property
    def wrapper_file(self) -> Path:
        return self.model_directory / f"pyrfr_wrapper.{self.file_stem}.pkl"

    @property
    def training_file(self) -> Path:
        return self.instance_directory / "training.txt"

    @property
    def test_file(self) -> Path:
        return self.instance_directory / "test.txt"

    def required_assets(self) -> tuple[Path, ...]:
        return (
            self.configspace_file,
            self.feature_file,
            self.model_file,
            self.wrapper_file,
            self.training_file,
            self.test_file,
        )


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "lingeling_circuitfuzz": BenchmarkSpec(
        key="lingeling_circuitfuzz",
        display_name="Lingeling Circuitfuzz",
        scenario_name="lingeling_circuitfuzz_surrogate",
        model_name="lingeling_circuitfuzz",
        instance_set=("sat", "sets", "CIRCUITFUZZ-CSSC14"),
        cutoff=300.0,
        deterministic=False,
        expected_hyperparameters=322,
        expected_conditions=0,
        expected_forbiddens=0,
        expected_features=119,
        expected_training_instances=299,
        expected_test_instances=302,
    ),
    "clasp_queens": BenchmarkSpec(
        key="clasp_queens",
        display_name="Clasp Queens",
        scenario_name="clasp_queens_surrogate",
        model_name="clasp_queens",
        instance_set=("sat", "sets", "QUEENS-CSSC14"),
        cutoff=300.0,
        deterministic=False,
        expected_hyperparameters=75,
        expected_conditions=55,
        expected_forbiddens=2,
        expected_features=119,
        expected_training_instances=484,
        expected_test_instances=351,
    ),
    "cplex_rcw": BenchmarkSpec(
        key="cplex_rcw",
        display_name="CPLEX RCW",
        scenario_name="cplex_rcw_surrogate",
        model_name="cplex_rcw",
        instance_set=("mip", "sets", "RCW"),
        cutoff=10_000.0,
        deterministic=True,
        expected_hyperparameters=74,
        expected_conditions=4,
        expected_forbiddens=0,
        expected_features=148,
        expected_training_instances=495,
        expected_test_instances=495,
    ),
    "lpg_zenotravel": BenchmarkSpec(
        key="lpg_zenotravel",
        display_name="LPG Zenotravel",
        scenario_name="lpg_zenotravel_surrogate",
        model_name="lpg_zenotravel",
        instance_set=("planning", "sets", "zenotravel"),
        cutoff=300.0,
        deterministic=False,
        expected_hyperparameters=67,
        expected_conditions=22,
        expected_forbiddens=12,
        expected_features=305,
        expected_training_instances=2_000,
        expected_test_instances=2_000,
    ),
    "clasp_weighted": BenchmarkSpec(
        key="clasp_weighted",
        display_name="Clasp weighted-sequence",
        scenario_name="clasp_weighted-sequence_surrogate",
        model_name="clasp_weighted",
        instance_set=("asp", "sets", "weighted-sequence"),
        cutoff=900.0,
        deterministic=False,
        expected_hyperparameters=98,
        expected_conditions=63,
        expected_forbiddens=2,
        expected_features=38,
        expected_training_instances=240,
        expected_test_instances=240,
    ),
}


@dataclass(frozen=True)
class BenchmarkData:
    configspace: ConfigurationSpace
    features: dict[str, list[float]]
    training_instances: tuple[str, ...]
    test_instances: tuple[str, ...]


def get_benchmark_spec(key: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[key]
    except KeyError as error:
        raise KeyError(f"Unknown ACLib surrogate benchmark {key!r}.") from error


def _read_configspace(spec: BenchmarkSpec) -> ConfigurationSpace:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with spec.configspace_file.open("r", encoding="utf-8") as handle:
            return pcs.read(handle)


def _decode_feature_vector(value: Any) -> list[float]:
    if isinstance(value, dict) and "__ndarray__" in value:
        value = value["__ndarray__"]
    if not isinstance(value, list):
        raise TypeError(f"Unexpected feature representation: {type(value)!r}")
    return [float(item) for item in value]


def _resolve_instances(
    filename: Path,
    canonical_by_basename: Mapping[str, str],
) -> tuple[str, ...]:
    requested = tuple(
        line.strip()
        for line in filename.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    resolved: list[str] = []
    for instance in requested:
        basename = Path(instance).name
        try:
            resolved.append(canonical_by_basename[basename])
        except KeyError as error:
            raise KeyError(
                f"Instance {instance!r} from {filename} has no surrogate feature vector."
            ) from error
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"Duplicate instances after resolving {filename}.")
    return tuple(resolved)


def load_benchmark_data(spec: BenchmarkSpec | str) -> BenchmarkData:
    if isinstance(spec, str):
        spec = get_benchmark_spec(spec)

    missing = [str(path) for path in spec.required_assets() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ACLib assets:\n" + "\n".join(missing))

    raw_features = json.loads(spec.feature_file.read_text(encoding="utf-8"))
    features = {
        str(instance): _decode_feature_vector(value)
        for instance, value in raw_features.items()
    }
    dimensions = {len(vector) for vector in features.values()}
    if dimensions != {spec.expected_features}:
        raise ValueError(
            f"{spec.key} expected {spec.expected_features} features, found {sorted(dimensions)}."
        )

    canonical_by_basename: dict[str, str] = {}
    duplicate_basenames: set[str] = set()
    for instance in features:
        basename = Path(instance).name
        if basename in canonical_by_basename:
            duplicate_basenames.add(basename)
        canonical_by_basename[basename] = instance
    if duplicate_basenames:
        raise ValueError(
            f"{spec.key} feature dictionary has duplicate basenames: "
            f"{sorted(duplicate_basenames)[:5]}"
        )

    training_instances = _resolve_instances(spec.training_file, canonical_by_basename)
    test_instances = _resolve_instances(spec.test_file, canonical_by_basename)
    if set(training_instances) & set(test_instances):
        raise ValueError(f"{spec.key} training and test instance sets overlap.")
    if len(training_instances) != spec.expected_training_instances:
        raise ValueError(
            f"{spec.key} expected {spec.expected_training_instances} training instances, "
            f"found {len(training_instances)}."
        )
    if len(test_instances) != spec.expected_test_instances:
        raise ValueError(
            f"{spec.key} expected {spec.expected_test_instances} test instances, "
            f"found {len(test_instances)}."
        )

    configspace = _read_configspace(spec)
    actual_space = (
        len(configspace),
        len(configspace.conditions),
        len(configspace.forbidden_clauses),
    )
    expected_space = (
        spec.expected_hyperparameters,
        spec.expected_conditions,
        spec.expected_forbiddens,
    )
    if actual_space != expected_space:
        raise ValueError(
            f"{spec.key} expected configuration-space counts {expected_space}, "
            f"found {actual_space}."
        )

    return BenchmarkData(
        configspace=configspace,
        features=features,
        training_instances=training_instances,
        test_instances=test_instances,
    )


def par10_cost(prediction: float, status: str, timeout_cost: float) -> float:
    normalized_status = status.upper()
    if normalized_status == "CUTOFF":
        return timeout_cost
    if normalized_status not in {"TRUE", "SAT", "SUCCESS"}:
        raise RuntimeError(f"Surrogate returned unexpected status {status!r}.")
    if not math.isfinite(prediction) or prediction < 0.0:
        raise RuntimeError(f"Surrogate returned invalid prediction {prediction!r}.")
    return float(prediction)


def _configuration_items(
    config: Configuration | Mapping[str, Any],
) -> Sequence[tuple[str, Any]]:
    if isinstance(config, Mapping):
        return tuple(config.items())
    return tuple(dict(config).items())


class ACLibSurrogateBenchmark:
    """In-process adapter from modern SMAC to one downloaded ACLib EPM."""

    def __init__(self, spec: BenchmarkSpec | str) -> None:
        if isinstance(spec, str):
            spec = get_benchmark_spec(spec)
        self.spec = spec

        surrogate = SurrogateModel(
            pyrfr_wrapper=str(spec.wrapper_file),
            pyrfr_model=str(spec.model_file),
            config_space=str(spec.configspace_file),
            inst_feat_dict=str(spec.feature_file),
            idle_time=0,
            impute_with="def",
            quality=False,
            dtype=np.float32,
            debug=False,
        )
        surrogate.load_model()
        surrogate.logger.disabled = True
        if getattr(surrogate, "model", None) is not None:
            surrogate.model.logger.disabled = True

        self.surrogate = surrogate
        self.evaluation_count = 0
        self.timeout_count = 0

    @property
    def model_type(self) -> str:
        return type(self.surrogate.model).__name__

    def _model_input(
        self,
        config: Configuration | Mapping[str, Any],
        instance: str,
    ) -> np.ndarray:
        supplied = dict(_configuration_items(config))
        typed = fix_types(supplied, self.surrogate.cs)
        imputed = {
            name: typed.get(name, hyperparameter.default_value)
            for name, hyperparameter in self.surrogate.cs.items()
        }
        legacy_config = Configuration(
            configuration_space=self.surrogate.cs,
            values=imputed,
            allow_inactive_with_values=True,
        )
        legacy_config = impute_inactive_values(legacy_config, "default")
        parameters = encode_config_as_array_with_true_values(
            config=legacy_config,
            cs=self.surrogate.cs,
        ).reshape((-1,))

        try:
            features = np.asarray(self.surrogate.inst_feat_dict[instance]).reshape((-1,))
        except KeyError as error:
            raise KeyError(f"Unknown {self.spec.key} instance {instance!r}.") from error
        return np.hstack((parameters, features)).reshape((1, -1)).astype(np.float32)

    def evaluate(
        self,
        config: Configuration | Mapping[str, Any],
        instance: str,
        *,
        seed: int = 0,
    ) -> tuple[float, dict[str, Any]]:
        quantile_seed = 0 if self.spec.deterministic else int(seed)
        prediction_array, status_array = self.surrogate.predict(
            X=self._model_input(config, instance),
            quality=False,
            cutoff=self.spec.cutoff,
            quantile_seed=quantile_seed,
        )
        prediction = float(np.asarray(prediction_array).reshape(-1)[0])
        status = str(np.asarray(status_array).reshape(-1)[0])
        cost = par10_cost(prediction, status, self.spec.timeout_cost)

        self.evaluation_count += 1
        if status.upper() == "CUTOFF":
            self.timeout_count += 1

        return cost, {
            "surrogate_prediction": prediction,
            "surrogate_status": status,
            "par10_cost": cost,
            "quantile_seed": quantile_seed,
        }

    def target(
        self,
        config: Configuration,
        seed: int,
        instance: str,
    ) -> tuple[float, dict[str, Any]]:
        cost, info = self.evaluate(config=config, instance=instance, seed=seed)
        info["smac_seed_argument"] = int(seed)
        return cost, info


def asset_metadata(spec: BenchmarkSpec) -> dict[str, dict[str, int]]:
    metadata: dict[str, dict[str, int]] = {}
    for path in spec.required_assets():
        stat = path.stat()
        metadata[str(path.relative_to(ACLIB_ROOT))] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return metadata

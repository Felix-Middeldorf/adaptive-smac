from __future__ import annotations

import json
import logging
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
REPOSITORY_ROOT = HERE.parents[2]
ACLIB_ROOT = REPOSITORY_ROOT / "external" / "aclib-surrogates" / "aclib2"
MODEL_DIRECTORY = ACLIB_ROOT / "target_algorithms" / "surrogate" / "cplex_regions200"
INSTANCE_DIRECTORY = ACLIB_ROOT / "instances" / "mip" / "sets" / "Regions200"

CONFIGSPACE_FILE = MODEL_DIRECTORY / "config_space.cplex_regions200.par10.random.pcs"
FEATURE_FILE = MODEL_DIRECTORY / "inst_feat_dict.cplex_regions200.par10.random.json"
MODEL_FILE = MODEL_DIRECTORY / "pyrfr_model.cplex_regions200.par10.random.bin"
WRAPPER_FILE = MODEL_DIRECTORY / "pyrfr_wrapper.cplex_regions200.par10.random.pkl"
TRAINING_FILE = INSTANCE_DIRECTORY / "training.txt"
TEST_FILE = INSTANCE_DIRECTORY / "test.txt"

CUTOFF = 10_000.0
PAR_FACTOR = 10.0
TIMEOUT_COST = CUTOFF * PAR_FACTOR
EXPECTED_HYPERPARAMETERS = 74
EXPECTED_CONDITIONS = 4
EXPECTED_FEATURES = 148
EXPECTED_INSTANCES_PER_SPLIT = 1_000


@dataclass
class BenchmarkData:
    configspace: ConfigurationSpace
    features: dict[str, list[float]]
    training_instances: tuple[str, ...]
    test_instances: tuple[str, ...]


def required_assets() -> tuple[Path, ...]:
    return (
        CONFIGSPACE_FILE,
        FEATURE_FILE,
        MODEL_FILE,
        WRAPPER_FILE,
        TRAINING_FILE,
        TEST_FILE,
    )


def _read_configspace() -> ConfigurationSpace:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with CONFIGSPACE_FILE.open("r", encoding="utf-8") as handle:
            return pcs.read(handle)


def _decode_feature_vector(value: Any) -> list[float]:
    if isinstance(value, dict) and "__ndarray__" in value:
        value = value["__ndarray__"]
    if not isinstance(value, list):
        raise TypeError(f"Unexpected feature representation: {type(value)!r}")
    vector = [float(item) for item in value]
    if len(vector) != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, found {len(vector)}."
        )
    return vector


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
                f"Instance {instance!r} from {filename} has no feature vector."
            ) from error
    if len(resolved) != len(set(resolved)):
        raise ValueError(f"Duplicate instances after resolving {filename}.")
    return tuple(resolved)


def load_benchmark_data() -> BenchmarkData:
    missing = [str(path) for path in required_assets() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing ACLib assets:\n" + "\n".join(missing))

    raw_features = json.loads(FEATURE_FILE.read_text(encoding="utf-8"))
    features = {
        str(instance): _decode_feature_vector(value)
        for instance, value in raw_features.items()
    }

    canonical_by_basename: dict[str, str] = {}
    duplicate_basenames: set[str] = set()
    for instance in features:
        basename = Path(instance).name
        if basename in canonical_by_basename:
            duplicate_basenames.add(basename)
        canonical_by_basename[basename] = instance
    if duplicate_basenames:
        sample = sorted(duplicate_basenames)[:5]
        raise ValueError(f"Feature dictionary has duplicate basenames: {sample}")

    training_instances = _resolve_instances(TRAINING_FILE, canonical_by_basename)
    test_instances = _resolve_instances(TEST_FILE, canonical_by_basename)
    if set(training_instances) & set(test_instances):
        raise ValueError("Training and test instance sets overlap.")
    if len(training_instances) != EXPECTED_INSTANCES_PER_SPLIT:
        raise ValueError(
            f"Expected {EXPECTED_INSTANCES_PER_SPLIT} training instances, "
            f"found {len(training_instances)}."
        )
    if len(test_instances) != EXPECTED_INSTANCES_PER_SPLIT:
        raise ValueError(
            f"Expected {EXPECTED_INSTANCES_PER_SPLIT} test instances, "
            f"found {len(test_instances)}."
        )

    configspace = _read_configspace()
    if len(configspace) != EXPECTED_HYPERPARAMETERS:
        raise ValueError(
            f"Expected {EXPECTED_HYPERPARAMETERS} hyperparameters, "
            f"found {len(configspace)}."
        )
    if len(configspace.conditions) != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Expected {EXPECTED_CONDITIONS} conditions, "
            f"found {len(configspace.conditions)}."
        )

    return BenchmarkData(
        configspace=configspace,
        features=features,
        training_instances=training_instances,
        test_instances=test_instances,
    )


def par10_cost(prediction: float, status: str) -> float:
    normalized_status = status.upper()
    if normalized_status == "CUTOFF":
        return TIMEOUT_COST
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


class CplexRegions200Benchmark:
    """In-process adapter from modern SMAC to the legacy EPM model.

    One object is constructed per SMAC run, so the roughly 1 GB PyRFR model is
    loaded once and reused for all target evaluations.
    """

    def __init__(self) -> None:
        surrogate = SurrogateModel(
            pyrfr_wrapper=str(WRAPPER_FILE),
            pyrfr_model=str(MODEL_FILE),
            config_space=str(CONFIGSPACE_FILE),
            inst_feat_dict=str(FEATURE_FILE),
            idle_time=0,
            impute_with="def",
            quality=False,
            dtype=np.float32,
            debug=False,
        )
        surrogate.load_model()

        # The historical quantile model logs a CRITICAL message for every
        # deterministic median prediction. Silence those routine messages.
        surrogate.logger.disabled = True
        if getattr(surrogate, "model", None) is not None:
            surrogate.model.logger.disabled = True

        self.surrogate = surrogate
        self.evaluation_count = 0
        self.timeout_count = 0

    @property
    def model_type(self) -> str:
        return type(self.surrogate.model).__name__

    def evaluate(
        self,
        config: Configuration | Mapping[str, Any],
        instance: str,
    ) -> tuple[float, dict[str, Any]]:
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
            raise KeyError(f"Unknown benchmark instance {instance!r}.") from error
        model_input = np.hstack((parameters, features)).reshape((1, -1)).astype(np.float32)

        # Seed zero selects the deterministic median of the legacy quantile
        # forest. The original fixed cutoff is applied inside EPM.
        prediction_array, status_array = self.surrogate.predict(
            X=model_input,
            quality=False,
            cutoff=CUTOFF,
            quantile_seed=0,
        )
        prediction = float(np.asarray(prediction_array).reshape(-1)[0])
        status = str(np.asarray(status_array).reshape(-1)[0])
        cost = par10_cost(prediction, status)

        self.evaluation_count += 1
        if status.upper() == "CUTOFF":
            self.timeout_count += 1

        return cost, {
            "surrogate_prediction": prediction,
            "surrogate_status": status,
            "par10_cost": cost,
            "quantile_seed": 0,
        }

    def target(
        self,
        config: Configuration,
        seed: int,
        instance: str,
    ) -> tuple[float, dict[str, Any]]:
        # ACFacade's intensifier supplies a seed even for a deterministic
        # scenario. The trained quantile surrogate deliberately uses median
        # seed 0, making the target surface deterministic.
        cost, info = self.evaluate(config=config, instance=instance)
        info["smac_seed_argument"] = int(seed)
        return cost, info

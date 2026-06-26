from pathlib import Path

from ConfigSpace import ConfigurationSpace, Configuration
from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade


def target_function(config: Configuration, seed: int = 0) -> float:
    return config["x1"] ** 2 + config["x2"] ** 2


cs = ConfigurationSpace(
    {
        "x1": (-10.0, 10.0),
        "x2": (-10.0, 10.0),
    },
    seed=0,
)

sc = Scenario(
    name="simple_scenario",
    configspace=cs,
    output_directory=Path(
        "/home/io632776/experiments/adaptive-smac/experiments/hartmann/Debug_04"
    ),
    deterministic=True,
    n_trials=100,
    seed=0,
)

smac = HPOFacade(
    scenario=sc,
    target_function=target_function,
    overwrite=True,
)

incumbent = smac.optimize()
print(dict(incumbent))
print(smac.validate(incumbent))
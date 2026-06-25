from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from ConfigSpace import ConfigurationSpace,Configuration,Float
from pathlib import Path
import submitit

SLURM_PARTITION = "c23ms"

def eval(config:Configuration,seed:int = 0):
    return sum([config[f"x{i}"]**2 for i in [1,2]])

cs = ConfigurationSpace({
    "x1": (-10.0,10.0),
    "x2": (-10.0,10.0)
})

sc = Scenario(
    name="simple_scenario",
    configspace=cs,
    output_directory=Path("/home/io632776/experiments/adaptive-smac/experiments/hartmann/Debug_04"),
    seed=0
)

smac = HPOFacade(
    scenario=sc,
    target_function=eval,
    overwrite=True
)

incumbent = smac.optimize()
rh = smac.runhistory

print(rh.get_cost(incumbent))


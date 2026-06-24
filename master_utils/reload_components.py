# smac_utils.py

from pathlib import Path
import json
from typing import Union, Optional

from smac import Scenario
from smac.facade import HyperparameterOptimizationFacade as HPOFacade
from smac.facade import AlgorithmConfigurationFacade as ACFacade
from smac.runhistory.runhistory import RunHistory


def load_smac_components(run_dir: Union[str,Path], facade: Optional[str] = None):
    """
    Load the main files from a finished SMAC run.

    Parameters
    ----------
    run_dir : str | pathlib.Path
        Directory containing scenario.json, configspace.json,
        runhistory.json, intensifier.json, and optimization.json.

        Example:
        "smac3_output/my_experiment/16"

    facade : {"hpo", "ac"} | None
        Facade type used in the original run. If None, it is inferred from
        scenario.json.

    Returns
    -------
    dict
        Dictionary with:
        - scenario
        - runhistory
        - intensifier
        - optimization_stats
    """
    run_dir = Path(run_dir)

    scenario = Scenario.load(run_dir)

    runhistory = RunHistory()
    runhistory.load(
        run_dir / "runhistory.json",
        configspace=scenario.configspace,
    )

    if facade is None:
        facade_name = scenario.meta["facade"]["name"]
        if facade_name == "HyperparameterOptimizationFacade":
            facade = "hpo"
        elif facade_name == "AlgorithmConfigurationFacade":
            facade = "ac"
        else:
            raise ValueError(f"Unsupported facade in scenario metadata: {facade_name!r}")

    if facade == "hpo":
        FacadeClass = HPOFacade
    elif facade == "ac":
        FacadeClass = ACFacade
    else:
        raise ValueError("facade must be either 'hpo' or 'ac'")

    intensifier_meta = scenario.meta["intensifier"]

    intensifier = FacadeClass.get_intensifier(
        scenario=scenario,
        max_config_calls=intensifier_meta.get("max_config_calls"),
        max_incumbents=intensifier_meta.get("max_incumbents"),
    )

    intensifier.runhistory = runhistory
    intensifier.load(run_dir / "intensifier.json")

    with open(run_dir / "optimization.json") as fh:
        optimization_stats = json.load(fh)

    return {
        "scenario": scenario,
        "runhistory": runhistory,
        "intensifier": intensifier,
        "optimization_stats": optimization_stats,
    }


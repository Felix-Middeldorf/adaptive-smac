from smac import Scenario
from smac import HyperparameterOptimizationFacade as HPOFacade
from ConfigSpace import ConfigurationSpace,Configuration,Float
from pathlib import Path
import submitit

SLURM_PARTITION = "c23ms"

def run():

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
        seed=0,
        deterministic=True
    )

    smac = HPOFacade(
        scenario=sc,
        target_function=eval
    )

    incumbent = smac.optimize()
    rh = smac.runhistory

    return print(rh.get_cost(incumbent))

def submit_jobs() -> None:
    executor = submitit.AutoExecutor(
        folder="/home/io632776/experiments/adaptive-smac/experiments/hartmann/Debug_04/log",
        cluster="slurm",
        slurm_max_num_timeout=1000,
    )
    executor.update_parameters(
        timeout_min=60 * 24,
        slurm_partition=SLURM_PARTITION,
        cpus_per_task=1,
        mem_gb=2.4,
        slurm_job_name="simple_smac",
        slurm_additional_parameters={"requeue": True},
    )

    job = executor.submit(run)
    print(f"submitted: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()




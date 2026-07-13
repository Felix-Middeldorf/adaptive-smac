from pathlib import Path
import sys
import submitit

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
from experiments.synthaticBench.o6_multi_modal.depth_policies.o6_dimension_runner import STAGED_POLICY, run_policy

SMAC_SEEDS = range(5)
DIMENSION = 10
N_INSTANCES = 30
PROBLEM_SEED = 52
N_TRIALS = 1500
PYTHONHASHSEED = "12345"
OUTPUT_DIRECTORY = HERE / "smac_output"


def submit_jobs() -> None:
    executor = submitit.AutoExecutor(folder=str(HERE / "submitit_logs" / "staged"), cluster="slurm", slurm_max_num_timeout=1000)
    executor.update_parameters(timeout_min=20, slurm_partition="c23ms", slurm_array_parallelism=5, cpus_per_task=1, mem_gb=4, slurm_job_name="SynthACtic_O6_I30_StagedDepth", slurm_setup=[f"export PYTHONHASHSEED={PYTHONHASHSEED}"], slurm_additional_parameters={"requeue": True})
    jobs = []
    with executor.batch():
        for seed in SMAC_SEEDS:
            jobs.append((seed, executor.submit(run_policy, STAGED_POLICY, seed, PROBLEM_SEED, DIMENSION, OUTPUT_DIRECTORY, N_TRIALS, N_INSTANCES)))
    for seed, job in jobs:
        print(f"seed={seed}: {job.job_id}")


if __name__ == "__main__":
    submit_jobs()

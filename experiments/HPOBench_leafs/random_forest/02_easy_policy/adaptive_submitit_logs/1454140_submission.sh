#!/bin/bash

# Parameters
#SBATCH --array=0-26%20
#SBATCH --cpus-per-task=1
#SBATCH --error=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/HPOBench/random_forest/02_easy_policy/adaptive_submitit_logs/%A_%a_0_log.err
#SBATCH --job-name=HPOBench_RF_AdaptiveLeaf
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/HPOBench/random_forest/02_easy_policy/adaptive_submitit_logs/%A_%a_0_log.out
#SBATCH --partition=c23ms
#SBATCH --requeue
#SBATCH --signal=USR2@90
#SBATCH --time=1440
#SBATCH --wckey=submitit

# setup
export PYTHONHASHSEED=12345

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/HPOBench/random_forest/02_easy_policy/adaptive_submitit_logs/%A_%a_%t_log.out --error /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/HPOBench/random_forest/02_easy_policy/adaptive_submitit_logs/%A_%a_%t_log.err /home/io632776/work/py-envs/py3.12-smac/bin/python -u -m submitit.core._submit /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/HPOBench/random_forest/02_easy_policy/adaptive_submitit_logs

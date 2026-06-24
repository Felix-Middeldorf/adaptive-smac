#!/bin/bash

# Parameters
#SBATCH --array=0-29%30
#SBATCH --cpus-per-task=1
#SBATCH --error=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/logs_fixed_vs_learned/%A_%a_0_log.err
#SBATCH --job-name=HartmannFixedVsLearned
#SBATCH --mem=2457MB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/logs_fixed_vs_learned/%A_%a_0_log.out
#SBATCH --requeue
#SBATCH --signal=USR2@90
#SBATCH --time=1440
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/logs_fixed_vs_learned/%A_%a_%t_log.out --error /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/logs_fixed_vs_learned/%A_%a_%t_log.err /home/io632776/work/py-envs/py3.12-smac/bin/python -u -m submitit.core._submit /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/logs_fixed_vs_learned

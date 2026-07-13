#!/bin/bash

# Parameters
#SBATCH --array=0-69%70
#SBATCH --cpus-per-task=1
#SBATCH --error=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment/submitit_logs_consensus_ramp/%A_%a_0_log.err
#SBATCH --job-name=SynthACtic_O1_ConsensusRamp
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment/submitit_logs_consensus_ramp/%A_%a_0_log.out
#SBATCH --partition=c23ms
#SBATCH --requeue
#SBATCH --signal=USR2@90
#SBATCH --time=240
#SBATCH --wckey=submitit

# setup
export PYTHONHASHSEED=12345
export PYTHONPATH='/rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment':$PYTHONPATH

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment/submitit_logs_consensus_ramp/%A_%a_%t_log.out --error /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment/submitit_logs_consensus_ramp/%A_%a_%t_log.err /home/io632776/work/py-envs/adaptive-smac-synthactic-py311/bin/python -u -m submitit.core._submit /rwthfs/rz/cluster/home/io632776/experiments/adaptive-smac/experiments/synthaticBench/o1_deterministic/depth_policies/09_big_experiment/submitit_logs_consensus_ramp

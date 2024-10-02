#!/bin/bash

#SBATCH --job-name=ring_train     # Job name
#SBATCH --output=logs/train_%j.out  # Output file for logs (%j will be replaced by job ID)
#SBATCH --error=logs/train_%j.err   # Error file for logs
#SBATCH --time=72:00:00             # Total runtime limit (HH:MM:SS)
#SBATCH --cpus-per-task=32           # Number of CPU cores per task

# Activate Conda environment
source activate sumo_py310

# Set environment variables if needed
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0 for training

python $F/ring.py . "e=True" "warmup_steps=2000" "skip_stat_steps=5000" \
"horizon=1000" "circumference=250" "n_steps=10" "n_rollouts_per_step=1" \
"skip_vehicle_info_stat_steps=False" "full_rollout_only=True" \
"result_save=$F/evaluations/test.csv" "vehicle_info_save=trajectories/test.npz" \
"save_agent=True"

#!/bin/bash

#SBATCH --job-name=ring_train     # Job name
#SBATCH --output=logs/train_%j.out  # Output file for logs (%j will be replaced by job ID)
#SBATCH --error=logs/train_%j.err   # Error file for logs
#SBATCH --time=72:00:00             # Total runtime limit (HH:MM:SS)
#SBATCH --cpus-per-task=32           # Number of CPU cores per task


# Activate Conda environment
source activate sumo_py310

# Run the training script
python $F/ring.py pareto/single_ring/seeding/beta1.0_SSM1_torch23558_np1409397498 \
    "worker_kwargs=[{'circumference': 250}]" "n_workers=45" "n_rollouts_per_step=45" \
    "warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "global_reward=True" \
    "n_steps=400" "alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=1.0" \
    "scale_ttc=1" "scale_drac=1" "seed_np=1409397498" "seed_torch=23558" \
    "residual_transfer=False" "mrtl=False" "handcraft=False" "step_save=False" \
    "lr=0.0001" "wb=False" "tb=False"

echo "Training job completed!"

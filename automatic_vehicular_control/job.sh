#!/bin/sh

#SBATCH -o output-%j.log
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:volta:1
#SBATCH -c 4

python $F/ring.py . "worker_kwargs=[{'circumference':250}]" "n_workers=10" "n_rollouts_per_step=10" "warmup_steps=500" "skip_stat_steps=2000" "horizon=2000" "global_reward=True" "n_steps=100" "alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=0.99" "save_agent=True" "vehicle_info_save=$R/single_ring/test/trajectories/test1.npz"

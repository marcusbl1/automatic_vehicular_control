#!/bin/sh

#SBATCH -o pareto/single_ring/seeding/beta1.0_SSM1_torch484071_np3550590767/train.log
#SBATCH --time=72:00:00          # Total run time limit (HH:MM:SS)
#SBATCH -c 45

python $F/ring.py pareto/single_ring/seeding/beta1.0_SSM1_torch484071_np3550590767 "worker_kwargs=[{'circumference': 250}]" "n_workers=45" "n_rollouts_per_step=45" "warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "global_reward=True" "n_steps=400" "alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=1.0" "scale_ttc=1" "scale_drac=1" "seed_np=3550590767" "seed_torch=484071" "residual_transfer=False" "mrtl=False" "handcraft=False" "step_save=False" "lr=0.0001" "wb=False" "tb=False" 
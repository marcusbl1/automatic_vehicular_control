#!/bin/sh

#SBATCH -o pareto/single_ring/from_scratch/0.9_0/train0.9_0.log
#SBATCH --time=72:00:00          # Total run time limit (HH:MM:SS)
#SBATCH -c 45

python $F/ring.py pareto/single_ring/from_scratch/0.9_0 "worker_kwargs=[{'circumference': 250}]" "n_workers=45" "n_rollouts_per_step=45" "warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "global_reward=True" "n_steps=150" "alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=0.9" "scale_ttc=0.8" "scale_drac=0.8" "seed_np=1228644375" "seed_torch=106619" "residual_transfer=False" "mrtl=False" "handcraft=False" "step_save=False" "lr=0.0001" "wb=False" "tb=False" 
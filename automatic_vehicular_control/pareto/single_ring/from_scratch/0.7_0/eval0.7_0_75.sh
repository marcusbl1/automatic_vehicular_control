#!/bin/sh

#SBATCH -o pareto/single_ring/from_scratch/0.7_0/eval0.7_0_75.log
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -c 40

python $F/ring.py pareto/single_ring/from_scratch/0.7_0 "e=75" "warmup_steps=2000" "skip_stat_steps=5000" "horizon=1000" "circumference=250" "n_steps=10" "n_rollouts_per_step=1" "skip_vehicle_info_stat_steps=False" "full_rollout_only=True" "result_save=pareto/single_ring/from_scratch/0.7_0/eval_e75.csv" "vehicle_info_save=pareto/single_ring/from_scratch/0.7_0/trajectory.npz" "save_agent=True" "beta=0.7" "scale_ttc=0.8" "scale_drac=0.8" "residual_transfer=False" "mrtl=False"
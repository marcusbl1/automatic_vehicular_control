# Project

You may find this project at: [Project Website](https://mit-wu-lab.github.io/automatic_vehicular_control), [IEEE Website](https://ieeexplore.ieee.org/document/9765650), [arXiv](https://arxiv.org/abs/2208.00268).

```
@article{yan2022unified,
  title={Unified Automatic Control of Vehicular Systems With Reinforcement Learning},
  author={Yan, Zhongxia and Kreidieh, Abdul Rahman and Vinitsky, Eugene and Bayen, Alexandre M and Wu, Cathy},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2022},
  publisher={IEEE}
}
```

# Environment Setup

This project uses Python with several dependencies managed by `conda`. Follow the instructions below to set up your development environment.

Updated date: Oct 2nd 2024.


## Requirements

To use this project, ensure you have the following installed:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [SUMO](https://sumo.dlr.de/docs/Installing/MacOS_Build.html)

## 1. Setup

### 1.1 Create the Conda Environment

Use the `environment.yml` file to create a new `conda` environment with all required dependencies. Run the following command:

```sh
conda env create -f environment.yml
```

### 1.2 Set the environmental variables
```
# Code directory
export F=automatic_vehicular_control

# Results directory extracted from the zip file
export R=results
```

## 2. Directory Structure

The code directory structure is
``` 
automatic_vehicular_control/
│   ├── __pycache__/                # Compiled Python files
│   ├── evaluations/                # Evaluation results and metrics
│   ├── models/                     # Model checkpoints
│   ├── pareto/
│   │   └── single_ring/            # Experiments for the single-ring traffic scenario
│   │       ├── from_scratch/       # Training from scratch experiments
│   │       ├── seeding/            # Experiments with different random seeds
│   │       └── ssm_scaling_5/      # Experiments with different SSM scaling weights
│   ├── sumo/                       # SUMO simulation-related files
│   ├── __init__.py                 # Package initialization script
│   ├── *.log                       # Log files for experiment runs
│   ├── actual_runs.ipynb           # Jupyter notebook for executing and plotting experiments
│   ├── config.yaml                 # Configuration file for experiments
│   ├── env.py                      # Environment setup and classes
│   ├── environment.yml             # Conda environment setup
│   ├── eval_commands.sh            # Shell script for running evaluations
│   ├── exp.py                      # Experiment setup script
│   ├── ring.py                     # Main script for running the ring road environment
│   ├── ut.py                       # Definie algo and NN related func
│   ├── u.py                        # Definie help func
```


## 3. Code running
### 3.1 Training Command:
```
python $F/ring.py $F/pareto/single_ring/seeding/beta1.0_SSM1_torch23558_np1409397498 \
"av=1" "circumference=200" "n_workers=45" "n_rollouts_per_step=45" \
"warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "global_reward=True" "n_steps=400" \
"alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=1.0" "scale_ttc=1" "scale_drac=1" \
"seed_np=1409397498" "seed_torch=23558" "residual_transfer=False" "mrtl=False" \
"handcraft=False" "step_save=False" "lr=0.0001" "wb=False" "tb=False" 
```

#### Explanation of Arguments:
- **$F/ring.py**: Path to the main running script.
- **pareto/single_ring/seeding/beta1.0_SSM1_torch23558_np1409397498**: Output directory for storing results.
- **worker_kwargs**: Configuration details for workers
- **n_workers**: Number of workers to use in parallel for rollouts.  
- **n_rollouts_per_step**: Number of rollouts per training step.  
- **warmup_steps**: Number of warmup steps before training begins.  
- **skip_stat_steps**: Number of steps to skip before collecting statistical data.  
- **horizon**: Number of simulation steps in an episode.  
- **global_reward**: Whether to use a global reward for training (`True` or `False`).
- **n_steps**: Number of gradient update steps during training.  
- **alg**: The RL algorithm to be used.  
- **use_critic**: Specify if a critic is used (`True` or `False`).
- **gamma**: Discount factor for future rewards.  
- **beta**: Weight for balancing safety and performance metrics.  
- **scale_ttc**, **scale_drac**: Scaling factors for safety measures like Time to Collision (TTC) and Deceleration Rate to Avoid Collision (DRAC).
- **seed_np**, **seed_torch**: Random seeds for reproducibility.  
- **residual_transfer**, **mrtl**, **handcraft**: Additional training configurations.
- **step_save**: Whether to save the model at each training step (`True` or `False`).
- **lr**: Learning rate for training.  
- **wb**, **tb**: Enable or disable logging with Weights & Biases (`wb`) and TensorBoard (`tb`) (`True` or `False`).

Each parameter has a specific role in controlling the training process, and modifying them can lead to different training outcomes, depending on the training scenario and requirements.

### 3.2 Evaluation Command:

```
python $F/ring.py . "e=True" "warmup_steps=2000" "skip_stat_steps=5000" \
"horizon=1000" "circumference=250" "n_steps=10" "n_rollouts_per_step=1" \
"skip_vehicle_info_stat_steps=False" "full_rollout_only=True" \
"result_save=$F/evaluations/test.csv" "vehicle_info_save=trajectories/test.npz" \
"save_agent=True"
```

#### Explanation of Arguments

- **$F/ring.py**: Path to the running script.
- **"."**: Represents the current directory as the output directory for storing evaluation results.
- **e**: Specifies evaluation mode (`True`). This enables evaluation without further training.
- **warmup_steps**: Number of warmup steps before starting the evaluation.
- **skip_stat_steps**: Number of steps to skip before collecting statistical data during evaluation.
- **horizon**: Number of simulation steps in an episode for evaluation.
- **circumference**: Circumference of the ring road environment.
- **n_steps**: Number of evaluation steps to be performed.
- **n_rollouts_per_step**: Number of rollouts per evaluation step.
- **skip_vehicle_info_stat_steps**: Whether to skip recording vehicle information statistics for the first few steps (`True` or `False`).
- **full_rollout_only**: Whether to consider only complete rollouts for evaluation (`True` or `False`).
- **result_save**: Path to save the evaluation results as a CSV file.
- **vehicle_info_save**: Path to save vehicle trajectory data in `.npz` format.
- **save_agent**: Whether to save the agent's information (`True` or `False`).


### 3.3 Running IDM without RL 
```
python $F/ring.py $F/pareto/single_ring/IDM/different_veh \
"av=0" "circumference=200" "n_workers=45" "n_rollouts_per_step=45" \
"warmup_steps=2000" "skip_stat_steps=5000" "horizon=5000" "global_reward=True" "n_steps=400" \
"alg='TRPO'" "use_critic=False" "gamma=0.9995" "beta=1.0" "scale_ttc=1" "scale_drac=1" \
"seed_np=1409397498" "seed_torch=23558" "residual_transfer=False" "mrtl=False" \
"handcraft=False" "step_save=False" "lr=0.0001" "wb=False" "tb=False" 

- **av**: set to 0.

```

### 3.4 Batch Running with Slurm:
Training:
```
sbatch train_commands.sh
```

Evaluation:
```
sbatch eval_commands.sh
```

## 4. TODO
1. IDM debug finished, check the RL bugs
2. Test batch running and check if the plot is right: Training with 50 veh one lane of 1000m on Super Cloud. 
3. Try different veh number under the scenarios of w or w/o RL and compare if the fundamental diagram get improved Super Cloud.


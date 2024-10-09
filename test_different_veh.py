from ring import *  # Import all from experimental utilities
from exp import *  # Import all from experimental utilities
from env import *  # Import all from environment utilities
from u import *    # Import all from utility functions



def test_with_different_vehicle_numbers(start=20, end=50, step=1):
    """Run tests with different numbers of vehicles in the environment."""
    for n_veh in range(start, end + 1, step):
        print(f"Running simulation with {n_veh} vehicles")
        # Set up the configuration for a one-lane ring road with different vehicle counts
        c = Ring.from_args(globals(), locals()).setdefaults(
            n_lanes=1,             # Number of lanes in the ring road
            horizon=3000,          # Total number of simulation steps
            warmup_steps=1000,     # Number of steps before RL control starts
            sim_step=0.1,          # Simulation time step
            av=1,                  # Number of autonomous vehicles
            max_speed=10,          # Maximum vehicle speed
            max_accel=0.5,         # Maximum acceleration
            max_decel=0.5,         # Maximum deceleration
            circumference=250,     # Circumference of the ring road
            circumference_max=300, # Maximum circumference
            circumference_min=200, # Minimum circumference
            circumference_range=None,  # Range for random circumference
            initial_space='free',      # Initial vehicle spacing: free typically indicate that vehicles are placed with some randomness, meaning the exact initial positions are not fixed but instead have some random variation.
            sigma=0.2,                 # Driver imperfection parameter

            circ_feature=False,    # Include circumference in observations
            accel_feature=False,   # Include acceleration in observations
            act_type='accel',      # Type of action (acceleration control)
            lc_act_type='discrete',# Type of lane change action
            low=-1,                # Minimum action value
            high=1,                # Maximum action value
            norm_action=True,      # Normalize action values
            global_reward=False,   # Use global reward (all vehicles)
            accel_penalty=0,       # Penalty for acceleration changes
            collision_penalty=100, # Penalty for collisions

            n_steps=100,           # Number of training steps
            gamma=0.999,           # Discount factor
            alg=PG,                # Learning algorithm (Policy Gradient)
            norm_reward=True,      # Normalize rewards
            center_reward=True,    # Center rewards
            adv_norm=False,        # Normalize advantages
            step_save=None,        # Steps between saving models

            render=False,          # Render the simulation

            beta=0,                # Weight for safety measures in reward
            scale_ttc=1,           # Scaling factor for TTC
            scale_pet=1,           # Scaling factor for PET
            scale_drac=1,          # Scaling factor for DRAC
            seed_np=False,         # Seed for NumPy
            seed_torch=False,      # Seed for PyTorch
            residual_transfer=False, # Modify which network (nominal/residual)
            mrtl=False,            # Include beta in observations
        )
        c.res = c.res +"/veh_"+str(n_veh)+"/"
        os.makedirs(c.res, exist_ok=True)

        # Set seeds for reproducibility if specified
        if c.seed_torch:
            torch.manual_seed(c.seed_torch)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(c.seed_torch)
        if c.seed_np:
            np.random.seed(c.seed_np)

        # Set number of vehicles and observation size based on the number of lanes
        if c.n_lanes == 1:
            c.setdefaults(
                n_veh=n_veh,  # Number of vehicles
                _n_obs=3 + c.circ_feature + c.accel_feature  # Observation size
            )
        elif c.n_lanes == 2:
            c.setdefaults(
                n_veh= int(n_veh*2),
                lc_mode=LC_MODE.no_lat_collide, # No lateral collision mode
                symmetric=False,
                symmetric_action=None,
                lc_av=2
            )
            c._n_obs = (1 + 2 * 2 * 2) if c.symmetric else (1 + 2 * 5)
        elif c.n_lanes == 3:
            c.setdefaults(
                n_veh=66,
                lc_mode=LC_MODE.no_lat_collide, # No lateral collision mode
                symmetric=False,
                symmetric_action=None,
                lc_av=3,
                _n_obs=1 + 3 * (1 + 2 * 2)
            )
        if c.mrtl:
            c._n_obs += 1  # Increase observation size for mrtl

        c.step_save = c.step_save or min(5, c.n_steps // 10)  # Set step save interval
        c.redef_sumo = bool(c.circumference_range)  # Redefine sumo if random circumference range is given

        # Run the environment with the set configuration
        c.run()  # Assuming the RingEnv class has a run method

        print(f"Simulation with {n_veh} vehicles completed")

# Main entry point to define configurations and run the experiment
if __name__ == '__main__':
    # Run the test with different numbers of vehicles
    test_with_different_vehicle_numbers(start=20, end=40, step=5)

    print("All simulations completed")




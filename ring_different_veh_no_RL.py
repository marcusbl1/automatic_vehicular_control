# from automatic_vehicular_control.exp import *  # Import all from experimental utilities
# from automatic_vehicular_control.env import *  # Import all from environment utilities
# from automatic_vehicular_control.u import *    # Import all from utility functions

from exp import *  # Import all from experimental utilities
from env import *  # Import all from environment utilities
from u import *    # Import all from utility functions
import os

os.environ['F'] = 'automatic_vehicular_control'


'''
    RingEnv is a subclass of the Env class and represents the simulation environment 
    where the vehicles (human-driven or RL-controlled) interact on a ring road.
'''
class RingEnv(Env):
    def def_sumo(self):
        c = self.c  # Access configuration object
        r = c.circumference / (2 * np.pi)  # Calculate the radius of the ring road based on circumference
        nodes = E('nodes',
            E('node', id='bottom', x=0, y=-r),  # Define bottom node at coordinates (0, -r)
            E('node', id='top', x=0, y=r),      # Define top node at coordinates (0, r)
        )
        # Function to generate the shape of the edges (circular arcs)
        get_shape = lambda start_angle, end_angle: ' '.join('%.5f,%.5f' % (r * np.cos(i), r * np.sin(i)) for i in np.linspace(start_angle, end_angle, 80))

        edges = E('edges',
            E('edge', **{
                'id': 'right',
                'from': 'bottom',
                'to': 'top',
                'length': c.circumference / 2,  # Edge length is half the circumference
                'shape': get_shape(-np.pi / 2, np.pi / 2),  # Define the right side of the ring road
                'numLanes': c.n_lanes  # Number of lanes
            }),
            E('edge', **{
                'id': 'left',
                'from': 'top',
                'to': 'bottom',
                'length': c.circumference / 2,  # Define left side of the ring road
                'shape': get_shape(np.pi / 2, np.pi * 3 / 2),
                'numLanes': c.n_lanes
            }),
        )

        connections = E('connections',
            # Create lane connections from 'left' to 'right' and vice versa for all lanes
            *[E('connection', **{
                'from': 'left',
                'to': 'right',
                'fromLane': i,
                'toLane': i
            }) for i in range(c.n_lanes)],
            *[E('connection', **{
                'from': 'right',
                'to': 'left',
                'fromLane': i,
                'toLane': i
            }) for i in range(c.n_lanes)],
        )

        additional = E('additional',
            # Define the vehicle type for human-driven vehicles using specific driving parameters
            E('vType', id='human', **{
                **IDM, **LC2013, **dict(
                    accel=1,
                    decel=1.5,
                    minGap=2,
                    sigma=c.sigma  # Driver imperfection parameter
                )
            }),
            # Define the vehicle type for RL-controlled vehicles
            # E('vType', id='rl', **{
            #     **IDM, **LC2013, **dict(
            #         accel=1,
            #         decel=1.5,
            #         minGap=2,
            #         sigma=0  # No randomness in RL vehicles
            #     )
            # }),
            # Build a closed route on the ring road with the specified number of vehicles
            *build_closed_route(
                edges,
                n_veh = c.n_veh,
                av = c.av,
                space=c.initial_space
            )
        )
        return super().def_sumo(nodes, edges, connections, additional)  # Call the base class method

    def reset_sumo(self):
        c = self.c
        if c.circumference_range:  # Check if the circumference should be randomized
            # Randomly set the circumference within the specified range
            c.circumference = np.random.randint(*c.circumference_range)
        return super().reset_sumo()  # Reset the simulation

    @property
    def stats(self):
        c = self.c # Get the configuration object
        # Gather environment statistics excluding flow-related stats
        stats = {
            k: v
            for k, v in super().stats.items()
            if 'flow' not in k
        }
        stats['circumference'] = c.circumference  # Add the current circumference to the stats
        stats['beta'] = c.beta                    # Add the beta parameter
        return stats
        
    def step(self, action=None):
        c = self.c  # Configuration object
        ts = self.ts  # Traffuc state object
        c, ts = self.c, self.ts
        super().step_2()
        return c.observation_space.low, 0, False, None

    # Calculate the average Time to Collision (TTC) for all vehicles
    def calc_ttc(self):
        cur_veh_list = self.ts.vehicles
        ttcs = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            leader_speed = leader.speed
            if leader_speed < v_speed:
                ttc = headway / (v_speed - leader_speed)  # Calculate TTC if leader speed is less than current vehicle speed
            else:
                ttc = np.nan  # TTC is undefined if leader is faster or at the same speed
            ttcs.append(ttc)
        fleet_ttc = np.nanmean(np.array(ttcs))  # Average TTC, ignoring NaN values
        return fleet_ttc

    # Calculate the average Deceleration Rate to Avoid Collision (DRAC) for all vehicles
    def calc_drac(self):
        cur_veh_list = self.ts.vehicles
        dracs = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            leader_speed = leader.speed
            drac = 0.5 * np.square(v_speed - leader_speed) / headway  # Calculate DRAC based on speed difference and headway
            dracs.append(drac)
        fleet_drac = np.nanmean(np.array(dracs))  # Average DRAC, ignoring NaN values
        return fleet_drac

    # Calculate the average Post-Encroachment Time (PET) for all vehicles
    def calc_pet(self):
        cur_veh_list = self.ts.vehicles
        pets = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            if v_speed > 1e-16:  # Avoid division by zero
                pet = headway / v_speed
                pets.append(pet)
        fleet_pet = np.nanmean(np.array(pets))  # Average PET, ignoring NaN values
        return fleet_pet


'''
    Ring is a subclass of Main, and it represents the configuration and control structure for running the simulation. 
    It defines how the environment is created and configured, 
    including parameters such as number of vehicles, observation space, action space, and various simulation settings.
    This class is primarily responsible for setting up and managing the parameters and settings of the environment (RingEnv) before running the simulation. 
    It configures the simulation based on arguments passed at runtime and ensures that the environment is properly set up for training or testing.
'''
class Ring(Main):
    # Method to create and return the environment
    def create_env(c):
        return NormEnv(c, RingEnv(c))

    # Property to define the observation space for the RL agent
    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(c._n_obs,), dtype=np.float32)

    # Property to define the action space for the RL agent
    @property
    def action_space(c):
        c.setdefaults(lc_av=False)  # Set default for lane changing availability
        # Ensure the action type is valid
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'accel_discrete']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            if not c.lc_av or c.lc_act_type == 'continuous':
                # Continuous action space with optional lane change
                return Box(low=c.low, high=c.high, shape=(1 + bool(c.lc_av),), dtype=np.float32)
            elif c.lc_act_type == 'discrete':
                # Discrete lane change action
                return Namespace(
                    accel=Box(low=c.low, high=c.high, shape=(1,), dtype=np.float32),
                    lc=Discrete(c.lc_av)
                )
        elif c.act_type in ['discrete', 'accel_discrete']:
            if c.lc_av:
                # Discrete acceleration and lane change actions
                return Namespace(
                    accel=Discrete(c.n_actions),
                    lc=Discrete(c.lc_av)
                )
            return Discrete(c.n_actions)

    # Override on_train_start to set last layer biases if specified
    def on_train_start(c):
        super().on_train_start()
        if c.get('last_unbiased'):
            # Set last layer biases to zero for specific actions
            c._model.p_head[-1].bias.data[c.lc_av:] = 0

    # Override on_step_end to maintain biases during training if specified
    def on_step_end(c, gd_stats):
        super().on_step_end(gd_stats)
        if c.get('last_unbiased'):
            c._model.p_head[-1].bias.data[c.lc_av:] = 0

if __name__ == '__main__':
    # Run the test with different numbers of vehicles
    for n_veh in range(20, 120, 10):
        print(f"Running simulation with {n_veh} vehicles")
        # Set up the configuration for a one-lane ring road with different vehicle counts
        c = Ring.from_args(globals(), locals()).setdefaults(
            n_lanes=1,             # Number of lanes in the ring road
            horizon=3000,          # Total number of simulation steps
            warmup_steps=1000,     # Number of steps before RL control starts
            sim_step=0.1,          # Simulation time step
            av=0,                  # Number of autonomous vehicles
            max_speed=10,          # Maximum vehicle speed
            max_accel=0.5,         # Maximum acceleration
            max_decel=0.5,         # Maximum deceleration
            circumference=1000,     # Circumference of the ring road
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

        c.step_save = c.step_save or min(5, c.n_steps // 10)  # Set step save interval
        c.redef_sumo = bool(c.circumference_range)  # Redefine sumo if random circumference range is given

        # Run the environment with the set configuration
        c.run_2()  # Assuming the RingEnv class has a run method

        print(f"Simulation with {n_veh} vehicles completed")






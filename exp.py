# from automatic_vehicular_control.u import Config
# from automatic_vehicular_control.ut import *

from u import Config
from ut import *
import numpy as np


class Main(Config):
    # Set the base path for flow results, using the environment variable 'F'
    flow_base = Path.env('F')._real

    # Define a set of keys that should never be saved in the configuration
    never_save = {'trial', 'has_av', 'e', 'disable_amp', 'opt_level', 'tmp'} | Config.never_save

    def __init__(c, res, *args, **kwargs):
        # Determine if the result path 'res' is a temporary directory
        tmp = Path(res)._real in [Path.env('HOME'), Main.flow_base, Main.flow_base / 'all_results']
        if tmp:
            # If it's a temp directory, redirect to a random temp subdirectory
            res = Main.flow_base / 'tmp' / rand_string(8)
        # Set default configurations
        kwargs.setdefault('disable_amp', True)
        kwargs.setdefault('tmp', tmp)
        super().__init__(res, *args, **kwargs)
        if tmp:
            # Use dry run mode for Weights & Biases if in a temp run
            os.environ['WANDB_MODE'] = 'dryrun'
        # Set default values for evaluation, TensorBoard, and Weights & Biases logging
        c.setdefaults(e=False, tb=True, wb=False)
        # Disable logger if evaluation mode is False
        c.logger = c.logger and c.e is False
        if tmp and c.e is False:
            # Create the result directory and log a message
            res.mk()
            c.log('Temporary run for testing with res=%s' % res)

    def create_env(c):
        # This method should be implemented in subclasses to create the environment
        raise NotImplementedError

    @property
    def dist_class(c):
        # Lazy initialization of the distribution class based on the action space
        if '_dist_class' not in c:
            c._dist_class = build_dist(c.action_space)
        return c._dist_class

    @property
    def model_output_size(c):
        # Get the model output size from the distribution class
        if '_model_output_size' not in c:
            c._model_output_size = c.dist_class.model_output_size
        return c._model_output_size

    @property
    def observation_space(c):
        # Should be implemented in subclasses to define the observation space
        raise NotImplementedError

    @property
    def action_space(c):
        # Should be implemented in subclasses to define the action space
        raise NotImplementedError

    def set_model(c):
        # Initialize the model using the specified model class or default to FFN
        c._model = c.get('model_cls', FFN)(c)
        return c

    def schedule(c, coef, schedule=None):
        # Compute a scheduled coefficient based on the current step
        if not schedule and isinstance(coef, (float, int)):
            return coef
        frac = c._i / c.n_steps
        frac_left = 1 - frac
        if callable(coef):
            # If coef is a callable function, use it with frac_left
            return coef(frac_left)
        elif schedule == 'linear':
            # Linear scheduling
            return coef * frac_left
        elif schedule == 'cosine':
            # Cosine scheduling
            return coef * (np.cos(frac * np.pi) + 1) / 2

    @property
    def _lr(c):
        # Compute the current learning rate using the schedule
        return c.schedule(c.get('lr', 1e-4), c.get('lr_schedule'))

    def log_stats(c, stats, ii=None, n_ii=None, print_time=False):
        # Log statistics, optionally including iteration info and total time
        stats = {k: v for k, v in stats.items() if v is not None}
        total_time = time() - c._run_start_time
        if print_time:
            stats['total_time'] = total_time

        prints = []
        if ii is not None:
            prints.append('ii {:2d}'.format(ii))

        # Format each stat for printing
        prints.extend('{} {:.3g}'.format(*kv) for kv in stats.items())

        # Determine terminal width for proper formatting
        widths = [len(x) for x in prints]
        line_w = terminal_width()
        prefix = 'i {:d}'.format(c._i)
        i_start = 0
        curr_w = len(prefix) + 3
        curr_prefix = prefix
        # Loop to format and print the stats within terminal width
        for i, w in enumerate(widths):
            if curr_w + w > line_w:
                c.log(' | '.join([curr_prefix, *prints[i_start: i]]))
                i_start = i
                curr_w = len(prefix) + 3
                curr_prefix = ' ' * len(prefix)
            curr_w += w + 3
        c.log(' | '.join([curr_prefix, *prints[i_start:]]))
        sys.stdout.flush()

        if ii is not None:
            # Buffer the stats for writing later
            c._writer_buffer.append(**stats)
            return

        # Add the stats to the results DataFrame
        c.add_stats(stats)

    def flush_writer_buffer(c):
        # Flush the buffered stats and add them to the results
        if len(c._writer_buffer):
            stats = {k: np.nanmean(c._writer_buffer.pop(k)) for k in list(c._writer_buffer)}
            c.add_stats(stats)

    def add_stats(c, stats):
        # Add stats to the DataFrame and log to TensorBoard or Weights & Biases
        total_time = time() - c._run_start_time
        df = c._results
        for k, v in stats.items():
            if k not in df:
                df[k] = np.nan
            df.loc[c._i, k] = v

        if c.e is False:
            if c.tb:
                # Log to TensorBoard
                for k, v in stats.items():
                    c._writer.add_scalar(k, v, global_step=c._i, walltime=total_time)
            if c.wb:
                # Log to Weights & Biases
                c._writer.log(stats, step=c._i)

    def get_log_ii(c, ii, n_ii=None, print_time=False):
        # Return a logging function for a specific iteration
        return lambda **kwargs: c.log_stats(kwargs, ii, print_time=print_time)

    def on_rollout_worker_start(c):
        # Initialize the environment and model for a rollout worker
        c._env = c.create_env()
        c.use_critic = False  # Don't need value function on workers
        c.set_model()
        c._model.eval()
        c._i = 0

    def set_weights(c, weights):  # For Ray
        # Load model weights into the worker
        c._model.load_state_dict(weights, strict=False)  # If c.use_critic, worker may not have critic weights

    def on_train_start(c):
        # Initialize training: environment, algorithm, model, optimizer, and logging
        c.setdefaults(alg='Algorithm') 
        c._env = c.create_env() # Create NormEnv
 
        # Instantiate the algorithm (e.g., PPO, DQN)
        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.train()
        c._model.to(c.device)

        c._i = 0  # for c._lr
        opt = c.get('opt', 'Adam')
        if opt == 'Adam':
            # Use Adam optimizer with specified parameters
            c._opt = optim.Adam(c._model.parameters(), lr=c._lr, betas=c.get('betas', (0.9, 0.999)), weight_decay=c.get('l2', 0))
        elif opt == 'RMSprop':
            # Use RMSprop optimizer
            c._opt = optim.RMSprop(c._model.parameters(), lr=c._lr, weight_decay=c.get('l2', 0))

        c._run_start_time = time()
        # Load existing training state if available
        c._i = c.set_state(c._model, opt=c._opt, step='max')
        if c._i:
            print(c._i)
            c._results = c.load_train_results().loc[:c._i]
            c._run_start_time -= c._results.loc[c._i, 'total_time']
        else:
            # Initialize results DataFrame
            c._results = pd.DataFrame(index=pd.Series(name='step'))
        c._i_gd = None

        # Try to save the current commit hash
        c.try_save_commit(Main.flow_base)

        # Set up logging
        if c.tb:
            from torch.utils.tensorboard import SummaryWriter
            c._writer = SummaryWriter(log_dir=c.res, flush_secs=10)
        if c.wb:
            import wandb
            wandb_id_path = (c.res / 'wandb' / 'id.txt').dir_mk()
            c._wandb_run = wandb.init(  # name and project should be set as env vars
                name=c.res.rel(Path.env('FA')),
                dir=c.res,
                id=wandb_id_path.load() if wandb_id_path.exists() else None,
                config={k: v for k, v in c.items() if not k.startswith('_')},
                save_code=False
            )
            wandb_id_path.save(c._wandb_run.id)
            c._writer = wandb
        c._writer_buffer = NamedArrays()

    def on_step_start(c, stats={}):
        # Update the learning rate and log stats at the start of a training step
        lr = c._lr # Update Learning Rate
        for g in c._opt.param_groups: # Log Statistics
            g['lr'] = float(lr)
        c.log_stats(dict(**stats, **c._alg.on_step_start(), lr=lr))

        if c._i % c.step_save == 0:
            # Periodically save training results and state
            c.save_train_results(c._results)
            c.save_state(c._i, c.get_state(c._model, c._opt, c._i))

    def rollouts(c):
        """ Collect a list of rollouts for the training step """
        if c.use_ray:
            # If using Ray for distributed rollouts
            import ray
            # Distribute the model weights to the workers
            weights_id = ray.put({k: v.cpu() for k, v in c._model.state_dict().items()})
            [w.set_weights.remote(weights_id) for w in c._rollout_workers]
            # Collect rollouts from workers
            rollout_stats = flatten(ray.get([w.rollouts_single_process.remote() for w in c._rollout_workers]))
        else:
            # Collect rollouts in a single process
            rollout_stats = c.rollouts_single_process()
        # Process each rollout and flush the writer buffer
        rollouts = [c.on_rollout_end(*rollout_stat, ii=ii) for ii, rollout_stat in enumerate(rollout_stats)]
        c.flush_writer_buffer()
        return NamedArrays.concat(rollouts, fn=flatten)

    def rollouts_single_process(c):
        if c.n_rollouts_per_worker > 1:
            # Collect multiple rollouts per worker
            rollout_stats = [c.var(i_rollout=i).rollout() for i in range(c.n_rollouts_per_worker)]
        else:
            # Collect rollouts until the total number of steps reaches the horizon
            n_steps_total = 0
            rollout_stats = []
            while n_steps_total < c.horizon:
                if c.get('full_rollout_only'):
                    n_steps_total = 0
                    rollout_stats = []
                rollout, stats = c.rollout()
                rollout_stats.append((rollout, stats))
                n_steps_total += stats.get('horizon') or len(stats.get('reward', []))
        return rollout_stats

    def get_env_stats(c):
        # Retrieve statistics from the environment
        return c._env.stats  # returns env stats() func

    def rollout(c):
        # Perform a single rollout in the environment
        c.setdefaults(skip_stat_steps=0, i_rollout=0, rollout_kwargs=None)
        if c.rollout_kwargs and c.e is False:
            # Update configuration with rollout-specific kwargs
            c.update(c.rollout_kwargs[c.i_rollout])
        t_start = time()

        # Reset the environment and initialize the rollout buffer
        ret = c._env.reset()
        if not isinstance(ret, dict):
            ret = dict(obs=ret)
        rollout = NamedArrays()
        rollout.append(**ret)
        done = False
        a_space = c.action_space
        step = 0
        density = c.var().n_veh/(c.var().circumference*1e-3)
        vehicle_flow = c._env.mean_speed * 3.6 * density 
        print(f"Current vehicle flow: {vehicle_flow:.2f} at new episode")

        while step < c.horizon + c.skip_stat_steps and not done:
            # Generate an action from the model's policy
            pred = from_torch(c._model(to_torch(rollout.obs[-1]), value=False, policy=True, argmax=False))
            if c.get('aclip', True) and isinstance(a_space, Box):
                # Clip the action if necessary
                pred.action = np.clip(pred.action, a_space.low, a_space.high)
            rollout.append(**pred)

            # Take a step in the environment
            ret = c._env.step(rollout.action[-1])
            if isinstance(ret, tuple):
                obs, reward, done, info = ret
                ret = dict(obs=obs, reward=reward, done=done, info=info)
            done = ret.setdefault('done', False)
            if done:
                # Remove unnecessary keys if the episode is done
                ret = {k: v for k, v in ret.items() if k not in ['obs', 'id']}
            rollout.append(**ret)
            step += 1
        # Collect stats from the environment
        # rollout_flow_each_step.append(len(c._env.passed_vehicle))
        # Flow= (Number of vehicles passing a point×3600) / Simulation Time (seconds) 
        vehicle_flow = c._env.mean_speed * density  * 3.6
        print(f"Current vehicle flow: {vehicle_flow:.2f} at step : {step:.2f}")

        stats = dict(rollout_time=time() - t_start, **c.get_env_stats())
        return rollout, stats

    def on_rollout_end(c, rollout, stats, ii=None):
        """ Compute value, calculate advantage, log stats """
        t_start = time()
        step_id_ = rollout.pop('id', None)
        done = rollout.pop('done', None)
        multi_agent = step_id_ is not None

        step_obs_ = rollout.obs
        step_obs = step_obs_ if done[-1] else step_obs_[:-1]
        assert len(step_obs) == len(rollout.reward)

        value_ = None
        if c.use_critic:
            # Compute the value function if using a critic
            (_, mb_), = rollout.filter('obs').iter_minibatch(concat=multi_agent, device=c.device)
            value_ = from_torch(c._model(mb_.obs, value=True).value.view(-1))

        if multi_agent:
            # Handle multi-agent rollout processing
            step_n = [len(x) for x in rollout.reward]
            reward = np.concatenate(rollout.reward)
            ret, adv = calc_adv_multi_agent(np.concatenate(step_id_), reward, c.gamma, value_=value_, lam=c.lam)
            rollout.update(obs=step_obs, ret=split(ret, step_n))
            if c.use_critic:
                rollout.update(value=split(value_[:len(ret)], step_n), adv=split(adv, step_n))
        else:
            # Handle single-agent rollout processing
            reward = rollout.reward
            ret, adv = calc_adv(reward, c.gamma, value_, c.lam)
            rollout.update(obs=step_obs, ret=ret)
            if c.use_critic:
                rollout.update(value=value_[:len(ret)], adv=adv)

        log = c.get_log_ii(ii)
        # Log environment and rollout statistics
        log(**stats)
        log(
            reward_mean=np.mean(reward),
            reward_std=np.std(reward),

            speed_reward_mean=np.mean(rollout.speed_reward) if rollout.speed_reward else None,
            speed_reward_std=np.std(rollout.speed_reward) if rollout.speed_reward else None,
            # Other stats related to safety measures
            ssm_mean=np.mean(rollout.ssm),
            ssm_std=np.std(rollout.ssm),

            ttc_mean=np.mean(rollout.ttc) if rollout.ttc else None,
            ttc_std=np.std(rollout.ttc) if rollout.ttc else None,
            drac_mean=np.mean(rollout.drac) if rollout.drac else None,
            drac_std=np.std(rollout.drac) if rollout.drac else None,
            pet_mean=np.mean(rollout.pet) if rollout.pet else None,
            pet_std=np.std(rollout.pet) if rollout.pet else None,

            raw_ttc_mean=np.mean(rollout.raw_ttc) if rollout.raw_ttc else None,
            raw_ttc_std=np.std(rollout.raw_ttc) if rollout.raw_ttc else None,
            raw_drac_mean=np.mean(rollout.raw_drac) if rollout.raw_drac else None,
            raw_drac_std=np.std(rollout.raw_drac) if rollout.raw_drac else None,

            value_mean=np.mean(value_) if c.use_critic else None,
            ret_mean=np.mean(ret),
            adv_mean=np.mean(adv) if c.use_critic else None,
            explained_variance=explained_variance(value_[:len(ret)], ret) if c.use_critic else None
        )
        log(rollout_end_time = time() - t_start)
        return rollout

    def on_step_end(c, stats={}):
        # Log stats at the end of a training step
        c.log_stats(stats, print_time=True)
        c.log('')

    def on_train_end(c):
        # Save results and close resources at the end of training
        if c._results is not None:
            c.save_train_results(c._results)

        save_path = c.save_state(c._i, c.get_state(c._model, c._opt, c._i))
        if c.tb:
            c._writer.close()
        if hasattr(c._env, 'close'):
            c._env.close()

    def train(c):
        # Main training loop
        c.on_train_start() # Ring Env
        while c._i < c.n_steps: # episode loop
            with torch.no_grad():
                # Collect rollouts without computing gradients
                rollouts = c.rollouts() # every time collect 1 eps trajectory to update algo
            gd_stats = {}
            if len(rollouts.obs):
                t_start = time()
                # Optimize the model using the collected rollouts
                c._alg.optimize(rollouts)
                gd_stats.update(gd_time=time() - t_start)
            c.on_step_end(gd_stats)
            c._i += 1
        c.on_step_start()  # save stat
        gd_stats = {}
        with torch.no_grad(): # collect the rollout after training finishing
            rollouts = c.rollouts()
            c.on_step_end(gd_stats)
        c.on_train_end()

    def eval(c):
        # Evaluation without training
        c.setdefaults(alg='PPO')
        c._env = c.create_env()

        c._alg = (eval(c.alg) if isinstance(c.alg, str) else c.alg)(c)
        c.set_model()
        c._model.eval()
        c._results = pd.DataFrame(index=pd.Series(name='step'))
        c._writer_buffer = NamedArrays()

        # Load the specified model state for evaluation
        kwargs = {'step' if isinstance(c.e, int) else 'path': c.e}
        step = c.set_state(c._model, opt=None, **kwargs)
        c.log('Loaded model from step %s' % step)

        c._run_start_time = time()
        c._i = 1
        for _ in range(c.n_steps):
            # Perform rollouts and save results if specified
            c.rollouts()
            if c.get('result_save'):
                c._results.to_csv(c.result_save)
            if c.get('vehicle_info_save'):
                np.savez_compressed(c.vehicle_info_save, **{k: v.values.astype(type(v.iloc[0])) for k, v in c._env.vehicle_info.items()})
                if c.get('save_agent'):
                    np.savez_compressed(c.vehicle_info_save.replace('.npz', '_agent.npz'), **{k: v.values.astype(type(v.iloc[0])) for k, v in c._env.agent_info.items()})
                c._env.sumo_paths['net'].cp(c.vehicle_info_save.replace('.npz', '.net.xml'))
            c._i += 1
            c.log('')
        if hasattr(c._env, 'close'):
            c._env.close()

  

    def run(c):
        # Determine whether to train or evaluate based on configuration
        c.log(format_yaml({k: v for k, v in c.items() if not k.startswith('_')}))
        c.setdefaults(n_rollouts_per_step=1)
        if c.e is not False:
            # Evaluation mode
            c.n_workers = 1
            c.setdefaults(use_ray=False, n_rollouts_per_worker=c.n_rollouts_per_step // c.n_workers)
            c.eval()
        else:
            # Training mode
            c.setdefaults(device='cuda' if torch.cuda.is_available() else 'cpu')
            if c.get('use_ray', True) and c.n_rollouts_per_step > 1 and c.get('n_workers', np.inf) > 1:
                # Use Ray for distributed rollouts
                c.setdefaults(n_workers=c.n_rollouts_per_step, use_ray=True)
                c.n_rollouts_per_worker = c.n_rollouts_per_step // c.n_workers
                import ray
                ray.init(num_cpus=c.n_workers, include_dashboard=False, _temp_dir='/tmp/')
                RemoteMain = ray.remote(type(c))
                worker_kwargs = c.get('worker_kwargs') or [{}] * c.n_workers
                print('worker kwargs', worker_kwargs)
                assert c.n_workers % len(worker_kwargs) == 0
                c.log(f'Running {c.n_workers} with {len(worker_kwargs)} different worker kwargs')
                n_repeats = c.n_workers // len(worker_kwargs)
                # Create remote worker instances
                worker_kwargs = [{**c, 'main': False, 'device': 'cpu', **args} for args in worker_kwargs for _ in range(n_repeats)]
                c._rollout_workers = [RemoteMain.remote(**kwargs, i_worker=i) for i, kwargs in enumerate(worker_kwargs)]
                # Start the rollout workers
                ray.get([w.on_rollout_worker_start.remote() for w in c._rollout_workers])
            else:
                # Single-process rollouts
                c.setdefaults(n_workers=1, n_rollouts_per_worker=c.n_rollouts_per_step, use_ray=False)
            assert c.n_workers * c.n_rollouts_per_worker == c.n_rollouts_per_step
            c.train()
        c.log('job done!')

    def run_2(c):
        # Determine whether to train or evaluate based on configuration
        c.log(format_yaml({k: v for k, v in c.items() if not k.startswith('_')}))
        c.setdefaults(n_workers=1, n_rollouts_per_worker=c.n_rollouts_per_step, use_ray=False)
        c.on_train_start()
        c._env = c.create_env() # Create NormEnv
        c.set_model()
        done = False
        flow_eps = []
        density = c.var().n_veh/(c.var().circumference*1e-3)

        while c._i < c.n_steps: #episode loop 
            ret = c._env.reset()
            step = 0
            vehicle_flow = c._env.mean_speed * 3.6 * density 
            flow_ep = []
            flow_ep.append(vehicle_flow)
            print(f"Current vehicle flow: {vehicle_flow:.2f} at new episode")
            while step < c.horizon + c.skip_stat_steps and not done: # step loop
                # Take a step in the environment
                ret = c._env.step()
                if isinstance(ret, tuple):
                    obs, reward, done, info = ret
                    ret = dict(obs=obs, reward=reward, done=done, info=info)
                done = ret.setdefault('done', False)
                vehicle_flow = c._env.mean_speed * density  * 3.6
                flow_ep.append(vehicle_flow)
                print(f"Current vehicle flow: {vehicle_flow:.2f} at step : {step:.2f}")
                step += 1
            flow_eps.append(np.array(flow_ep))
            c._i += 1
        
        np.save((c.res+"flow_eps.np"), np.array(flow_eps))

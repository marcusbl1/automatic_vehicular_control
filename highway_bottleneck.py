from automatic_vehicular_control.exp import *
from automatic_vehicular_control.env import *
from automatic_vehicular_control.u import *

class BNeckEnv(Env):
    def def_sumo(self):
        c = self.c
        nodes = E('nodes',
            E('node', id='n_0', x=0, y=0), *(
            E('node', id=f'n_{1 + i}', x=x, y=0, type='zipper', radius=20) for i, x in enumerate(np.cumsum(c.edge_lengths))
        ))
        edges = E('edges', *(
            E('edge', **{'id': f'e_{i}', 'from': f'n_{i}', 'to': f'n_{i + 1}', 'spreadType': 'center',
                'numLanes': n_lane,
                'speed': c.max_speed
            }) for i, n_lane in enumerate(c.n_lanes)
        ))
        connections = E('connections', *(
            E('connection', **{
                'from': prev.id,
                'to': curr.id,
                'fromLane': i,
                'toLane': int(i / prev.numLanes * curr.numLanes)
            }) for prev, curr in zip(edges, edges[1:]) for i in range(prev.numLanes)
        ))

        if c.split_flow:
            n_inflow_lanes = c.n_lanes[0]
            flows = [E('flow', **FLOW(f'f_{i}', type='generic', route='route',
                departSpeed=c.depart_speed, departLane=i, vehsPerHour=c.flow_rate // n_inflow_lanes))
                for i in range(n_inflow_lanes)
            ]
        else:
            flows = [E('flow', **FLOW(f'f', type='generic', route='route',
                departSpeed=c.depart_speed, departLane='random', vehsPerHour=c.flow_rate))
            ]
        routes = E('routes',
            E('route', id='route', edges=' '.join(str(x.id) for x in edges)),
            *flows
        )
        idm_params = {**IDM, **LC2013, 'accel': c.max_accel, 'decel': c.max_decel, 'tau': c.tau, 'minGap': c.min_gap, 'maxSpeed': c.max_speed, 'delta': c.delta}
        additional = E('additional',
            E('vType', id='generic', **idm_params),
            E('vType', id='rl', **idm_params),
            E('vType', id='human', **idm_params),
        )
        sumo_args = {'collision.action': COLLISION.remove}
        return super().def_sumo(nodes, edges, connections, routes, additional, sumo_args=sumo_args)

    def reset_sumo(self):
        c = self.c
        if c.flow_rate_range:
            c.flow_rate = np.random.randint(*c.flow_rate_range)
        ret = super().reset_sumo()
        if c.vinitsky:
            self.obs_lanes = [(lane, i_start * c.piece_length)
                for edge in self.ts.routes.route.edges if not edge.id.startswith(':')
                for i_start in range(round(edge.length) // c.piece_length)
                for lane in edge.lanes
            ]
            self.control_lanes = {(lane, piece_start): c.max_speed for lane, piece_start in self.obs_lanes[:-1]}
            self.last_outflows = []
        return ret

    def step(self, action=[]):
        c = self.c
        ts = self.ts
        max_dist = 100
        max_speed = c.max_speed
        human_type = ts.types.human
        rl_type = ts.types.rl

        prev_rls = sorted(rl_type.vehicles, key=lambda x: x.id)
        if c.vinitsky:
            if len(action):
                assert len(action) == len(self.control_lanes)
                for (lane, piece_start), act in zip(self.control_lanes, action):
                    self.control_lanes[lane, piece_start] = np.clip(
                        self.control_lanes[lane, piece_start] + act * (c.max_accel if act > 0 else c.max_decel),
                        0.1, max_speed
                    )
                for veh in prev_rls:
                    lane, piece_start = veh.lane, veh.laneposition // c.piece_length * c.piece_length
                    ts.set_max_speed(veh, self.control_lanes.get((lane, piece_start), max_speed))
        else:
            for veh, act in zip(prev_rls, action):
                if c.handcraft:
                    route, edge, lane = veh.route, veh.edge, veh.lane
                    next_lane = lane.next(veh.route)
                    level = 1
                    if next_lane and next_lane.get('junction'):
                        merge_dist = lane.length - veh.laneposition
                        if merge_dist < c.handcraft:
                            other_lane = edge.lanes[lane.index - 1 if lane.index % 2 else lane.index + 1]
                            other_veh, offset = other_lane.prev_vehicle(veh.laneposition)
                            if other_veh and offset + merge_dist < 30 and other_veh.type is human_type:
                                level = 0
                    if c.act_type.startswith('accel'):
                        ts.accel(veh, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                    else:
                        ts.set_max_speed(veh, max_speed * level)
                    continue
                if not isinstance(act, (int, np.integer)):
                    act = (act - c.low) / (1 - c.low)
                if c.act_type.startswith('accel'):
                    level = act[0] if c.act_type == 'accel' else act / (c.n_actions - 1)
                    ts.accel(veh, (level * 2 - 1) * (c.max_accel if level > 0.5 else c.max_decel))
                else:
                    if c.act_type == 'continuous':
                        level = act[0]
                    elif c.act_type == 'discretize':
                        level = min(int(act[0] * c.n_actions), c.n_actions - 1) / (c.n_actions - 1)
                    elif c.act_type == 'discrete':
                        level = act / (c.n_actions - 1)
                    ts.set_max_speed(veh, max_speed * level)

                if c.lc_av == 'binary' and act[1] > 0.5:
                    ts.lane_change(veh, -1 if veh.lane_index % 2 else +1)

        super().step()

        if c.vinitsky:
            n_last_outflows = 20
            max_vehs = 4
            max_veh_outflows = np.ceil(c.sim_step * c.max_speed / 5) # 5 is the vehicle length
            obs = []
            for lane, piece_start in self.obs_lanes:
                piece_end = piece_start + c.piece_length
                lane_vehs = [veh for veh in lane.vehicles if piece_start <= veh.laneposition < piece_end]
                lane_avs = [v for v in lane_vehs if v.type is rl_type]
                lane_humans = [v for v in lane_vehs if v.type is human_type]
                obs.extend([
                    len(lane_humans) / max_vehs,
                    len(lane_avs) / max_vehs,
                    np.mean([v.speed for v in lane_humans]) / max_speed if len(lane_humans) else 0,
                    np.mean([v.speed for v in lane_avs]) / max_speed if len(lane_avs) else 0,
                ])
            if len(self.last_outflows) == n_last_outflows:
                self.last_outflows = self.last_outflows[1:]
            self.last_outflows.append(len(ts.new_arrived))
            reward = np.mean(self.last_outflows)
            obs.append(reward / max_veh_outflows)
            obs = np.array(obs, dtype=np.float32)
            assert 0 <= obs.min() and obs.max() <= 1
            return obs, reward, False, None
        route = nexti(ts.routes)
        obs, ids = [], []
        default_close = [0, max_speed]
        default_far = [max_dist, 0]
        for veh in sorted(rl_type.vehicles, key=lambda v: v.id):
            speed, edge, lane = veh.speed, veh.edge, veh.lane
            merge_dist = max_dist
            default_human = default_far

            other_info = {}
            next_lane = lane.next(veh.route)
            if next_lane and next_lane.get('junction'):
                merge_dist = lane.length - veh.laneposition
                other_lane = edge.lanes[lane.index - 1 if lane.index % 2 else lane.index + 1]
                pos = other_lane.length if c.veh_junction else veh.laneposition
                # Look for veh on other lane but do not extend past that lane
                for other_veh, offset in other_lane.prev_vehicles(pos, route=None):
                    other_info.setdefault(other_veh.type, [offset + other_lane.length - pos, other_veh.speed])
                    if other_veh.type is rl_type:
                        default_human = default_close
                        break

            obs.append([merge_dist, speed] + other_info.get(human_type, default_human) + other_info.get(rl_type, default_far))
            ids.append(veh.id)
        obs = np.array(obs).reshape(-1, c._n_obs) / ([max_dist, max_speed] * 3)
        obs = np.clip(obs, 0, 1).astype(np.float32) * (1 - c.low) + c.low
        reward = len(ts.new_arrived) - c.collision_coef * len(ts.new_collided)
        
        theoretical_outnum = c.flow_rate/3600 * c.sim_step # units: number of vehicles. sim_step: sec, flow_rate: veh/hr
        outflow_reward=np.clip(reward/theoretical_outnum, -1, 1)

        raw_ttc, raw_drac = self.calc_ttc(), self.calc_drac()
        ttc = np.log10(raw_ttc) if not np.isnan(raw_ttc) else 7  # empirically set big ttc
        ttc = np.clip(ttc/7, -1, 1)
        drac = np.log10(raw_drac) if not np.isnan(raw_drac) else 1e-4 # empirically set small drac
        drac = np.clip(drac/10, -1, 1)

        raw_pet = self.calc_pet()
        pet = np.log10(raw_pet) if not np.isnan(raw_pet) else 6 # empirically set big pet
        pet = np.clip(pet, -1, 1)

        ssm = (c.scale_ttc*ttc - c.scale_drac*drac)/2
        reward = (1-c.beta)*outflow_reward + c.beta*ssm
        
        returned = dict(obs=obs, id=ids, reward=reward, outflow_reward=outflow_reward, ttc=ttc, drac=drac, pet=pet, ssm=ssm, raw_ttc=raw_ttc, raw_drac=raw_drac, raw_pet=raw_pet) 
        return returned
        
        # return Namespace(obs=obs, id=ids, reward=reward)
    
    def calc_ttc(self):
        cur_veh_list = self.ts.vehicles
        ttcs = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            leader_speed = leader.speed
            if leader_speed < v_speed:
                ttc =  headway/(v_speed-leader_speed)
            else:
                ttc = np.nan
            ttcs.append(ttc)
        fleet_ttc = np.nanmean(np.array(ttcs))
        return fleet_ttc
    
    def calc_drac(self):
        cur_veh_list = self.ts.vehicles
        dracs = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            leader_speed = leader.speed
            drac = 0.5*np.square(v_speed-leader_speed)/headway
            dracs.append(drac)
        fleet_drac = np.nanmean(np.array(dracs))
        return fleet_drac

    def calc_pet(self):
        cur_veh_list = self.ts.vehicles
        pets = []
        for v in cur_veh_list:
            leader, headway = v.leader()
            v_speed = v.speed
            if v_speed > 1e-16:
                pet = headway/(v_speed)
                pets.append(pet)
        fleet_pet = np.nanmean(np.array(pets))
        return fleet_pet

class BNeck(Main):
    def create_env(c):
        return BNeckEnv(c)

    @property
    def observation_space(c):
        low = np.full(c._n_obs, c.low)
        return Box(low, np.ones_like(low))

    @property
    def action_space(c):
        if c.vinitsky:
            return Box(low=-1, high=1, shape=(c._n_action,), dtype=np.float32)
        assert c.act_type in ['discretize', 'discrete', 'continuous', 'accel', 'accel_discrete']
        if c.act_type in ['discretize', 'continuous', 'accel']:
            assert c.lc_av in [False, 'binary']
            return Box(low=c.low, high=1, shape=(1 + bool(c.lc_av),), dtype=np.float32)
        elif c.act_type in ['discrete', 'accel_discrete']:
            return Discrete(c.n_actions)

    def on_rollout_end(c, rollout, stats, ii=None, n_ii=None):
        if c.vinitsky:
            return super().on_rollout_end(rollout, stats, ii=n_ii)
        log = c.get_log_ii(ii, n_ii)
        step_obs_ = rollout.obs
        step_obs = step_obs_[:-1]

        ret, _ = calc_adv(rollout.reward, c.gamma)

        n_veh = np.array([len(o) for o in step_obs])
        step_ret = [[r] * nv for r, nv in zip(ret, n_veh)]
        rollout.update(obs=step_obs, ret=step_ret)

        step_id_ = rollout.pop('id')
        id = np.concatenate(step_id_[:-1])
        id_unique = np.unique(id)

        reward = np.array(rollout.pop('reward'))

        log(**stats)
        log(reward_mean=reward.mean(), reward_sum=reward.sum())
        log(
            n_veh_step_mean=n_veh.mean(), 
            n_veh_step_sum=n_veh.sum(), 
            n_veh_unique=len(id_unique),
            
            reward_mean=np.mean(reward),
            reward_std=np.std(reward),        
            outflow_reward_mean=np.mean(rollout.outflow_reward) if rollout.outflow_reward else None,
            outflow_reward_std=np.std(rollout.outflow_reward) if rollout.outflow_reward else None,
            ssm_mean=np.mean(rollout.ssm),
            ssm_std=np.std(rollout.ssm),
            drac_mean=np.mean(rollout.drac) if rollout.drac else None,
            drac_std=np.std(rollout.drac) if rollout.drac else None,
            pet_mean=np.mean(rollout.pet) if rollout.pet else None,
            pet_std=np.std(rollout.pet) if rollout.pet else None,
            raw_drac_mean=np.mean(rollout.raw_drac) if rollout.raw_drac else None,
            raw_drac_std=np.std(rollout.raw_drac) if rollout.raw_drac else None,
            raw_pet_mean=np.mean(rollout.raw_pet) if rollout.raw_pet else None,
            raw_pet_std=np.std(rollout.raw_pet) if rollout.raw_pet else None,

            ttc_mean=np.mean(rollout.ttc) if rollout.ttc else None,
            ttc_std=np.std(rollout.ttc) if rollout.ttc else None,
            raw_ttc_mean=np.mean(rollout.raw_ttc) if rollout.raw_ttc else None,
            raw_ttc_std=np.std(rollout.raw_ttc) if rollout.raw_ttc else None,
            # nom_action = np.mean(rollout.nom_action),
            # res_action = np.mean(rollout.res_action),
            )
        return rollout

if __name__ == '__main__':
    c = BNeck.from_args(globals(), locals()).setdefaults(
        warmup_steps=1000,
        horizon=2000,
        n_steps=100,
        step_save=5,

        av_frac=0.2,
        sim_step=0.5,
        depart_speed=0,
        edge_lengths=[100, 100, 50],
        n_lanes=[4, 2, 1],
        max_speed=30,

        lc_av=False,
        flow_rate=2300,
        flow_rate_range=None,
        split_flow=True,
        generic_type='rand',
        speed_mode=SPEED_MODE.all_checks,
        lc_mode=LC_MODE.off,
        collision_coef=5, # If there's a collision, it always involves an even number of vehicles

        veh_junction=False,
        act_type='accel_discrete',
        max_accel=2.6,
        max_decel=4.5,
        tau=1.0,
        min_gap=2.5,
        delta=4,
        n_actions=3,
        low=-1,
        handcraft=False,
        vinitsky=False,

        render=False,

        alg=PG,
        lr=1e-3,

        gamma=0.99,
        adv_norm=False,
        batch_concat=True,

        beta=0,
        scale_ttc=1,
        scale_drac=1,
        seed_np=False,
        seed_torch = False,
        residual_transfer=False, # this flag deals with which network to modify (nominal if False, residual if True). instantiates both.
        mrtl=False, # this flag deals with adding beta to observation vector

    )
    
    if c.seed_torch:
        # Set seed for PyTorch CPU operations
        torch.manual_seed(c.seed_torch)
        # Set seed for PyTorch CUDA operations (if available)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(c.seed_torch)
    if c.seed_np:
        np.random.seed(c.seed_np)
        
    if c.vinitsky:
        c.piece_length = 20
        bneck_lengths = [20] + [40] * (len(c.edge_lengths) - 2) + [20]
        n_actions = [(L - Lb) // c.piece_length * n for L, Lb, n in zip(c.edge_lengths, bneck_lengths, c.n_lanes)]
        c._n_action = sum(n_actions[:-1])
        c._n_obs = sum(n_actions) * 4 + 1
        c.batch_concat = False
    else:
        c._n_obs = 2 + 2 + 2
        
    if c.mrtl:
        c._n_obs += 1 # modified for mrtl related
        
    c.redef_sumo = bool(c.flow_rate_range)
    c.run()
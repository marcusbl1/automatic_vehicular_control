from lxml.etree import Element, SubElement, tostring, XMLParser
from xml.etree import ElementTree
import subprocess
import sumolib
import traci
import traci.constants as T  # https://sumo.dlr.de/pydoc/traci.constants.html
from traci.exceptions import FatalTraCIError, TraCIException
import bisect
import warnings
import gym

from automatic_vehicular_control.u import *
from automatic_vehicular_control.ut import *

def val_to_str(x):
    # Convert a value to string; if it's a boolean, convert to lowercase string
    return str(x).lower() if isinstance(x, bool) else str(x)

def dict_kv_to_str(x):
    # Convert all keys and values in a dictionary to strings
    return dict(map(val_to_str, kv) for kv in x.items())

def str_to_val(x):
    # Try to convert a string to an integer or float; if unsuccessful, return the original string
    for f in int, float:
        try:
            return f(x)
        except (ValueError, TypeError):
            pass
    return x

def values_str_to_val(x):
    # Apply str_to_val to all values in a dictionary
    for k, v in x.items():
        x[k] = str_to_val(v)
    return x

class E(list):
    """
    Builder for lxml.etree.Element
    """
    # Define the XML Schema Instance namespace
    xsi = Path('F') / 'resources' / 'xml' / 'XMLSchema-instance'  # http://www.w3.org/2001/XMLSchema-instance
    root_args = dict(nsmap=dict(xsi=xsi))
    def __init__(self, _name, *args, **kwargs):
        # Ensure all positional arguments are instances of E
        assert all(isinstance(a, E) for a in args)
        super().__init__(args)
        self._dict = kwargs
        self._name = _name

    def keys(self):
        # Return the keys of the attribute dictionary
        return self._dict.keys()

    def values(self):
        # Return the values of the attribute dictionary
        return self._dict.values()

    def items(self):
        # Return the items of the attribute dictionary
        return self._dict.items()

    def __getitem__(self, k):
        # Access child elements by index or attributes by key
        if isinstance(k, (int, slice)):
            return super().__getitem__(k)
        return self._dict.__getitem__(k)

    def __setitem__(self, k, v):
        # Set child elements by index or attributes by key
        if isinstance(k, (int, slice)):
            return super().__setitem__(k, v)
        return self._dict.__setitem__(k, v)

    def __getattr__(self, k):
        # Allow attribute-style access to the attribute dictionary
        if k in ['__array_struct__']:
            raise AttributeError
        if k in ['_dict', '_name']:
            return self.__dict__[k]
        else:
            return self[k]

    def __setattr__(self, k, v):
        # Allow setting attributes directly
        if k in ['_dict', '_name']:
            self.__dict__[k] = v
        else:
            self[k] = v

    def __repr__(self):
        # Return the string representation of the XML element
        return self.to_string().decode()

    def to_element(self, root=True):
        # Convert to an lxml Element
        e = Element(self._name, attrib={k: val_to_str(v) for k, v in self.items()}, **(E.root_args if root else {}))
        e.extend([x.to_element(root=False) for x in self])
        return e

    def to_string(self):
        # Convert the Element to a pretty-printed XML string
        return tostring(self.to_element(), pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def to_path(self, p):
        # Save the XML string to a file
        p.save_bytes(self.to_string())

    def children(self, tag):
        # Return child elements with a specific tag
        return [x for x in self if x._name == tag]

    @classmethod
    def from_element(cls, e):
        # Create an E instance from an lxml Element
        return E(e.tag, *(cls.from_element(x) for x in e), **e.attrib)

    @classmethod
    def from_path(cls, p):
        # Create an E instance from an XML file
        return cls.from_element(ElementTree.parse(p, parser=XMLParser(recover=True)).getroot())

    @classmethod
    def from_string(cls, s):
        # Create an E instance from an XML string
        return cls.from_element(ElementTree.fromstring(s))

# Create namespaces for variables and traffic light constants
V = Namespace(**{k[4:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('VAR_')})
TL = Namespace(**{k[3:].lower(): k for k, v in inspect.getmembers(T, lambda x: not callable(x)) if k.startswith('TL_')})

class SubscribeDef:
    """
    SUMO subscription manager
    """
    def __init__(self, tc_module, subs):
        # Initialize with the TraCI module and the list of subscriptions
        self.tc_mod = tc_module
        self.names = [k.split('_', 1)[1].lower() for k in subs]
        self.constants = [getattr(T, k) for k in subs]

    def subscribe(self, *id):
        # Subscribe to variables for given IDs
        self.tc_mod.subscribe(*id, self.constants)
        return self

    def get(self, *id):
        # Retrieve subscription results
        res = self.tc_mod.getSubscriptionResults(*id)
        return Namespace(((n, res[v]) for n, v in zip(self.names, self.constants)))

# Define speed modes and lane change modes
SPEED_MODE = Namespace(
    aggressive=0,
    obey_safe_speed=1,
    no_collide=7,
    right_of_way=25,
    all_checks=31
)

LC_MODE = Namespace(off=0, no_lat_collide=512, strategic=1621)

# Traffic light defaults
PROGRAM_ID = 1
MAX_GAP = 3.0
DETECTOR_GAP = 0.6
SHOW_DETECTORS = True

# Car following models
IDM = dict(
    accel=2.6,
    decel=4.5,
    tau=1.0,  # Past 1 at sim_step=0.1 you no longer see waves
    minGap=2.5,
    maxSpeed=30,
    speedFactor=1.0,
    speedDev=0.1,
    impatience=0.5,
    delta=4,
    carFollowModel='IDM',
    sigma=0.2,
)

Krauss = dict(
    accel=2.6,
    decel=4.5,
    tau=1.0,
    minGap=2.5,
    sigma=0.5,
    maxSpeed=30,
    speedFactor=1.0,
    speedDev=0.1,
    impatience=0.5,
    carFollowModel='Krauss',
)

# Lane change models
LC2013 = dict(
    laneChangeModel='LC2013',
    lcStrategic=1.0,
    lcCooperative=1.0,
    lcSpeedGain=1.0,
    lcKeepRight=1.0,
)

SL2015 = dict(
    laneChangeModel='SL2015',
    lcStrategic=1.0,
    lcCooperative=1.0,
    lcSpeedGain=1.0,
    lcKeepRight=1.0,
    lcLookAheadLeft=2.0,
    lcSpeedGainRight=1.0,
    lcSublane=1.0,
    lcPushy=0,
    lcPushyGap=0.6,
    lcAssertive=1,
    lcImpatience=0,
    lcTimeToImpatience=float('inf'),
    lcAccelLat=1.0,
)

# Builder for an inflow
def FLOW(id, type, route, departSpeed, departLane='random', vehsPerHour=None, probability=None, period=None, number=None):
    flow = Namespace(
        id=id,
        type=type,
        route=route,
        departSpeed=departSpeed,
        departLane=departLane,
        begin=1
    )
    flow.update(dict(number=number) if number else dict(end=86400))
    if vehsPerHour:
        flow.vehsPerHour = vehsPerHour
    elif probability:
        flow.probability = probability
    elif period:
        flow.period = period
    return flow

# Vehicle colors
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
RED = (255, 0, 0)

COLLISION = Namespace(teleport='teleport', warn='warn', none='none', remove='remove')

class NetBuilder:
    """
    Builder for the traffic network, which includes nodes, edges, connections, and additional elements.
    Output can be saved into XML and serve as input to SUMO.
    """
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.connections = {}
        self.additional = []

    def add_nodes(self, node_infos):
        # Add nodes to the network
        nodes = [E('node', **n.setdefaults(id=f'n_{n.x}.{n.y}')) for n in node_infos]
        self.nodes.update((n.id, n) for n in nodes)
        ret = np.empty(len(nodes), dtype=object)
        ret[:] = nodes
        return ret

    def chain(self, nodes, lane_maps=None, edge_attrs={}, route_id=None):
        """
        Add a chain of nodes while building the edges, connections, and routes.
        """
        edge_attrs = [edge_attrs] * (len(nodes) - 1) if isinstance(edge_attrs, dict) else edge_attrs
        lane_maps = lane_maps or [{0: 0} for _ in range(len(nodes) - 2)]
        num_lanes = ([len(l) for l in lane_maps] + [len(set(lane_maps[-1].values()))])
        edges = [E('edge', **{
            'id': f'e_{n1.id}_{n2.id}',
            'from': n1.id, 'to': n2.id,
            'length': np.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2),
            'numLanes': nl,
            **e
        }) for n1, n2, nl, e in zip(nodes, nodes[1:], num_lanes, edge_attrs)]
        connections = flatten([[E('connection', **{
            'from': e1.id, 'to': e2.id,
            'fromLane': from_, 'toLane': to
        }) for from_, to in lmap.items()] for e1, e2, lmap in zip(edges, edges[1:], lane_maps)])
        route = E('route', id=route_id or f'r_{len(self.additional)}', edges=' '.join(e.id for e in edges))
        self.edges.update((e.id, e) for e in edges)
        self.connections.update(((con['from'], con.to, con.fromLane, con.toLane), con) for con in connections)
        self.additional.append(route)
        return edges, connections, route

    def intersect(self, center, neighbors, movements, separate_right=False):
        """
        Build an intersection with specified movements and configurations.
        """
        turn_nodes, turn_lanes = movements_to_turns(neighbors, movements)
        opp_turn_lanes = np.roll(turn_lanes, 2, axis=0)
        separate_right = [separate_right] * len(neighbors) if isinstance(separate_right, bool) else separate_right
        edges = []
        for x, num_x_to, num_opp_to, num_to_x, sep in zip(neighbors, turn_lanes, opp_turn_lanes, movements.T, separate_right):
            pairs = []
            p1, p2 = np.array([x.x, x.y]), np.array([center.x, center.y])
            d = p2 - p1
            dist = np.linalg.norm(d)
            d1, d2 = np.array([-d[1], d[0]]) / dist * 3.2 * min(num_x_to[-1], num_opp_to[-1]) * 0.5  # 3.2 is the lane width
            if num_x_to.any():
                pairs.append((x, center, num_x_to.sum() - min(num_x_to[0], not sep)))
            if num_to_x.any():
                pairs.append((center, x, num_to_x.max()))
            for x1, x2, n_lanes in pairs:
                edges.append(E('edge', **{
                    'id': f'e_{x1.id}_{x2.id}',
                    'from': x1.id, 'to': x2.id,
                    'length': dist,
                    'shape': f'{x1.x + d1},{x1.y + d2} {x2.x + d1},{x2.y + d2}',
                    'numLanes': n_lanes,
                }))
        self.edges.update((e.id, e) for e in edges)

        connections = []
        for x1, x1_to, num_x1_to, sep in zip(neighbors, turn_nodes, turn_lanes, separate_right):
            offsets1 = np.cumsum(num_x1_to) - num_x1_to
            offsets1[1:] = offsets1[1:] - min(num_x1_to[0], not sep)
            e1 = f'e_{x1.id}_{center.id}'
            for i, (x2, num_x1_to_x2, o1) in enumerate(zip(x1_to, num_x1_to, offsets1)):
                e2 = f'e_{center.id}_{x2.id}'
                o2 = 0 if i == 0 else self.edges[e2].numLanes - num_x1_to_x2  # 0 for right turns
                connections.extend(E('connection', **{'from': e1, 'to': e2, 'fromLane': o1 + k, 'toLane': o2 + k}) for k in range(num_x1_to_x2))
        self.connections.update(((con['from'], con.to, con.fromLane, con.toLane), con) for con in connections)

        routes = []
        for x1, x1_to in zip(neighbors, turn_nodes):
            e1, r1 = f'e_{x1.id}_{center.id}', f'r_{x1.id}_{center.id}'
            routes.append(E('route', id=r1, edges=e1))
            routes1 = [E('route', id=f'r_{x1.id}_{x2.id}', edges=f'{e1} e_{center.id}_{x2.id}') for x2 in x1_to]
            routes.extend(routes1)

            routes.append(E('rerouter', E('interval',
                *(E('routeProbReroute', id=r.id, probability=1) for r in routes1), begin=0, end=1e9)
            , id='reroute', edges=e1))
        self.additional.extend(routes)
        return edges, connections, routes

    def build(self):
        # Return the built network components
        return E('nodes', *self.nodes.values()), E('edges', *self.edges.values()), E('connections', *self.connections.values()), self.additional

def movements_to_turns(nodes, movements):
    """
    Convert movements at an intersection to turning movements.
    """
    return (
        [(nodes[i + 1:] + nodes[:i])[::-1] for i, _ in enumerate(nodes)],
        np.array([np.roll(ms, -i)[:0:-1] for i, ms in enumerate(movements)])
    )

def build_closed_route(edges, n_veh=0, av=0, space='random_free', type_fn=None, depart_speed=0, offset=0, init_length=None):
    """
    Build a closed route with initial vehicles.
    """
    assert isinstance(space, (float, int)) or space in ('equal', 'random_free', 'free', 'random', 'base', 'last')
    order = lambda i: edges[i:] + edges[:i + 1]
    routes = [E('route', id=f'route_{e.id}', edges=' '.join(e_.id for e_ in order(i))) for i, e in enumerate(edges)]
    rerouter = E('rerouter', E('interval', E('routeProbReroute', id=routes[0].id), begin=0, end=1e9), id='reroute', edges=edges[0].id)
    vehicles = []
    if n_veh > 0:
        lane_lengths, lane_routes, lane_idxs = map(np.array, zip(*[(e.length, r.id, i) for e, r in zip(edges, routes) for i in range(e.numLanes)]))
        lane_ends = np.cumsum(lane_lengths)
        lane_starts = lane_ends - lane_lengths
        total_length = lane_ends[-1]
        init_length = init_length or total_length

        positions = (offset + np.linspace(0, init_length, n_veh, endpoint=False)) % total_length
        veh_lane = (positions.reshape(-1, 1) < lane_ends.reshape(1, -1)).argmax(axis=1)

        if space == 'equal':
            space = total_length / n_veh
        if isinstance(space, (float, int)):
            veh_lane_pos = positions - lane_starts[veh_lane]
        else:
            veh_lane_pos = [space] * n_veh
        veh_routes = lane_routes[veh_lane]
        veh_lane_idxs = lane_idxs[veh_lane]
        type_fn = type_fn or (lambda i: 'rl' if i < av else 'human')
        vehicles = [E('vehicle', id=f'{i}', type=type_fn(i), route=r, depart='0', departPos=p, departLane=l, departSpeed=depart_speed) for i, (r, p, l) in enumerate(zip(veh_routes, veh_lane_pos, veh_lane_idxs))]
        # Modify vehicle objects to add SSM device parameter if needed
    return [*routes, rerouter, *vehicles]

class SumoDef:
    """
    Given the network definitions in Python:
    1. Save them into XML files for SUMO.
    2. Run SUMO's netconvert command on the input files.
    3. Start the SUMO simulation as a subprocess.
    """
    no_ns_attr = '{%s}noNamespaceSchemaLocation' % E.xsi
    xsd = Path.env('F') / 'resources' / 'xml' / '%s_file.xsd'  # XSD schema path
    config_xsd = Path.env('F') / 'resources' / 'xml' / 'sumoConfiguration.xsd'  # Configuration XSD schema
    # See https://sumo.dlr.de/docs/NETCONVERT.html
    netconvert_args = dict(nodes='n', edges='e', connections='x', types='t')
    config_args = dict(
        net='net-file', routes='route-files',
        additional='additional-files', gui='gui-settings-file'
    )
    file_args = set(netconvert_args.keys()) | set(config_args.keys())

    def __init__(self, c):
        self.c = c
        self.dir = c.res.rel() / 'sumo'
        if 'i_worker' in c:
            self.dir /= c.i_worker
        self.dir.mk()  # Use relative path here to shorten SUMO arguments
        self.sumo_cmd = None

    def save(self, *args, **kwargs):
        # Save the XML elements to files
        for e in args:
            e[SumoDef.no_ns_attr] = SumoDef.xsd % e._name
            kwargs[e._name] = path = self.dir / e._name[:3] + '.xml'
            e.to_path(path)
        return Namespace(**kwargs)

    def generate_net(self, **kwargs):
        # Generate the network using netconvert
        net_args = Namespace(**kwargs.get('net_args', {})).setdefaults(**{
            'no-turnarounds': True
        })

        # https://sumo.dlr.de/docs/NETCONVERT.html
        dyld_env_var = os.environ.get('DYLD_LIBRARY_PATH')
        net_path = self.dir / 'net.xml'
        args = [*lif(dyld_env_var, f'DYLD_LIBRARY_PATH={dyld_env_var}'), 'netconvert', '-o', net_path]
        for name, arg in SumoDef.netconvert_args.items():
            path = kwargs.pop(name, None)
            if path:
                args.append('-%s %s' % (arg, path))
        args.extend('--%s %s' % (k, val_to_str(v)) for k, v in net_args.items())

        cmd = ' '.join(args)
        self.c.log(cmd)
        out, err = shell(cmd, stdout=None)
        if err:
            self.c.log(err)

        return net_path

    def generate_sumo(self, **kwargs):
        c = self.c

        gui_path = self.dir / 'gui.xml'
        # Create GUI settings XML
        E('viewsettings',
            E('scheme', name='real world'),
            E('background',
                backgroundColor='150,150,150',
                showGrid='0',
                gridXSize='100.00',
                gridYSize='100.00'
            )).to_path(gui_path)
        kwargs['gui'] = gui_path

        # https://sumo.dlr.de/docs/SUMO.html
        sumo_args = Namespace(
            **{arg: kwargs[k] for k, arg in SumoDef.config_args.items() if k in kwargs},
            **kwargs.get('sumo_args', {})).setdefaults(**{
            'begin': 0,
            'step-length': c.sim_step,
            'no-step-log': True,
            'time-to-teleport': -1,
            'no-warnings': c.get('no_warnings', True),
            'collision.action': COLLISION.remove,
            'collision.check-junctions': True,
            'max-depart-delay': c.get('max_depart_delay', 0.5),
            'random': True,
            'start': c.get('start', True)
        })
        cmd = ['sumo-gui' if c.render else 'sumo']
        for k, v in sumo_args.items():
            cmd.extend(['--%s' % k, val_to_str(v)] if v is not None else [])
        c.log(' '.join(cmd))
        return cmd

    def start_sumo(self, tc, tries=3):
        # Start the SUMO simulation
        for _ in range(tries):
            try:
                if tc and 'TRACI_NO_LOAD' not in os.environ:
                    tc.load(self.sumo_cmd[1:])
                else:
                    if tc:
                        tc.close()
                    else:
                        self.port = sumolib.miscutils.getFreeSocketPort()
                    # Taken from traci.start but add the DEVNULL here
                    p = subprocess.Popen(self.sumo_cmd + ['--remote-port', f'{self.port}'], **dif(self.c.get('sumo_no_errors', True), stderr=subprocess.DEVNULL))
                    tc = traci.connect(self.port, 10, 'localhost', p)
                return tc
            except traci.exceptions.FatalTraCIError:
                # Handle errors by restarting SUMO
                if tc:
                    tc.close()
                self.c.log('Restarting SUMO...')
                tc = None

class Namespace(Namespace):
    """
    A wrapper around dictionary with nicer formatting for the Entity subclasses.
    """
    @classmethod
    def format(cls, x):
        # Format the Namespace for pretty printing
        def helper(x):
            if isinstance(x, Entity):
                return f"{type(x).__name__}('{x.id}')"
            elif isinstance(x, (list, tuple, set, np.ndarray)):
                return [helper(y) for y in x]
            elif isinstance(x, np.generic):
                return x.item()
            elif isinstance(x, dict):
                return {helper(k): helper(v) for k, v in x.items()}
            return x
        return format_yaml(helper(x))

    def __repr__(self):
        # Return the formatted representation
        return Namespace.format(self)

class Container(Namespace):
    def __iter__(self):
        # Allow iteration over the values
        return iter(self.values())

class Entity(Namespace):
    def __hash__(self):
        # Hash based on the entity ID
        return hash(self.id)

    def __str__(self):
        # String representation is the ID
        return self.id

    def __repr__(self):
        # Custom representation including the type and ID
        inner_content = Namespace.format(dict(self)).replace('\n', '\n  ').rstrip(' ')
        return f"{type(self).__name__}('{self.id}',\n  {inner_content})\n\n"

class Vehicle(Entity):
    def leader(self, use_edge=False, use_route=True, max_dist=np.inf):
        # Get the closest vehicle ahead
        try:
            return next(self.leaders(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def leaders(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        # Generator for vehicles ahead
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.next_vehicles_helper(i + 1, route, max_dist + self.laneposition, max_count):
            yield veh, dist - self.laneposition

    def follower(self, use_edge=False, use_route=True, max_dist=np.inf):
        # Get the closest vehicle behind
        try:
            return next(self.followers(use_edge, use_route, max_dist, 1))
        except StopIteration:
            return None, 0

    def followers(self, use_edge=False, use_route=True, max_dist=np.inf, max_count=np.inf):
        # Generator for vehicles behind
        ent, i = (self.edge, self.edge_i) if use_edge else (self.lane, self.lane_i)
        route = self.route if use_route else None
        for veh, dist in ent.prev_vehicles_helper(i - 1, route, max_dist - self.laneposition, max_count):
            yield veh, dist + self.laneposition

class Type(Entity):
    pass

class Flow(Entity):
    pass

class Junction(Entity):
    pass

class Edge(Entity):
    def next(self, route):
        # Get the next edge in the route
        return route.next.get(self)

    def prev(self, route):
        # Get the previous edge in the route
        return route.prev.get(self)

    def find(self, position, route):
        """
        Find the entity where the position belongs and return the relative position to that entity.
        """
        ent = self
        offset = 0
        while position < 0 and ent and route:
            ent = ent.prev(route)
            position += ent.length
            offset -= ent.length
        while position >= ent.length and ent and route:
            position -= ent.length
            offset += ent.length
            ent = ent.next(route)
        assert ent is None or 0 <= position < ent.length
        return ent, position, offset

    def next_vehicle(self, position, route=None, max_dist=np.inf, filter=lambda veh: True):
        # Get the next vehicle ahead
        try:
            return next(self.next_vehicles(position, route, max_dist, 1, filter))
        except StopIteration:
            return None, 0

    def prev_vehicle(self, position, route=None, max_dist=np.inf, filter=lambda veh: True):
        # Get the next vehicle behind
        try:
            return next(self.prev_vehicles(position, route, max_dist, 1, filter))
        except StopIteration:
            return None, 0

    def next_vehicles(self, position, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        Generator for vehicles ahead starting from a position.
        """
        assert max_dist == np.inf or position == 0, 'Haven\'t implemented general case yet'
        ent, position, offset = self.find(position, route) if route else (self, position, 0)
        i = bisect.bisect_left(ent.positions, position)
        for veh, dist in ent.next_vehicles_helper(i, route, max_dist, max_count, filter):
            yield veh, dist - position

    def prev_vehicles(self, position, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        Generator for vehicles behind starting from a position.
        """
        assert max_dist == np.inf or position == 0, 'Haven\'t implemented general case yet'
        ent, position, offset = self.find(position, route) if route else (self, position, 0)
        i = bisect.bisect_right(ent.positions, position)
        for veh, dist in ent.prev_vehicles_helper(i - 1, route, max_dist, max_count, filter):
            yield veh, dist + position

    def next_vehicles_helper(self, i, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        Helper method to iterate over vehicles ahead.
        """
        while i < len(self.vehicles):
            veh = self.vehicles[i]
            veh_dist = veh.laneposition
            if veh_dist > max_dist:
                return
            if filter(veh):
                yield veh, veh_dist
            max_count -= 1
            if max_count <= 0:
                return
            i += 1
        dist = self.length
        if route and dist <= max_dist:
            ent = self.next(route)
            while ent is not None:
                for veh in ent.vehicles:
                    veh_dist = dist + veh.laneposition
                    if veh_dist > max_dist:
                        return
                    if filter(veh):
                        yield veh, veh_dist
                    else:
                        continue
                    max_count -= 1
                    if max_count <= 0:
                        return
                ent = ent.next(route)

    def prev_vehicles_helper(self, i, route=None, max_dist=np.inf, max_count=np.inf, filter=lambda veh: True):
        """
        Helper method to iterate over vehicles behind.
        """
        while i >= 0:
            veh = self.vehicles[i]
            veh_dist = -veh.laneposition
            if veh_dist > max_dist:
                return
            if filter(veh):
                yield veh, veh_dist
            max_count -= 1
            if max_count <= 0:
                return
            i -= 1
        dist = 0
        if route and dist <= max_dist:
            ent = self.prev(route)
            while ent is not None:
                dist += ent.length
                for veh in reversed(ent.vehicles):
                    veh_dist = dist - veh.laneposition
                    if veh_dist > max_dist:
                        return
                    if filter(veh):
                        yield veh, veh_dist
                    else:
                        continue
                    max_count -= 1
                    if max_count <= 0:
                        return
                ent = ent.prev(route)

class Lane(Edge):
    pass

class Route(Entity):
    def next_edges(self, edge):
        # Get edges after the current edge in the route
        return self.edges[self.edges.index(edge) + 1:]

    def prev_edges(self, edge):
        # Get edges before the current edge in the route
        return self.edges[self.edges.index(edge) - 1::-1]

    def prev_vehicle(self, route_position):
        # Get the previous vehicle based on route position
        i = bisect.bisect(self.positions, route_position)
        return self.vehicles[i - 1] if i > 0 else None

    def next_vehicle(self, route_position):
        # Get the next vehicle based on route position
        i = bisect.bisect(self.positions, route_position)
        return self.vehicles[i + 1] if i + 1 < len(self.vehicles) else None

# The rest of the code continues with the definition of TrafficState, Env, NormEnv classes, and their methods.


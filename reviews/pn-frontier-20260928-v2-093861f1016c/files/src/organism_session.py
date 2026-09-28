"""One continuing MaleCNS rate brain, own synapses, physical body and world.

The whole graph participates in recurrent dynamics. The named ORN/DAN inputs
and DN-to-FlyGym bridge are declared experimental transductions, not inferred
physiology. Walking still uses FlyGym's CPGs, trajectory templates, servos and
adhesion. A checkpoint includes those aids rather than concealing a reset.
"""
from pathlib import Path
import json
import math
import shutil
import tempfile
import numpy as np
from session_io import sha256, write_state, read_state


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('organism_session.py', 'walking_body_checkpoint.py', 'embodied_navigation.py',
           'passive_body.py', 'session_io.py', 'anatomical_rate_brain.py', 'anatomical_plasticity.py')


class OdorPatchWorld:
    """Own flat-world field; positions mm, time ns, concentration dimensionless.

    Starts on a patch. Patch contact is an artificial reinforcement sensor,
    not ingestion. There is no target heading or distance-improvement reward.
    Geometry comes from the body only at the sensory transduction boundary.
    """
    def __init__(self):
        self.time_ns = 0
        self.source_mm = np.array([0., 0.])
        self.sigma_mm = 15.
        self.contact_radius_mm = 3.
        self.antenna_forward_mm = 1.
        self.antenna_half_separation_mm = .3
        self.events = []
        self.next_event = 0

    def advance_to(self, time_ns):
        if time_ns < self.time_ns:
            raise ValueError('World cannot run backwards')
        self.time_ns = time_ns
        while self.next_event < len(self.events) and self.events[self.next_event]['time_ns'] <= time_ns:
            self.source_mm = np.asarray(self.events[self.next_event]['source_mm'], dtype=float)
            self.next_event += 1

    def sense(self, observation):
        p = observation['position_mm'][:2]
        yaw = observation['yaw']
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        left = np.array([-forward[1], forward[0]])
        center = p + self.antenna_forward_mm * forward
        points = np.array([center + self.antenna_half_separation_mm * left,
                           center - self.antenna_half_separation_mm * left])
        odor = np.exp(-np.sum((points-self.source_mm)**2, axis=1)/(2*self.sigma_mm**2))
        contact = float(np.linalg.norm(p-self.source_mm) <= self.contact_radius_mm)
        return np.array([odor[0], odor[1], contact], dtype=np.float64)


class OrganismSession:
    CONTROL_NS = 5_000_000

    @classmethod
    def create(cls, seed=0, learning=False, preparation=None, ports_directory=None):
        from anatomical_preparation import load_preparation
        from anatomical_rate_brain import AnatomicalRateBrain
        from anatomical_plasticity import CandidateGammaPlasticity
        from walking_body_checkpoint import ContinuableWalkingBody
        preparation = Path(preparation or ROOT/'data/anatomical_brain/v0.1/all')
        ports_directory = Path(ports_directory or ROOT/'data/anatomical_brain/v0.1/ports')
        anatomy = load_preparation(preparation)
        self = cls()
        self.brain = AnatomicalRateBrain(anatomy, seed=seed)
        port_record = json.loads((ports_directory/'ports.json').read_text())
        self.ports = port_record['ports']
        self.port_identity = sha256(ports_directory/'ports.json')
        self.population_indices = {str(label): np.flatnonzero(anatomy.nodes['superclass'].to_numpy() == label)
                                   for label in anatomy.nodes['superclass'].dropna().unique()}
        self._index_ports()
        self.plasticity = CandidateGammaPlasticity(self.brain, ports_directory, enabled=learning)
        self.body = ContinuableWalkingBody(seed=seed, scaffold='no_contact_rules')
        self.world = OdorPatchWorld()
        self.time_ns = 0
        self.ticks = 0
        self.pending_sensors = self.world.sense(self.body.observe())
        self.last_action = np.zeros(2)
        self.motor_disconnected = False
        self.sensory_replay = None
        self.history = []
        self.used_sensors = []
        self.failed = False
        self.source_identity = {name: sha256(ROOT/'src'/name) for name in SOURCES}
        self.config = dict(control_ns=self.CONTROL_NS, odor_drive=80., reward_drive=80.,
                           arousal_drive=80., readout_half_rate=50., turn_fraction=.33,
                           max_motor_amplitude=1.2,
                           motor_delay_ns=self.CONTROL_NS,
                           candidate='ALL_MALECNS_RATE_OLFACTION_GAMMA4_DN_BODY_WORLD_v1')
        return self

    def _index_ports(self):
        self.port_indices = {}
        for name, ids in self.ports.items():
            indices = np.searchsorted(self.brain.node_ids, ids)
            if (np.any(indices >= self.brain.n_neurons) or
                    not np.array_equal(self.brain.node_ids[indices], ids)):
                raise ValueError(f'Port {name} has an unknown anatomical ID')
            self.port_indices[name] = indices

    def _validate_clock(self):
        dt_ns = round(self.body.dt*1e9)
        if (self.config['control_ns'] != self.CONTROL_NS or dt_ns <= 0 or
                self.CONTROL_NS % dt_ns or self.config['motor_delay_ns'] != self.CONTROL_NS):
            raise ValueError('Unsupported clock ratio or motor delay')
        if (self.time_ns != self.brain.time_ns or self.time_ns != self.world.time_ns
                or self.time_ns != self.plasticity.time_ns or self.ticks != self.plasticity.update_count
                or self.time_ns != self.ticks*self.CONTROL_NS
                or self.body.steps*round(self.body.dt*1e9) != self.time_ns):
            raise ValueError('Brain, body, world and pending-sensor clocks disagree')

    def step(self):
        if self.failed:
            raise ValueError('Failed sessions cannot silently resume')
        self._validate_clock()
        sensory = self.pending_sensors.copy()
        if self.sensory_replay is not None:
            if self.ticks >= len(self.sensory_replay):
                raise ValueError('Replay exhausted; live sensing is not substituted')
            sensory = self.sensory_replay[self.ticks].copy()
        if sensory.shape != (3,) or not np.isfinite(sensory).all() or np.any((sensory < 0) | (sensory > 1)):
            raise ValueError('Invalid pending environmental input')
        drive = np.zeros(self.brain.n_neurons, dtype=np.float32)
        for i, side in enumerate(('L', 'R')):
            drive[self.port_indices[f'ORN_DM1_{side}']] = self.config['odor_drive']*sensory[i]
            drive[self.port_indices[f'PAM08_{side}']] = self.config['reward_drive']*sensory[2]
            drive[self.port_indices[f'DNg100_{side}']] = self.config['arousal_drive']
        dt = self.CONTROL_NS*1e-9
        # Both solvers advance from t to t+h using data already available at t.
        # New DN activity chooses the command held in the NEXT body interval.
        applied_action = self.last_action.copy()
        if self.motor_disconnected:
            applied_action[:] = 0.
        try:
            self.brain.step(dt, drive)
            self.plasticity.step(dt)
            left = float(self.brain.rates[self.port_indices['DNa02_L']].mean())
            right = float(self.brain.rates[self.port_indices['DNa02_R']].mean())
            walk = float(np.mean([self.brain.rates[self.port_indices[f'DNg100_{s}']].mean() for s in ('L', 'R')]))
            amplitude = self.config['max_motor_amplitude']*walk/(self.config['readout_half_rate']+walk)
            turn = (left-right)/(self.config['readout_half_rate']+left+right)
            action = amplitude*np.array([1-self.config['turn_fraction']*turn,
                                          1+self.config['turn_fraction']*turn])
            action = np.clip(action, 0, self.config['max_motor_amplitude'])
            if self.motor_disconnected:
                action[:] = 0.
            self.body.advance(applied_action, self.CONTROL_NS//round(self.body.dt*1e9))
            self.time_ns += self.CONTROL_NS
            self.ticks += 1
            self.world.advance_to(self.time_ns)
            observation = self.body.observe()
            self.pending_sensors = self.world.sense(observation)
            self.last_action = action
            self.used_sensors.append(sensory)
            row = dict(time_s=self.time_ns*1e-9, position_mm=observation['position_mm'].tolist(),
                       qpos=observation['qpos'].tolist(),
                       yaw_rad=observation['yaw'], upright_cos=observation['upright_cos'],
                       contact_count=observation['contact_count'], sensors_used=sensory.tolist(),
                       sensors_pending=self.pending_sensors.tolist(), action=applied_action.tolist(),
                       next_action=action.tolist(),
                       DNa02_L_hz=left, DNa02_R_hz=right, DNg100_mean_hz=walk,
                       active_neurons=int(np.count_nonzero(self.brain.rates > 1.)),
                       saturated_neurons=int(np.count_nonzero(self.brain.rates > .95*self.brain.r_max)),
                       mean_rate_hz=float(self.brain.rates.mean()), max_rate_hz=float(self.brain.rates.max()),
                       populations={name: dict(mean_hz=float(self.brain.rates[idx].mean()),
                                               active_gt_1hz=int(np.count_nonzero(self.brain.rates[idx] > 1.)))
                                    for name, idx in self.population_indices.items()},
                       ports_hz={name: float(self.brain.rates[idx].mean())
                                 for name, idx in self.port_indices.items() if len(idx)})
            self.history.append(row)
            self._validate_clock()
            return row
        except BaseException:
            self.failed = True
            raise

    def advance(self, duration_s):
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError('Invalid duration')
        ticks = round(duration_s*1e9/self.CONTROL_NS)
        if abs(ticks*self.CONTROL_NS-duration_s*1e9) > 1e-4:
            raise ValueError('Duration must be a multiple of 5 ms')
        for _ in range(ticks):
            self.step()

    def save(self, path):
        self._validate_clock()
        if self.failed:
            raise ValueError('Cannot promote a failed session to a continuing checkpoint')
        if self.source_identity != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Session source changed during the experiment')
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f'Checkpoint already exists: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.'+path.name+'-', dir=path.parent))
        try:
            self.brain.save_checkpoint(staging/'brain')
            state = dict(schema=1, time_ns=self.time_ns, ticks=self.ticks, config=self.config,
                         world=self.world.__dict__, body=self.body.state_dict(),
                         plasticity=self.plasticity.state_dict(), ports=self.ports,
                         port_identity=self.port_identity, population_indices=self.population_indices,
                         pending_sensors=self.pending_sensors, last_action=self.last_action,
                         motor_disconnected=self.motor_disconnected, sensory_replay=self.sensory_replay,
                         history=self.history, used_sensors=np.asarray(self.used_sensors).reshape((-1, 3)),
                         source_identity=self.source_identity)
            write_state(staging/'session', state)
            (staging/'source').mkdir()
            for name in SOURCES:
                shutil.copyfile(ROOT/'src'/name, staging/'source'/name)
            manifest = dict(schema='matrix_organism_session_v1', time_ns=self.time_ns,
                            neuron_count=self.brain.n_neurons, stored_edges=self.brain.W.nnz,
                            plasticity_enabled=self.plasticity.enabled,
                            biological_validation=False,
                            scaffold=['synthetic_olfactory_transduction', 'patch_contact_to_PAM08',
                                      'tonic_DNg100_drive', 'phenomenological_DN_decoder',
                                      'FlyGym_CPG_step_templates_position_servos_adhesion'],
                            files={str(p.relative_to(staging)): sha256(p)
                                   for p in sorted(staging.rglob('*')) if p.is_file()})
            (staging/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
            staging.rename(path)
            return path
        except BaseException:
            shutil.rmtree(staging)
            raise

    @classmethod
    def load(cls, path):
        from anatomical_rate_brain import AnatomicalRateBrain
        from anatomical_plasticity import CandidateGammaPlasticity
        from walking_body_checkpoint import ContinuableWalkingBody
        path = Path(path)
        manifest = json.loads((path/'manifest.json').read_text())
        if manifest.get('schema') != 'matrix_organism_session_v1':
            raise ValueError('Unknown organism checkpoint')
        actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()}
        if actual != set(manifest['files']) | {'manifest.json'}:
            raise ValueError('Incomplete or unrecognized session files')
        for name, digest in manifest['files'].items():
            if sha256(path/name) != digest:
                raise ValueError(f'Session integrity failed: {name}')
        state = read_state(path/'session')
        expected = {'schema', 'time_ns', 'ticks', 'config', 'world', 'body', 'plasticity', 'ports',
                    'port_identity', 'population_indices', 'pending_sensors', 'last_action',
                    'motor_disconnected', 'sensory_replay', 'history', 'used_sensors', 'source_identity'}
        if set(state) != expected or state['schema'] != 1:
            raise ValueError('Unsupported organism state schema')
        if state['source_identity'] != {name: sha256(ROOT/'src'/name) for name in SOURCES}:
            raise ValueError('Use the archived session source to restore this version')
        self = cls()
        self.brain = AnatomicalRateBrain.load_checkpoint(path/'brain')
        for name in expected-{'schema', 'world', 'body', 'plasticity', 'used_sensors'}:
            setattr(self, name, state[name])
        self.world = OdorPatchWorld()
        if set(self.world.__dict__) != set(state['world']):
            raise ValueError('Incomplete world state')
        self.world.__dict__.update(state['world'])
        self._index_ports()
        self.plasticity = CandidateGammaPlasticity.from_state(self.brain, state['plasticity'])
        self.body = ContinuableWalkingBody.from_state(state['body'])
        self.used_sensors = list(state['used_sensors'])
        self.failed = False
        try:
            self._validate_clock()
            if manifest['time_ns'] != self.time_ns or len(self.history) != self.ticks or len(self.used_sensors) != self.ticks:
                raise ValueError('History, manifest and session clocks disagree')
            if not np.array_equal(self.pending_sensors, self.world.sense(self.body.observe())):
                raise ValueError('Pending live sensor is inconsistent with body and world')
            return self
        except BaseException:
            self.body.close()
            raise

    def close(self):
        self.body.close()

"""Continuable FlyGym reference body, with its motor scaffolding preserved.

Unlike old M1E trajectories, this adapter recomputes forward dynamics BEFORE
every controller/physics step. Thus cached observations are a function of the
saved integration state; restarting does not substitute a stale contact cache.
This is an explicit integration convention, not a claim of native VNC walking.
"""
import hashlib
import importlib.metadata
import inspect
import pickle
from pathlib import Path
import numpy as np
import mujoco
from embodied_navigation import ExperimentalWalkingBody
from passive_body import reject_external_callbacks
from session_io import sha256


class ContinuableWalkingBody(ExperimentalWalkingBody):
    SIM_STATE = ('retraction_correction', 'stumbling_correction',
                 'retraction_persistence_counter')
    CPG_STATE = ('curr_phases', 'curr_magnitudes', 'intrinsic_freqs', 'intrinsic_amps')
    STATIC = ('intrinsic_freqs', 'intrinsic_amps', 'phase_biases',
              'coupling_weights', 'convergence_coefs', 'stumble_segments',
              'stumbling_force_threshold', 'correction_vectors', 'correction_rates',
              'amplitude_range', 'max_increment', 'retraction_persistence_duration',
              'retraction_persistence_initiation_threshold', 'right_leg_inversion',
              'stumbling_sensors', 'preprogrammed_steps', 'phasic_multiplier')

    def __init__(self, yaw=0., seed=0, dt=.0001, scaffold='no_contact_rules'):
        self.constructor = dict(yaw=float(yaw), seed=int(seed), dt=float(dt), scaffold=scaffold)
        super().__init__(**self.constructor)
        self.identity = self._identity()

    def _identity(self):
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=buffer)
        # Pickle is used ONLY as a hash serialization of trusted in-memory
        # coefficients. No pickle is stored or loaded from a checkpoint.
        static = {k: getattr(self.sim, k) for k in self.STATIC}
        cpg = self.sim.cpg_network
        static['cpg_fixed'] = {k: getattr(cpg, k) for k in
                              ('timestep', 'coupling_weights', 'phase_biases', 'convergence_coefs')}
        files = {str(Path(inspect.getfile(cls)).resolve()):
                 sha256(inspect.getfile(cls)) for cls in
                 (ExperimentalWalkingBody, ContinuableWalkingBody,
                  type(self.sim), type(self.fly), type(cpg), type(self.sim.preprogrammed_steps))}
        return dict(model=hashlib.sha256(buffer).hexdigest(),
                    controller=hashlib.sha256(pickle.dumps(static, protocol=4)).hexdigest(),
                    source=files, packages={p: importlib.metadata.version(p) for p in
                                           ('mujoco', 'flygym', 'dm_control', 'numpy', 'scipy')})

    def advance(self, descending, nsteps):
        if type(nsteps) is not int or nsteps < 0:
            raise ValueError('Physical steps must be a nonnegative integer')
        descending = np.asarray(descending, dtype=float)
        if (self.failed or descending.shape != (2,) or not np.isfinite(descending).all()
                or np.any(descending < 0) or np.any(descending > 1.2)):
            raise ValueError('Invalid neural motor command (declared envelope 0..1.2)')
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError('Undeclared external body force')
        for _ in range(nsteps):
            reject_external_callbacks()
            mujoco.mj_forward(self.model, self.data)
            _, _, terminated, truncated, _ = self.sim.step(descending)
            self.steps += 1
            self.max_actuator_force = max(self.max_actuator_force,
                                         float(np.max(np.abs(self.data.qfrc_actuator))))
            self.max_contact_count = max(self.max_contact_count, self.data.ncon)
            if (terminated or truncated or np.any(self.data.warning.number)
                    or not np.isfinite(self.data.qpos).all()
                    or not np.isfinite(self.data.qvel).all()
                    or abs(self.data.time-self.steps*self.dt) > 1e-8):
                self.failed = True
                raise FloatingPointError('Physical failure; no reset or rescue')

    def state_dict(self):
        reject_external_callbacks()
        if self.identity != self._identity():
            raise ValueError('Effective body/controller/source configuration changed')
        if self.failed or np.any(self.data.warning.number):
            raise ValueError('Cannot continue failed body as a healthy checkpoint')
        mujoco.mj_getState(self.model, self.data, self.state, self.spec)
        return dict(schema=1, constructor=self.constructor, identity=self.identity,
                    integration=self.state.copy(), steps=self.steps,
                    sim_time=float(self.sim.curr_time),
                    sim={k: getattr(self.sim, k).copy() for k in self.SIM_STATE},
                    cpg={k: getattr(self.sim.cpg_network, k).copy() for k in self.CPG_STATE},
                    cpg_rng=list(self.sim.cpg_network.random_state.get_state()),
                    sim_rng=self.sim.np_random.bit_generator.state,
                    last_adhesion=self.fly._last_adhesion.copy(),
                    max_actuator_force=self.max_actuator_force,
                    max_contact_count=self.max_contact_count)

    @classmethod
    def from_state(cls, payload):
        expected = {'schema', 'constructor', 'identity', 'integration', 'steps', 'sim_time',
                    'sim', 'cpg', 'cpg_rng', 'sim_rng', 'last_adhesion',
                    'max_actuator_force', 'max_contact_count'}
        if set(payload) != expected or payload['schema'] != 1:
            raise ValueError('Unsupported walking body state')
        body = cls(**payload['constructor'])
        try:
            if body.identity != payload['identity']:
                raise ValueError('Walking body configuration/source mismatch')
            state = payload['integration']
            if state.shape != body.state.shape or not np.isfinite(state).all():
                raise ValueError('Invalid physical integration state')
            if type(payload['steps']) is not int or payload['steps'] < 0:
                raise ValueError('Invalid physical clock')
            mujoco.mj_setState(body.model, body.data, state, body.spec)
            body.steps = payload['steps']
            body.sim.curr_time = payload['sim_time']
            if (abs(body.data.time-body.steps*body.dt) > 1e-8 or
                    body.sim.curr_time != body.data.time):
                raise ValueError('Inconsistent physical clocks')
            for key, fields, obj in [('sim', cls.SIM_STATE, body.sim),
                                     ('cpg', cls.CPG_STATE, body.sim.cpg_network)]:
                if set(payload[key]) != set(fields):
                    raise ValueError('Incomplete controller state')
                for name in fields:
                    value = payload[key][name]
                    if value.shape != (6,) or not np.isfinite(value).all():
                        raise ValueError('Invalid controller array')
                    setattr(obj, name, value.copy())
            rng = list(payload['cpg_rng'])
            body.sim.cpg_network.random_state.set_state(tuple(rng))
            body.sim.np_random.bit_generator.state = payload['sim_rng']
            adhesion = payload['last_adhesion']
            if adhesion.shape != (6,) or not np.isin(adhesion, [0, 1]).all():
                raise ValueError('Invalid adhesion state')
            body.fly._last_adhesion = adhesion.copy()
            body.max_actuator_force = float(payload['max_actuator_force'])
            body.max_contact_count = int(payload['max_contact_count'])
            return body
        except BaseException:
            body.close()
            raise

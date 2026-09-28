"""FlyBody physical preparation on the continuing CNS clock.

No learned controller or position actuator. The old body is archived, not
mistaken for this body's incompatible generalized coordinates. Absolute time
is inherited; elapsed physical preparation time starts at origin_ns.
"""
import copy
import mujoco
import numpy as np
from flybody_torque_port import FlyBodyTorquePort, FlyBodyEffectiveTibia, FLYGYM_TORQUE_TO_NM
from session_io import sha256


class CNSFlyBody(FlyBodyTorquePort):
    def __init__(self):
        super().__init__()
        self.origin_ns = 0
        self.identity.update(cns_adapter_sha256=sha256(__file__),
            initialization='Source qpos0, floor and zero velocity; explicit CNS clock origin; no physical pose/history migration',
            neural_closed_loop=True, neural_closed_loop_is_configuration_not_validation=True)

    @classmethod
    def at_clock(cls, time_ns):
        obj = cls()
        if type(time_ns) is not int or time_ns < 0 or time_ns % round(obj.dt*1e9):
            raise ValueError('CNS clock must align with the physical step')
        obj.origin_ns = time_ns
        obj.steps = time_ns // round(obj.dt*1e9)
        obj.data.time = time_ns*1e-9
        return obj

    def observe(self):
        out = super().observe()
        thorax = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')
        rotation = self.scratch.xmat[thorax].reshape(3,3)
        out.update(position_mm=self.scratch.xpos[thorax].copy()*10.,
            yaw=float(np.arctan2(rotation[1,0], rotation[0,0])),
            upright_cos=float(rotation[2,2]))
        return out

    def advance(self, torque_native, nsteps=1):
        self.advance_joint_torque_si(np.asarray(torque_native)*FLYGYM_TORQUE_TO_NM, nsteps)

    def state_dict(self):
        out = super().state_dict()
        out.update(schema='matrix_flybody_cns_body_v1', origin_ns=self.origin_ns)
        return out

    @classmethod
    def from_state(cls, state):
        if set(state) != {'schema','identity','steps','integration','origin_ns'} or state['schema'] != 'matrix_flybody_cns_body_v1':
            raise ValueError('Wrong CNS FlyBody state')
        base = copy.deepcopy(state)
        origin = base.pop('origin_ns')
        base['schema'] = 'flybody_torque_port_v1'
        obj = super().from_state(base)
        if type(origin) is not int or origin < 0 or origin > obj.steps*round(obj.dt*1e9) or origin % round(obj.dt*1e9):
            raise ValueError('Invalid physical preparation origin')
        obj.origin_ns = origin
        return obj

    def close(self):
        pass


class CNSFlyBodyMuscles(FlyBodyEffectiveTibia):
    @classmethod
    def from_state(cls, body, state):
        default=cls(body)
        if state.get('parameters')!=default.parameters:
            raise ValueError('Undeclared muscle parameter change')
        obj=super().from_state(body,state)
        if (type(obj.time_ns) is not int or obj.time_ns!=round(body.data.time*1e9)
                or obj.last_force.shape!=(6,2) or obj.last_torque.shape!=(6,)
                or not np.isfinite(obj.last_force).all() or not np.isfinite(obj.last_torque).all()
                or np.any(obj.last_force<0)):
            raise ValueError('Invalid muscle clock or force history')
        return obj

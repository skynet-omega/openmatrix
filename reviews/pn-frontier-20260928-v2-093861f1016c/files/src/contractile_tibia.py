"""Twelve effective contractile elements driven ONLY by 27 anatomical MNs.

Generic activation and Hill-like active length/velocity relations are explicit
engineering hypotheses. These are not reconstructed individual fly muscles.
There is no target joint angle, oscillator, gait phase, posture feedback,
artificial adhesion or force chosen from vision. Source passive joints remain.
"""
import copy
import mujoco
import numpy as np
from native_motor_body import LEGS, MAX_TORQUE_NATIVE


class ContractileTibia:
    def __init__(self, body):
        self.body = body
        self.parameters = dict(activation_tau_s=.01, deactivation_tau_s=.04,
            moment_arm_native=.05, optimal_length_native=.4, maximum_shortening_lengths_s=5.,
            active_length_width=.45, maximum_lengthening_factor=1.5,
            force_max_native=MAX_TORQUE_NATIVE/(.05*1.5),
            status="UNCALIBRATED_EFFECTIVE_CONTRACTILE_PAIRS_NOT_ANATOMICAL_MUSCLES")
        self.activation = np.zeros((6, 2), dtype=np.float64)
        self.last_force = np.zeros((6, 2), dtype=np.float64)
        self.last_torque = np.zeros(6, dtype=np.float64)
        self.time_ns = round(body.data.time*1e9)
        self._geometry()

    def _geometry(self):
        m = self.body.model
        joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"matrixfly/joint_{leg}Tibia") for leg in LEGS]
        if min(joints) < 0:
            raise ValueError("Missing tibial mechanics")
        self.qadr = m.jnt_qposadr[joints]
        self.vadr = m.jnt_dofadr[joints]
        # Reference only defines the uncalibrated force-length geometry; it is
        # never subtracted as an error driving neural activity or joint torque.
        self.reference_q = self.body.sign_calibration_qpos[self.qadr].copy()

    def advance(self, excitation, dt_ns):
        u = np.asarray(excitation, dtype=float)
        if u.shape != (6, 2) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)) or type(dt_ns) is not int or dt_ns <= 0:
            raise ValueError("Contractile excitation must be normalized MN output")
        p = self.parameters
        tau = np.where(u > self.activation, p["activation_tau_s"], p["deactivation_tau_s"])
        self.activation += -np.expm1(-dt_ns*1e-9/tau)*(u-self.activation)
        sign = self.body.flexion_sign
        q = sign*(self.body.data.qpos[self.qadr]-self.reference_q)
        v = sign*self.body.data.qvel[self.vadr]
        moment = p["moment_arm_native"]
        length = p["optimal_length_native"] + moment*q[:, None]*np.array([-1., 1.])
        velocity = moment*v[:, None]*np.array([-1., 1.])
        if np.any(length <= 0.):
            raise ValueError("Effective muscle length invalid; no force rescue")
        fl = np.exp(-((length/p["optimal_length_native"]-1.)/p["active_length_width"])**2)
        vn = velocity/(p["maximum_shortening_lengths_s"]*p["optimal_length_native"])
        # Shortening reduces active tension; lengthening approaches 1.5 Fmax.
        fv = np.where(vn < 0., 1./(1.+np.maximum(-vn, 0.)),
                      1.+(p["maximum_lengthening_factor"]-1.)*np.tanh(np.maximum(vn, 0.)))
        self.last_force = p["force_max_native"]*self.activation*fl*fv
        self.last_torque = sign*moment*(self.last_force[:, 0]-self.last_force[:, 1])
        if not np.isfinite(self.last_torque).all() or np.any(np.abs(self.last_torque) > MAX_TORQUE_NATIVE+1e-12):
            raise FloatingPointError("Contractile force bound violated")
        self.time_ns += dt_ns
        return self.last_torque.copy()

    def state_dict(self):
        return dict(parameters=copy.deepcopy(self.parameters), activation=self.activation.copy(),
                    last_force=self.last_force.copy(), last_torque=self.last_torque.copy(),
                    reference_q=self.reference_q.copy(), time_ns=self.time_ns)

    @classmethod
    def from_state(cls, body, state):
        obj = cls(body)
        if set(state) != set(obj.state_dict()) or not np.array_equal(state["reference_q"], obj.reference_q):
            raise ValueError("Incomplete/different effective muscle geometry")
        for key, value in state.items():
            setattr(obj, key, copy.deepcopy(value))
        if obj.activation.shape != (6, 2) or not np.isfinite(obj.activation).all() or np.any((obj.activation < 0) | (obj.activation > 1)):
            raise ValueError("Invalid contractile activation")
        return obj

"""Effective RH depressor port with canonical input, preserving the committed millisecond."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import mujoco as mj
from guarded_tibia_body import GuardedTibiaBody
from flybody_torque_port import FLYBODY_TORQUE_TO_NM
from rf_tarsal_body import PRIOR, PRIOR_SHA256
from session_io import sha256


class RHTarsalBody(GuardedTibiaBody):
    SCHEMA = "matrix_rh_tarsal_body_v1"

    @classmethod
    def from_parent(cls, parent, *, enabled):
        if type(parent) is not GuardedTibiaBody or type(enabled) is not bool:
            raise ValueError("Expected original guarded body and explicit intervention mode")
        parent.validate_coxa()
        if sha256(PRIOR) != PRIOR_SHA256:
            raise ValueError("Changed RF source prior")
        obj = cls.__new__(cls)
        obj.__dict__.update(parent.__dict__)
        obj.rh_parent_identity = copy.deepcopy(parent.identity)
        obj.rh_prior = json.loads(PRIOR.read_text())
        obj.rh_joint = obj.model.joint("tarsus_T3_right").id
        obj.rh_qadr = int(obj.model.jnt_qposadr[obj.rh_joint])
        obj.rh_vadr = int(obj.model.jnt_dofadr[obj.rh_joint])
        obj.rh_landmarks = [obj.model.body(n).id for n in
                            ["tibia_T3_right", "tarsus_T3_right", "tarsus2_T3_right"]]
        neutral = mj.MjData(obj.model)
        mj.mj_forward(obj.model, neutral)
        _, sine = obj.rh_angle_from(neutral)
        obj.rh_sign = float(np.sign(sine))
        obj.rh_enabled = enabled
        obj.rh_activation = 0.0
        obj.rh_pending = 0.0
        obj.rh_last_torque_Nm = 0.0
        obj.rh_origin_ns = obj.steps * round(obj.dt * 1e9)
        obj.rh_time_ns = obj.rh_origin_ns
        obj.identity = copy.deepcopy(obj.identity)
        obj.identity.update(rh_tarsal_source_sha256=sha256(__file__),
                            rh_tarsal_prior_sha256=PRIOR_SHA256,
                            rh_tarsal_joint="tarsus_T3_right",
                            rh_tarsal_depressor_sign=obj.rh_sign,
                            rh_tarsal_candidate_id=800358,
                            rh_tarsal_model_prior=True,
                            rh_tarsal_admission_delay_ns=1000000)
        obj.validate_coxa()
        return obj

    def rh_angle_from(self, data):
        axis = data.xaxis[self.rh_joint]
        x = data.xpos[self.rh_landmarks]
        u, v = (x[0] - x[1]).copy(), (x[2] - x[1]).copy()
        u -= u.dot(axis) * axis
        v -= v.dot(axis) * axis
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        sine = axis.dot(np.cross(u, v)) / norm
        if not np.isfinite(sine) or norm <= 0 or abs(sine) < 1e-10:
            raise ValueError("Undefined RH native projected angle")
        return float(np.arctan2(abs(sine), np.clip(u.dot(v) / norm, -1., 1.))), sine

    def _inherited_generalized(self):
        result = super()._inherited_generalized()
        result[self.rh_vadr] += self.rh_last_torque_Nm
        return result

    def _muscle_step(self):
        super()._muscle_step()
        a = self.rh_activation
        u = self.rh_pending if self.rh_enabled and self.rh_time_ns >= self.rh_origin_ns + 1000000 else 0.0
        tau = self.rh_prior["activation_tau_s" if u > a else "deactivation_tau_s"]
        anew = a + (-np.expm1(-self.dt / tau)) * (u - a)
        self.rh_last_torque_Nm = float(self.rh_prior["torque_budget_Nm"] * self.rh_sign * .5 * (a + anew))
        self.rh_activation = float(anew)

    def advance(self, torque_native, nsteps=1):
        super().advance(torque_native, nsteps)
        self.rh_time_ns += round(self.dt * 1e9)
        self.validate_coxa()

    def validate_coxa(self):
        super().validate_coxa()
        if type(self.rh_enabled) is not bool:
            raise ValueError("Invalid RH mode")
        if not all(np.isfinite(x) and 0 <= x <= 1 for x in [self.rh_activation, self.rh_pending]):
            raise ValueError("Invalid RH activation or command")
        if (not np.isfinite(self.rh_last_torque_Nm)
                or abs(self.rh_last_torque_Nm) > self.rh_prior["torque_budget_Nm"] * (1 + 1e-12)):
            raise ValueError("Invalid RH torque")
        if (type(self.rh_time_ns) is not int or type(self.rh_origin_ns) is not int
                or not 0 <= self.rh_origin_ns <= self.rh_time_ns
                or self.rh_time_ns != self.steps * round(self.dt * 1e9)):
            raise ValueError("Invalid RH clock")

    def state_dict(self):
        self.validate_coxa()
        actual = self.integration_state().copy()
        base = GuardedTibiaBody.state_dict(self)
        # Construct the parent loader view in scratch; never invalidate live forces.
        scratch = mj.MjData(self.model)
        mj.mj_setState(self.model, scratch, actual, self.spec)
        parent_force = GuardedTibiaBody._inherited_generalized(self) + self.last_serial_generalized_SI
        scratch.qfrc_applied[:] = parent_force / FLYBODY_TORQUE_TO_NM
        parent_integration = np.empty_like(actual)
        mj.mj_getState(self.model, scratch, parent_integration, self.spec)
        base["integration"] = parent_integration
        base["schema"] = GuardedTibiaBody.SCHEMA
        base["identity"] = copy.deepcopy(self.rh_parent_identity)
        # Parent loader uses its fixed 25us clock; saved checkpoints are on ms boundaries.
        if self.rh_time_ns % 25000:
            raise ValueError("Snapshot must align with the inherited 25us clock")
        base["steps"] = self.rh_time_ns // 25000
        np.testing.assert_array_equal(self.integration_state(), actual)
        return dict(schema=self.SCHEMA, identity=copy.deepcopy(self.identity), parent=base,
                    integration=actual, dt_ns=round(self.dt * 1e9),
                    rh=dict(enabled=self.rh_enabled, activation=self.rh_activation,
                            pending=self.rh_pending, last_torque_Nm=self.rh_last_torque_Nm,
                            origin_ns=self.rh_origin_ns, time_ns=self.rh_time_ns))

    @classmethod
    def from_state(cls, state):
        if set(state) != {"schema", "identity", "parent", "integration", "dt_ns", "rh"} or state["schema"] != cls.SCHEMA:
            raise ValueError("Wrong RH state schema")
        if state["dt_ns"] not in (25000, 12500):
            raise ValueError("Unplanned RH timestep")
        r = state["rh"]
        if set(r) != {"enabled", "activation", "pending", "last_torque_Nm", "origin_ns", "time_ns"}:
            raise ValueError("Incomplete RH history")
        obj = cls.from_parent(GuardedTibiaBody.from_state(state["parent"]), enabled=r["enabled"])
        if state["identity"] != obj.identity:
            raise ValueError("Changed RH identity")
        obj.dt = state["dt_ns"] * 1e-9
        obj.model.opt.timestep = obj.dt
        obj.steps = r["time_ns"] // state["dt_ns"]
        obj.rh_activation, obj.rh_pending = r["activation"], r["pending"]
        obj.rh_last_torque_Nm = r["last_torque_Nm"]
        obj.rh_origin_ns, obj.rh_time_ns = r["origin_ns"], r["time_ns"]
        if state["integration"].shape != obj.integration_state().shape or not np.isfinite(state["integration"]).all():
            raise ValueError("Invalid RH integration state")
        mj.mj_setState(obj.model, obj.data, state["integration"], obj.spec)
        obj.validate_coxa()
        if abs(obj.data.time - obj.steps * obj.dt) > 1e-10:
            raise ValueError("RH physical clock differs")
        return obj

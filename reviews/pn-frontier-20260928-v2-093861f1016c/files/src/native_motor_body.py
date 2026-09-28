"""Free FlyGym body with six direct tibial torque actuators and no gait policy.

No HybridTurningController, oscillator, pose target, adhesion or rescue acts on
this body. The remaining joints retain source passive mechanics. Torque scale
0.01 is an a priori engineering hypothesis in native model units, NOT a fit,
measured force or physiological muscle. Geometric flexion signs are determined
by an infinitesimal change of the interior femur-tibia angle at initial pose.
"""
import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import mujoco
import numpy as np

from embodied_navigation import ExperimentalWalkingBody
from passive_body import reject_external_callbacks
from session_io import sha256


LEGS = ("LF", "RF", "LM", "RM", "LH", "RH")
MAX_TORQUE_NATIVE = .01


class NativeMotorBody(ExperimentalWalkingBody):
    """Reuse the source body's pure physical observer, never its controller."""

    def __init__(self, yaw=0., seed=0, dt=.0001):
        from flygym import Fly, SingleFlySimulation
        from flygym.preprogrammed import default_leg_sensor_placements
        if dt != .0001 or not np.isfinite(yaw) or type(seed) is not int or seed < 0:
            raise ValueError("Native v1 requires finite yaw, nonnegative seed and 100us physics")
        self.constructor = dict(yaw=float(yaw), seed=seed, dt=float(dt))
        self.dt = dt
        self.fly = Fly(name="matrixfly", spawn_orientation=(0., 0., float(yaw)),
                       control="motor", actuated_joints=[f"joint_{leg}Tibia" for leg in LEGS],
                       actuator_gain=None, actuator_forcerange=MAX_TORQUE_NATIVE,
                       enable_adhesion=False, draw_adhesion=False,
                       contact_sensor_placements=default_leg_sensor_placements,
                       head_stabilization_model=None, enable_vision=False, enable_olfaction=False)
        # FlyGym 1.2.1 creates adhesion actuators even when its flag is false.
        # Remove those MJCF elements before compilation, not merely their input.
        for actuator in self.fly.adhesion_actuators:
            actuator.remove()
        self.fly.adhesion_actuators = []
        self.sim = SingleFlySimulation(fly=self.fly, cameras=[], timestep=dt)
        self.sim.reset(seed=seed)
        self.model, self.data = self.sim.physics.model.ptr, self.sim.physics.data.ptr
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        self.scratch = mujoco.MjData(self.model)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.spec))
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "matrixfly/Thorax")
        self.steps = 0
        self.failed = False
        self.max_actuator_force = 0.
        self.max_contact_count = 0
        self.torque_indices = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"matrixfly/actuator_motor_joint_{leg}Tibia") for leg in LEGS])
        if self.model.nu != 6 or np.any(self.torque_indices < 0) or self.model.na != 0:
            raise ValueError("Native body must have exactly six memoryless joint motors")
        self.sign_calibration_qpos = self.data.qpos.copy()
        self.flexion_sign, self.angle_derivatives = self._geometric_signs()
        self.replacement = None
        self.identity = self._identity()
        reject_external_callbacks()

    def _identity(self):
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=buffer)
        classes = (NativeMotorBody, ExperimentalWalkingBody, type(self.fly), type(self.sim))
        return {"model_sha256": hashlib.sha256(buffer).hexdigest(),
                "source": {str(Path(inspect.getfile(c)).resolve()): sha256(inspect.getfile(c)) for c in classes},
                "packages": {p: importlib.metadata.version(p) for p in ("mujoco", "flygym", "dm_control", "numpy", "scipy")},
                "torque_limit_native": MAX_TORQUE_NATIVE,
                "no_cpg_templates_position_servos_adhesion": True}

    def _angle(self, view, leg):
        b = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"matrixfly/{leg}{seg}")
             for seg in ("Femur", "Tibia", "Tarsus1")]
        j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"matrixfly/joint_{leg}Tibia")
        if min(b + [j]) < 0:
            raise ValueError("Missing tibial geometric landmarks")
        axis = view.xaxis[j]
        u, v = view.xpos[b[0]] - view.xpos[b[1]], view.xpos[b[2]] - view.xpos[b[1]]
        u, v = u - (u @ axis) * axis, v - (v @ axis) * axis
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Degenerate tibial angle geometry")
        return np.arccos(np.clip((u @ v) / norm, -1., 1.))

    def _geometric_signs(self):
        view = mujoco.MjData(self.model)
        baseline = self.sign_calibration_qpos
        derivatives = []
        epsilon = 1e-5
        for leg in LEGS:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"matrixfly/joint_{leg}Tibia")
            angles = []
            for delta in (-epsilon, epsilon):
                view.qpos[:] = baseline
                view.qpos[self.model.jnt_qposadr[j]] += delta
                mujoco.mj_forward(self.model, view)
                angles.append(self._angle(view, leg))
            derivatives.append((angles[1] - angles[0]) / (2 * epsilon))
        derivative = np.asarray(derivatives)
        if not np.isfinite(derivative).all() or np.any(np.abs(derivative) < .5):
            raise ValueError("Geometric flexion sign is unresolved at this pose")
        return -np.sign(derivative), derivative

    def adopt_initial_reference(self, reference):
        """Explicit efector replacement at t=0, with coordinate/frame checks."""
        if reference.steps != 0 or reference.data.time != 0 or self.steps != 0:
            raise ValueError("Native v1 replacement only supports the initial t=0 checkpoint")
        a, b = reference.model, self.model
        if (a.nq, a.nv, a.njnt, a.nbody, a.ngeom) != (b.nq, b.nv, b.njnt, b.nbody, b.ngeom):
            raise ValueError("Reference/native physical coordinate or geometry dimensions differ")
        for kind, n in ((mujoco.mjtObj.mjOBJ_JOINT, a.njnt), (mujoco.mjtObj.mjOBJ_BODY, a.nbody),
                        (mujoco.mjtObj.mjOBJ_GEOM, a.ngeom)):
            if [mujoco.mj_id2name(a, kind, i) for i in range(n)] != [mujoco.mj_id2name(b, kind, i) for i in range(n)]:
                raise ValueError("Reference/native anatomical model names differ")
        fields = ("jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos",
                  "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat",
                  "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_dataid",
                  "mesh_vert", "mesh_face")
        for name in fields:
            if not np.array_equal(getattr(a, name), getattr(b, name)):
                raise ValueError(f"Reference/native geometry or inertia differs: {name}")
        if not np.array_equal(a.opt.gravity, b.opt.gravity):
            raise ValueError("Reference/native gravity differs")
        # Same named joints and addresses established above: this is an exact
        # coordinate transfer, not a fit to a desired pose or new trajectory.
        self.data.qpos[:] = reference.data.qpos
        self.data.qvel[:] = reference.data.qvel
        self.data.qacc_warmstart[:] = 0.
        self.data.ctrl[:] = 0.
        self.data.time = 0.
        self.sim.curr_time = 0.
        self.sign_calibration_qpos = self.data.qpos.copy()
        self.flexion_sign, self.angle_derivatives = self._geometric_signs()
        self.replacement = {"time_ns": 0, "qpos_qvel": "exact named-coordinate transfer",
                            "geometry_and_mass_fields_checked": list(fields),
                            "warmstart": "reset explicitly after actuator/passive-mechanics replacement",
                            "former_position_targets_cpg_phase_adhesion": "removed, not transferred",
                            "unactuated_joints": "source passive coefficients apply; changes are in model hash"}

    def advance(self, torques, nsteps):
        torques = np.asarray(torques, dtype=float)
        if (self.failed or type(nsteps) is not int or nsteps < 0 or torques.shape != (6,)
                or not np.isfinite(torques).all() or np.any(np.abs(torques) > MAX_TORQUE_NATIVE + 1e-12)):
            raise ValueError("Invalid native torque command or physical step count")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError("Undeclared external body force")
        for _ in range(nsteps):
            reject_external_callbacks()
            mujoco.mj_forward(self.model, self.data)
            _, _, terminated, truncated, _ = self.sim.step({"joints": torques})
            self.steps += 1
            self.max_actuator_force = max(self.max_actuator_force, float(np.max(np.abs(self.data.qfrc_actuator))))
            self.max_contact_count = max(self.max_contact_count, self.data.ncon)
            if (terminated or truncated or np.any(self.data.warning.number)
                    or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                    or abs(self.data.time - self.steps * self.dt) > 1e-8):
                self.failed = True
                raise FloatingPointError("Native body failed; no reset or rescue")

    def state_dict(self):
        reject_external_callbacks()
        if self.identity != self._identity() or self.failed or np.any(self.data.warning.number):
            raise ValueError("Changed configuration or failed native body cannot be checkpointed")
        mujoco.mj_getState(self.model, self.data, self.state, self.spec)
        return {"schema": 1, "constructor": self.constructor, "identity": self.identity,
                "integration": self.state.copy(), "steps": self.steps, "sim_time": self.sim.curr_time,
                "sim_rng": self.sim.np_random.bit_generator.state,
                "flexion_sign": self.flexion_sign, "angle_derivatives": self.angle_derivatives,
                "sign_calibration_qpos": self.sign_calibration_qpos,
                "replacement": self.replacement, "max_actuator_force": self.max_actuator_force,
                "max_contact_count": self.max_contact_count}

    @classmethod
    def from_state(cls, state):
        keys = {"schema", "constructor", "identity", "integration", "steps", "sim_time", "sim_rng",
                "flexion_sign", "angle_derivatives", "sign_calibration_qpos", "replacement", "max_actuator_force", "max_contact_count"}
        if set(state) != keys or state["schema"] != 1:
            raise ValueError("Unknown native body schema")
        body = cls(**state["constructor"])
        try:
            if body.identity != state["identity"]:
                raise ValueError("Native body model/source/runtime identity differs")
            if type(state["steps"]) is not int or state["steps"] < 0:
                raise ValueError("Invalid physical clock")
            value = state["integration"]
            if value.shape != body.state.shape or not np.isfinite(value).all():
                raise ValueError("Invalid native integration state")
            calibration = state["sign_calibration_qpos"]
            if calibration.shape != body.data.qpos.shape or not np.isfinite(calibration).all():
                raise ValueError("Invalid sign-calibration geometry")
            body.sign_calibration_qpos = calibration.copy()
            body.flexion_sign, body.angle_derivatives = body._geometric_signs()
            for key in ("flexion_sign", "angle_derivatives"):
                if not np.array_equal(state[key], getattr(body, key)):
                    raise ValueError("Native geometric motor signs differ")
            mujoco.mj_setState(body.model, body.data, value, body.spec)
            body.steps = state["steps"]
            body.sim.curr_time = state["sim_time"]
            if abs(body.data.time - body.steps * body.dt) > 1e-8 or body.sim.curr_time != body.data.time:
                raise ValueError("Inconsistent native physical clocks")
            body.sim.np_random.bit_generator.state = state["sim_rng"]
            body.replacement = state["replacement"]
            body.max_actuator_force = float(state["max_actuator_force"])
            body.max_contact_count = int(state["max_contact_count"])
            return body
        except BaseException:
            body.close()
            raise

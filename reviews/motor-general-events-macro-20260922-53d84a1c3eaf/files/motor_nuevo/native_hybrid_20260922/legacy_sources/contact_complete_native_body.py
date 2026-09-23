"""Native six-tibia torque body with ground contact for every existing geom.

The former FlyGym default admitted ground contacts for legs only, so an
overturned torso could pass below the floor without a solver warning. This
explicit model intervention adds physical contact pairs for all source body
geometries. It changes no mass, passive coefficient, muscle, neural command,
pose, gravity or motor law. It is not a fall rescue or a biological calibration.
"""
from pathlib import Path
import copy

import mujoco
import numpy as np

from native_motor_body import NativeMotorBody, LEGS, MAX_TORQUE_NATIVE
from passive_body import reject_external_callbacks
from session_io import sha256


class ContactCompleteNativeBody(NativeMotorBody):
    """Same physics and six motors, with all body-surface ground contacts."""

    def __init__(self, yaw=0., seed=0, dt=.0001):
        from flygym import Fly, SingleFlySimulation
        from flygym.preprogrammed import default_leg_sensor_placements
        if dt != .0001 or not np.isfinite(yaw) or type(seed) is not int or seed < 0:
            raise ValueError("Native contacts require finite yaw, nonnegative seed and 100us physics")
        self.constructor = dict(yaw=float(yaw), seed=seed, dt=float(dt))
        self.dt = dt
        self.fly = Fly(name="matrixfly", spawn_orientation=(0., 0., float(yaw)),
                       control="motor", actuated_joints=[f"joint_{leg}Tibia" for leg in LEGS],
                       actuator_gain=None, actuator_forcerange=MAX_TORQUE_NATIVE,
                       enable_adhesion=False, draw_adhesion=False, floor_collisions="all",
                       contact_sensor_placements=default_leg_sensor_placements,
                       head_stabilization_model=None, enable_vision=False, enable_olfaction=False)
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
        ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        contacted = set()
        for a, b in zip(self.model.pair_geom1, self.model.pair_geom2):
            if ground in (a, b):
                contacted.add(int(b if a == ground else a))
        actual_body_geoms = {i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] != 0}
        if ground < 0 or contacted != actual_body_geoms:
            raise ValueError("Every source body geometry requires an explicit ground contact pair")
        self.identity = self._identity()
        reject_external_callbacks()

    def _identity(self):
        identity = super()._identity()
        identity["source"][str(Path(__file__).resolve())] = sha256(__file__)
        ground = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        pairs = []
        for a, b in zip(self.model.pair_geom1, self.model.pair_geom2):
            if ground in (a, b):
                pairs.append([mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in (a, b)])
        identity.update(body_class="ContactCompleteNativeBody", ground_contact_scope="all_existing_body_geometries",
                        explicit_ground_pairs=pairs, reference_passive_coefficients_preserved=True)
        return identity

    def adopt_native(self, reference):
        """Transfer a recorded life at its current time; reset only warmstart.

Contact constraints have changed, so the old constraint solver's acceleration
warmstart is explicitly discarded. Every generalized position/velocity, motor
input, random-generator state and clock is carried across without repositioning.
"""
        if self.steps != 0 or self.data.time != 0 or not isinstance(reference, NativeMotorBody):
            raise ValueError("Contact intervention needs a fresh target and a recorded native source")
        state = reference.state_dict()  # Reject a failed or modified source first.
        a, b = reference.model, self.model
        for name in ("nq", "nv", "njnt", "nbody", "ngeom", "nu", "na", "nmocap", "nuserdata"):
            if getattr(a, name) != getattr(b, name):
                raise ValueError(f"Physical model dimension differs: {name}")
        for kind, n in ((mujoco.mjtObj.mjOBJ_JOINT, a.njnt), (mujoco.mjtObj.mjOBJ_BODY, a.nbody),
                        (mujoco.mjtObj.mjOBJ_GEOM, a.ngeom), (mujoco.mjtObj.mjOBJ_ACTUATOR, a.nu)):
            if [mujoco.mj_id2name(a, kind, i) for i in range(n)] != [mujoco.mj_id2name(b, kind, i) for i in range(n)]:
                raise ValueError("Native/contact-complete physical names differ")
        fields = ("jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_axis", "jnt_pos", "jnt_range",
                  "jnt_stiffness", "dof_damping", "dof_armature", "dof_frictionloss", "qpos_spring", "qpos0",
                  "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat", "body_gravcomp",
                  "geom_type", "geom_bodyid", "geom_pos", "geom_quat", "geom_size", "geom_dataid", "geom_friction",
                  "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity", "geom_margin", "geom_gap",
                  "mesh_vert", "mesh_face", "actuator_trntype", "actuator_trnid", "actuator_dyntype",
                  "actuator_gaintype", "actuator_biastype", "actuator_dynprm", "actuator_gainprm",
                  "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear")
        for name in fields:
            if not np.array_equal(getattr(a, name), getattr(b, name)):
                raise ValueError(f"Non-contact physical field differs: {name}")
        option_fields = ("timestep", "gravity", "wind", "density", "viscosity", "integrator", "cone", "solver",
                         "iterations", "tolerance", "disableflags", "enableflags")
        for name in option_fields:
            if not np.array_equal(getattr(a.opt, name), getattr(b.opt, name)):
                raise ValueError(f"Physical solver option differs: {name}")
        if state["integration"].shape != self.state.shape:
            raise ValueError("Integration state dimensions differ")
        mujoco.mj_setState(self.model, self.data, state["integration"], self.spec)
        self.data.qacc_warmstart[:] = 0.
        self.steps = reference.steps
        self.sim.curr_time = reference.sim.curr_time
        self.sim.np_random.bit_generator.state = copy.deepcopy(reference.sim.np_random.bit_generator.state)
        self.sign_calibration_qpos = reference.sign_calibration_qpos.copy()
        self.flexion_sign, self.angle_derivatives = self._geometric_signs()
        for name in ("flexion_sign", "angle_derivatives"):
            if not np.array_equal(getattr(self, name), getattr(reference, name)):
                raise ValueError("Physical contact intervention changed geometric motor directions")
        self.max_actuator_force = reference.max_actuator_force
        self.max_contact_count = reference.max_contact_count
        self.replacement = {
            "time_s": float(reference.data.time), "time_ns": round(reference.data.time * 1e9),
            "operation": "Add ground contact pairs for all existing physical body geometries",
            "parent_physical_identity": copy.deepcopy(reference.identity),
            "parent_replacement": copy.deepcopy(reference.replacement),
            "checked_equal_fields": list(fields), "checked_equal_solver_options": list(option_fields),
            "integration_transfer": "All integration fields copied; only qacc_warmstart explicitly reset to zero",
            "qpos_qvel_ctrl_clock_rng": "exact transfer without repositioning or motor intervention",
            "warmstart_reason": "Old constraint solver acceleration is not valid for the new contact set",
            "new_ground_contact_pairs": copy.deepcopy(self.identity["explicit_ground_pairs"]),
            "new_active_constraints_are_physical": True,
            "no_pose_target_rescue_or_reset": True,
        }
        if (not np.array_equal(self.data.qpos, reference.data.qpos)
                or not np.array_equal(self.data.qvel, reference.data.qvel)
                or not np.array_equal(self.data.ctrl, reference.data.ctrl)
                or self.data.time != reference.data.time):
            raise ValueError("Contact intervention failed exact named-state preservation")
        return copy.deepcopy(self.replacement)

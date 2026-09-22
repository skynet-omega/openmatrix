"""Passive, compiled NeuroMechFly body. No neural/muscle mapping is claimed.

The original model's stiffness, damping and contacts remain approximations.
Only the state used by native MuJoCo is evolved; no FlyGym behavior policy runs.
"""
from pathlib import Path
import copy
import hashlib
import numpy as np
import mujoco


def reject_external_callbacks():
    for name in ("control", "passive", "sensor", "act_dyn", "act_gain", "act_bias", "contactfilter"):
        getter = getattr(mujoco, "get_mjcb_" + name, None)
        if getter is not None and getter() is not None:
            raise RuntimeError(f"External MuJoCo {name} callback active in passive preparation")


class PassiveFlyBody:
    def __init__(self, model_path, initial_state_path, dt_us=200):
        model_path = Path(model_path)
        self.dt_us = int(dt_us)
        if self.dt_us <= 0 or self.dt_us != dt_us:
            raise ValueError("dt_us must be a positive integer")
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.state_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.identity = {
            "kind": "passive_neuromechfly_no_neural_ports",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "initial_state_sha256": hashlib.sha256(Path(initial_state_path).read_bytes()).hexdigest(),
            "dt_us": self.dt_us,
            "mujoco": mujoco.__version__,
            "state_spec": self.state_spec,
        }
        if abs(self.model.opt.timestep - self.dt_us / 1e6) > 1e-15:
            raise ValueError("Compiled physical timestep differs from requested timestep")
        reject_external_callbacks()
        if np.any(self.model.actuator_biasprm):
            raise RuntimeError("Passive test requires zero actuator bias: no position servos")
        self.steps = 0
        self.warning_count = np.zeros(len(self.data.warning.number), dtype=np.int64)
        self.failed = False
        # Observation is evaluated on separate data, never on the integrator's
        # data. Its warm-start and contact/solver state therefore remain intact.
        self._observation_data = None
        self._observation_schema = None
        state = np.load(initial_state_path, allow_pickle=False)
        if state.shape != (mujoco.mj_stateSize(self.model, self.state_spec),):
            raise ValueError("Initial integration state shape mismatch")
        mujoco.mj_setState(self.model, self.data, state, self.state_spec)
        if self.data.time != 0 or np.any(self.data.ctrl):
            raise ValueError("Passive preparation must start at t=0 with zero controls")
        self._initial_model_sha256 = self._effective_model_sha256()

    def _effective_model_sha256(self):
        # MuJoCo's binary serializer includes effective options, parameters,
        # geometry and sensors; hashing only the original file misses edits.
        buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=buffer)
        return hashlib.sha256(buffer).hexdigest()

    def _ensure_model_unchanged(self):
        # The compiled model includes large meshes/textures. Check at persistence
        # boundaries, not every 200 us physical step. A changed preparation must
        # be compiled and identified separately before it can be checkpointed.
        if self._effective_model_sha256() != self._initial_model_sha256:
            self.failed = True
            raise RuntimeError('Effective body model was mutated; create a separately identified preparation')

    def step(self, dt_us, start_us, end_us):
        if self.failed:
            raise RuntimeError("Failed body cannot continue without explicit diagnosis")
        if dt_us != self.dt_us or end_us - start_us != self.dt_us:
            raise ValueError("Physical interval mismatch")
        if start_us != self.steps * self.dt_us:
            raise ValueError("Scheduler and body step counters disagree")
        if abs(self.data.time - start_us / 1e6) > 1e-10:
            raise ValueError("Scheduler and physical clock disagree before step")
        if not np.isfinite(self._integration_state()).all():
            self.failed = True
            raise FloatingPointError("Nonfinite physical input state; refusing native integration without reset/rescue")
        if np.any(self.data.ctrl) or np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError("Unexpected external force/control in passive preparation")
        reject_external_callbacks()
        # No policy, chosen target, implicit reset or actuation is applied here.
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        warnings = self.data.warning.number.copy()
        changed = np.any(warnings != self.warning_count)
        self.warning_count = warnings
        state = self._integration_state()
        if changed or not np.isfinite(state).all() or abs(self.data.time - end_us / 1e6) > 1e-10:
            self.failed = True
            raise FloatingPointError("MuJoCo warning, nonfinite state or clock discontinuity; no reset/rescue")
        if np.any(self.data.qfrc_actuator):
            self.failed = True
            raise RuntimeError("Nonzero actuator force detected in the passive preparation")

    def _integration_state(self):
        state = np.empty(mujoco.mj_stateSize(self.model, self.state_spec))
        mujoco.mj_getState(self.model, self.data, state, self.state_spec)
        return state

    def _build_observation_schema(self):
        model = self.model
        def names(kind, count):
            return [mujoco.mj_id2name(model, kind, i) or f"unnamed_{i}" for i in range(count)]
        joints = []
        qpos_units, qvel_units = [], []
        for i in range(model.njnt):
            kind = int(model.jnt_type[i])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                qu = ["model_length"] * 3 + ["unit_quaternion_wxyz"] * 4
                vu = ["model_length/s"] * 3 + ["rad/s"] * 3
            elif kind == int(mujoco.mjtJoint.mjJNT_BALL):
                qu, vu = ["unit_quaternion_wxyz"] * 4, ["rad/s"] * 3
            elif kind == int(mujoco.mjtJoint.mjJNT_SLIDE):
                qu, vu = ["model_length"], ["model_length/s"]
            else:
                qu, vu = ["rad"], ["rad/s"]
            joints.append(dict(id=i, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                               type=mujoco.mjtJoint(kind).name,
                               qpos_adr=int(model.jnt_qposadr[i]), qpos_size=len(qu),
                               dof_adr=int(model.jnt_dofadr[i]), dof_size=len(vu)))
            qpos_units.extend(qu)
            qvel_units.extend(vu)
        native_units = {
            "mjSENS_FORCE": "model_force",
            "mjSENS_TORQUE": "model_torque",
            "mjSENS_TOUCH": "model_force",
            "mjSENS_FRAMEPOS": "model_length",
            "mjSENS_FRAMELINVEL": "model_length/s",
            "mjSENS_FRAMEANGVEL": "rad/s",
            "mjSENS_GYRO": "rad/s",
            "mjSENS_ACCELEROMETER": "model_length/s^2",
            "mjSENS_FRAMELINACC": "model_length/s^2",
            "mjSENS_FRAMEANGACC": "rad/s^2",
            "mjSENS_FRAMEQUAT": "unit_quaternion_wxyz",
            "mjSENS_FRAMEXAXIS": "dimensionless_unit_vector",
            "mjSENS_FRAMEYAXIS": "dimensionless_unit_vector",
            "mjSENS_FRAMEZAXIS": "dimensionless_unit_vector",
            "mjSENS_ACTUATORFRC": "native_actuator_scalar_force_transmission_dependent",
        }
        sensors = []
        for i in range(model.nsensor):
            sensor_type = mujoco.mjtSensor(int(model.sensor_type[i])).name
            units = native_units.get(sensor_type, "native_sensor_units_not_classified")
            obj_id = int(model.sensor_objid[i])
            if sensor_type in ("mjSENS_JOINTPOS", "mjSENS_JOINTVEL"):
                linear = int(model.jnt_type[obj_id]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
                units = ("model_length" if linear else "rad") + ("/s" if sensor_type == "mjSENS_JOINTVEL" else "")
            sensors.append(dict(id=i, name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i),
                                type=sensor_type, adr=int(model.sensor_adr[i]),
                                dim=int(model.sensor_dim[i]), units=units,
                                object_type=mujoco.mjtObj(int(model.sensor_objtype[i])).name,
                                object_id=obj_id, reference_type=int(model.sensor_reftype[i]),
                                reference_id=int(model.sensor_refid[i])))
        hinge_ids = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
        return dict(schema_version=1, physical_observations_only=True, biological_transduction=False,
                    evaluation="mj_forward on separate MjData at the current integration state",
                    length_unit="model_length; SI conversion is not established by this adapter",
                    force_unit="model_force; SI conversion is not established by this adapter",
                    orientation="MuJoCo world-frame positions/rotation matrices; quaternions wxyz",
                    body_names=names(mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                    geom_names=names(mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
                    site_names=names(mujoco.mjtObj.mjOBJ_SITE, model.nsite),
                    joints=joints, qpos_units=qpos_units, qvel_units=qvel_units,
                    hinge_joint_ids=hinge_ids.tolist(),
                    hinge_joint_names=[joints[int(i)]["name"] for i in hinge_ids],
                    sensors=sensors, contact_geometry_indices="MuJoCo order, never sorted or reordered",
                    contact_wrench_units=["model_force"] * 3 + ["model_torque"] * 3,
                    contact_wrench_frame="native MuJoCo contact frame")

    def observe(self):
        """Current physical observations without changing the integration data.

        mj_setState alone does not refresh geometry or sensor caches. Forward
        dynamics on an independent data object provides fresh values at the
        same instant. No live warm-start, contact ordering, RNG, time or force
        is changed. Results are independent copies, not writable simulator views.
        Native model sensors are not biological sensory transductions.
        """
        if self.failed:
            raise RuntimeError("Failed body cannot supply observations")
        reject_external_callbacks()
        # Stochastic/user sensors or plugins require their own explicit state
        # protocol; the current audited preparation has none of these.
        if self.model.nplugin or np.any(self.model.sensor_noise):
            raise RuntimeError("Observation protocol does not support plugins or sensor noise")
        if np.any(self.model.sensor_type == int(mujoco.mjtSensor.mjSENS_USER)):
            raise RuntimeError("User-defined sensors need an explicit observation protocol")
        before = self._integration_state()
        if not np.isfinite(before).all():
            self.failed = True
            raise FloatingPointError("Nonfinite physical state cannot be observed")
        if abs(self.data.time - self.steps * self.dt_us / 1e6) > 1e-10:
            raise ValueError("Physical observation clock disagrees with step count")
        if np.any(self.data.ctrl) or np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError("Unexpected external force/control during passive observation")
        if self._observation_data is None:
            self._observation_data = mujoco.MjData(self.model)
            self._observation_schema = self._build_observation_schema()
        scratch = self._observation_data
        live_warnings = self.data.warning.number.copy()
        try:
            mujoco.mj_setState(self.model, scratch, before, self.state_spec)
            mujoco.mj_forward(self.model, scratch)
            if np.any(scratch.warning.number):
                self.failed = True
                raise FloatingPointError("Forward observation reported a MuJoCo warning")
            h = np.asarray(self._observation_schema["hinge_joint_ids"], dtype=np.int64)
            contact_geoms = np.asarray([[c.geom1, c.geom2] for c in scratch.contact[:scratch.ncon]], dtype=np.int32).reshape(-1, 2)
            contact_dist = np.asarray([c.dist for c in scratch.contact[:scratch.ncon]], dtype=np.float64)
            contact_pos = np.asarray([c.pos.copy() for c in scratch.contact[:scratch.ncon]], dtype=np.float64).reshape(-1, 3)
            contact_frame = np.asarray([c.frame.copy() for c in scratch.contact[:scratch.ncon]], dtype=np.float64).reshape(-1, 3, 3)
            contact_wrench = np.empty((scratch.ncon, 6))
            for i in range(scratch.ncon):
                mujoco.mj_contactForce(self.model, scratch, i, contact_wrench[i])
            out = dict(time_us=self.steps * self.dt_us, time_seconds=float(scratch.time),
                       qpos=scratch.qpos.copy(), qvel=scratch.qvel.copy(),
                       body_xpos=scratch.xpos.copy(), body_xquat=scratch.xquat.copy(),
                       geom_xpos=scratch.geom_xpos.copy(), geom_xmat=scratch.geom_xmat.reshape(-1, 3, 3).copy(),
                       site_xpos=scratch.site_xpos.copy(), site_xmat=scratch.site_xmat.reshape(-1, 3, 3).copy(),
                       hinge_angle_rad=scratch.qpos[self.model.jnt_qposadr[h]].copy(),
                       hinge_velocity_rad_s=scratch.qvel[self.model.jnt_dofadr[h]].copy(),
                       sensor_values=scratch.sensordata.copy(), contact_geom_ids=contact_geoms,
                       contact_distance=contact_dist, contact_position=contact_pos,
                       contact_frame=contact_frame, contact_wrench=contact_wrench,
                       metadata=copy.deepcopy(self._observation_schema))
            for value in out.values():
                if isinstance(value, np.ndarray):
                    if not np.isfinite(value).all():
                        self.failed = True
                        raise FloatingPointError("Nonfinite derived physical observation")
                    value.setflags(write=False)
            return out
        finally:
            # This guard also executes if extraction or forward dynamics fails.
            if not np.array_equal(before, self._integration_state()) or not np.array_equal(live_warnings, self.data.warning.number):
                self.failed = True
                raise RuntimeError("Observation changed live integration/warnings")

    def state_dict(self):
        # Built-in containers keep runtime checkpoints compatible with safe
        # torch.load(weights_only=True); no arbitrary NumPy pickle constructors.
        self._ensure_model_unchanged()
        return {"identity": self.identity.copy(), "integration": self._integration_state().tolist(),
                "steps": self.steps, "warning_count": self.warning_count.tolist(), "failed": self.failed}

    def load_state_dict(self, payload):
        if self.failed:
            raise RuntimeError("Cannot restore over failed body; preserve failure and construct a new instance")
        self._ensure_model_unchanged()
        reject_external_callbacks()
        if payload["identity"] != self.identity:
            raise ValueError("Body model/configuration mismatch")
        state = np.asarray(payload["integration"], dtype=np.float64)
        if state.shape != (mujoco.mj_stateSize(self.model, self.state_spec),) or not np.isfinite(state).all():
            raise ValueError("Invalid body integration state")
        steps = payload["steps"]
        if not isinstance(steps, int) or steps < 0 or abs(state[0] - steps * self.dt_us / 1e6) > 1e-10:
            raise ValueError("Saved physical time disagrees with saved step count")
        warnings = np.asarray(payload["warning_count"], dtype=np.int64)
        if warnings.shape != self.warning_count.shape or payload["failed"] or np.any(warnings):
            raise ValueError("Refusing to hide a failed numerical state by restoring it")
        mujoco.mj_setState(self.model, self.data, state, self.state_spec)
        if np.any(self.data.ctrl) or np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            self.failed = True
            raise ValueError("Saved passive state contains external control/forces")
        self.steps = steps
        self.warning_count = warnings.copy()
        self.data.warning.number[:] = warnings
        self.failed = False

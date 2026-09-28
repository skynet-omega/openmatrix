"""M1E experimental body: published walking scaffold, never a native VNC claim.

FlyGym 1.2.1 supplies six CPGs, recorded joint trajectories, position servos,
contact correction rules and adhesion. This module only exposes its two
descending channels and fresh physical observations. No goal/reward/heading
policy is passed to the walking controller. SI force calibration is not claimed.
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np
import mujoco


class ExperimentalWalkingBody:
    def __init__(self, yaw=0.0, seed=0, dt=0.0001, scaffold='full'):
        from flygym import Fly
        from flygym.examples.locomotion import HybridTurningController
        from flygym.preprogrammed import default_leg_sensor_placements
        self.dt = float(dt)
        self.fly = Fly(name='matrixfly', spawn_orientation=(0., 0., float(yaw)),
                       enable_adhesion=True, draw_adhesion=False,
                       contact_sensor_placements=default_leg_sensor_placements,
                       head_stabilization_model=None, enable_vision=False,
                       enable_olfaction=False)
        if scaffold not in ('full','no_contact_rules','actuation_disabled'):
            raise ValueError('Unknown motor scaffold intervention')
        self.scaffold = scaffold
        options = {} if scaffold!='no_contact_rules' else {
            'correction_rates':{'retraction':(0,0),'stumbling':(0,0)}}
        self.sim = HybridTurningController(fly=self.fly, cameras=[], timestep=dt,
                                          seed=seed, draw_corrections=False, **options)
        self.sim.reset(seed=seed)
        self.model, self.data = self.sim.physics.model.ptr, self.sim.physics.data.ptr
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        if scaffold=='actuation_disabled':
            self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
        self.scratch = mujoco.MjData(self.model)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state = np.empty(mujoco.mj_stateSize(self.model, self.spec))
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        'matrixfly/Thorax')
        if self.body_id < 0:
            raise ValueError('Thorax body identity missing')
        self.steps = 0
        self.failed = False
        self.max_actuator_force = 0.0
        self.max_contact_count = 0
        from passive_body import reject_external_callbacks
        reject_external_callbacks()

    def advance(self, descending, nsteps):
        descending = np.asarray(descending, dtype=float)
        if self.failed or descending.shape != (2,) or not np.isfinite(descending).all():
            raise ValueError('Invalid command or failed physical body')
        if np.any(descending < 0.4) or np.any(descending > 1.2):
            raise ValueError('Command outside declared source demonstration envelope')
        from passive_body import reject_external_callbacks
        reject_external_callbacks()
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError('Unexpected external force outside declared actuators')
        for _ in range(int(nsteps)):
            _, _, terminated, truncated, _ = self.sim.step(descending)
            self.steps += 1
            self.max_actuator_force = max(self.max_actuator_force, float(np.max(np.abs(self.data.qfrc_actuator))))
            self.max_contact_count = max(self.max_contact_count, self.data.ncon)
            if self.scaffold=='actuation_disabled' and np.any(self.data.qfrc_actuator):
                raise RuntimeError('Disabled motor scaffold still exerts force')
            if (terminated or truncated or np.any(self.data.warning.number)
                    or not np.isfinite(self.data.qpos).all()
                    or not np.isfinite(self.data.qvel).all()
                    or abs(self.data.time-self.steps*self.dt) > 1e-8):
                self.failed = True
                raise FloatingPointError('Physical failure/termination: trajectory retained, no reset or rescue')

    def observe(self):
        """Environment observables; caller must not feed global yaw to its policy.

        Forward on separate MjData avoids stale sensors without altering live
        contacts or integrator warm starts. Yaw increments can provide an ideal
        motion sensor; they are not a reconstruction of fly proprioception.
        """
        if self.failed:
            raise RuntimeError('Failed body')
        mujoco.mj_getState(self.model, self.data, self.state, self.spec)
        mujoco.mj_setState(self.model, self.scratch, self.state, self.spec)
        mujoco.mj_forward(self.model, self.scratch)
        if np.any(self.scratch.warning.number):
            self.failed = True
            raise FloatingPointError('Fresh observation produced MuJoCo warning')
        R = self.scratch.xmat[self.body_id].reshape(3,3)
        return {'time_s':float(self.data.time),
                'yaw':float(np.arctan2(R[1,0], R[0,0])),
                'upright_cos':float(R[2,2]),
                'position_mm':self.scratch.xpos[self.body_id].copy(),
                'qpos':self.data.qpos.copy(), 'qvel':self.data.qvel.copy(),
                'contact_count':int(self.scratch.ncon),
                'max_actuator_force_native':self.max_actuator_force,
                'warning_counts':self.data.warning.number.copy()}

    def close(self):
        self.sim.close()

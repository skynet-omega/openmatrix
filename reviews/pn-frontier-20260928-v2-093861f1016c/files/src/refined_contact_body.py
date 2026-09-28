"""Explicit step refinement of the existing physical model, with exact state adoption."""
import copy
from pathlib import Path
import mujoco
import numpy as np
from contact_complete_native_body import ContactCompleteNativeBody
from native_motor_body import MAX_TORQUE_NATIVE
from passive_body import reject_external_callbacks
from session_io import sha256


class RefinedContactBody(ContactCompleteNativeBody):
    def __init__(self, yaw=0., seed=0, dt=.000025):
        if dt not in (.0001,.00005,.000025):
            raise ValueError('Supported physics refinements: 100, 50, 25 us')
        super().__init__(yaw=yaw,seed=seed,dt=.0001)
        self.dt=float(dt)
        self.constructor=dict(yaw=float(yaw),seed=seed,dt=self.dt)
        self.model.opt.timestep=self.dt
        self.sim.timestep=self.dt
        self.identity=self._identity()

    def _identity(self):
        identity=super()._identity()
        identity['source'][str(Path(__file__).resolve())]=sha256(__file__)
        identity['body_class']='RefinedContactBody'
        identity['observation_schedule']='reference control binding, physics step and bound sensor refresh; unused observation formatting omitted'
        return identity

    def adopt(self, reference):
        state=reference.state_dict()
        if self.steps!=0 or self.data.time!=0 or self.state.shape!=state['integration'].shape:
            raise ValueError('Requires fresh target and identical physical coordinates')
        for field in ('jnt_type','jnt_bodyid','jnt_qposadr','jnt_dofadr','jnt_axis','jnt_pos',
            'body_mass','body_inertia','body_pos','body_quat','jnt_stiffness','dof_damping',
            'geom_size','pair_geom1','pair_geom2','actuator_gear','actuator_forcerange'):
            if not np.array_equal(getattr(self.model,field),getattr(reference.model,field)):
                raise ValueError('Model adoption changes '+field)
        ns=round(reference.data.time*1e9)
        step_ns=round(self.dt*1e9)
        if ns%step_ns:
            raise ValueError('Physical clock cannot be represented exactly')
        mujoco.mj_setState(self.model,self.data,state['integration'],self.spec)
        self.steps=ns//step_ns
        self.sim.curr_time=reference.sim.curr_time
        self.sim.np_random.bit_generator.state=copy.deepcopy(state['sim_rng'])
        for key in ('flexion_sign','angle_derivatives','sign_calibration_qpos','max_actuator_force','max_contact_count'):
            setattr(self,key,copy.deepcopy(state[key]))
        self.replacement=dict(parent_identity=state['identity'],parent_replacement=state['replacement'],
            time_ns=ns,operation='precision-only body replacement',
            integration_transfer='every integration field copied, including warmstart; no state reset',
            dt_s=self.dt,force_coefficients_unchanged=True)

    def advance(self,torques,nsteps):
        torques=np.asarray(torques,dtype=float)
        if self.failed or type(nsteps) is not int or nsteps<0 or torques.shape!=(6,) or not np.isfinite(torques).all() or np.any(np.abs(torques)>MAX_TORQUE_NATIVE+1e-12):
            raise ValueError('Invalid physical input')
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise ValueError('Undeclared external force')
        for _ in range(nsteps):
            reject_external_callbacks()
            self.sim.arena.step(dt=self.dt,physics=self.sim.physics)
            mujoco.mj_forward(self.model,self.data)
            self.fly.pre_step({'joints':torques},self.sim)
            self.sim.physics.step()
            self.sim.curr_time+=self.dt
            self.steps+=1
            # This access has a physical side effect in dm_control: refreshing
            # bound sensordata also refreshes solver state. It must not be
            # replaced by a direct MjData read or silently omitted.
            _=self.sim.physics.bind(self.fly._actuated_joint_sensors).sensordata
            self.max_actuator_force=max(self.max_actuator_force,float(np.max(np.abs(self.data.qfrc_actuator))))
            self.max_contact_count=max(self.max_contact_count,self.data.ncon)
            if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or abs(self.data.time-self.steps*self.dt)>1e-8:
                self.failed=True
                raise FloatingPointError('Physical integration failed without rescue')

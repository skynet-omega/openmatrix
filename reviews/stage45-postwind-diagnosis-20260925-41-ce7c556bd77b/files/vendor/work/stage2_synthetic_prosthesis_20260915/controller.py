"""Resumable synthetic leg-only controller; does not advance physics or CNS.

speed is reference playback rate, not achieved mm/s. steering is differential
left/right playback, not a calibrated yaw-rate command. The finite reference
stops at its end; there is no hidden pose reset or discontinuous cycling.
"""
from pathlib import Path
import hashlib, importlib.util
import numpy as np

_path=Path(__file__).with_name('prototype.py')
_spec=importlib.util.spec_from_file_location('synthetic_prototype_library',_path)
_lib=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_lib)

class Controller:
    SCHEMA='synthetic_leg_reference_controller_v1'
    def __init__(self,model,data,initial_qpos=None):
        self.model=model;self.servo=_lib.LegServo(model,1.,.15,.002)
        self.reference,self.reference_dt,self.reference_roots=_lib.load_reference(model,self.servo)
        self.reference_sha256=hashlib.sha256(self.reference.tobytes()).hexdigest()
        self.initial=self.servo.coordinate(data.qpos if initial_qpos is None else initial_qpos)
        self.phase_s=np.zeros(2);self.age_s=0.;self.speed=0.;self.steering=0.
        self.last_target=self.initial.copy();self.last_actuator=np.zeros(48);self.last_saturated=np.zeros(48,dtype=bool)

    def set_command(self,speed=0.,steering=0.):
        if not np.isfinite([speed,steering]).all() or not 0<=speed<=1 or not -1<=steering<=1:
            raise ValueError('speed must be 0..1 playback rate; steering must be -1..1 differential rate')
        self.speed=float(speed);self.steering=float(steering)

    def torque(self,data,dt):
        """Return native generalized leg force (nv,); first six entries zero.

        Call exactly once before each physical step. Caller owns exclusive force
        replacement, ctrl/xfrc zeroing and mj_step. No qpos writes occur here.
        """
        dt=float(dt)
        if not np.isfinite(dt) or not 0<dt<=.001:raise ValueError('Invalid physical timestep')
        target=np.empty(48);velocity=np.zeros(48)
        rates=self.speed*np.array([1.-.5*self.steering,1.+.5*self.steering])
        last=(len(self.reference)-1)*self.reference_dt
        for side in range(2):
            x=self.phase_s[side]/self.reference_dt;i=min(int(x),len(self.reference)-2);a=min(x-i,1.)
            ids=np.concatenate([np.arange(8*leg,8*(leg+1)) for leg in range(side,6,2)])
            target[ids]=(1-a)*self.reference[i,ids]+a*self.reference[i+1,ids]
            if self.phase_s[side]<last:velocity[ids]=(self.reference[i+1,ids]-self.reference[i,ids])/self.reference_dt*rates[side]
        r=min(self.age_s/.050,1.);r=r*r*(3-2*r)
        target=(1-r)*self.initial+r*target
        if self.age_s<.050:velocity[:]=0
        force,self.last_actuator,self.last_saturated=self.servo.force(data,target,velocity)
        self.last_target=target.copy();self.age_s+=dt
        self.phase_s=np.minimum(last,self.phase_s+dt*rates)
        assert force.shape==(self.model.nv,) and not np.any(force[:6]) and np.isfinite(force).all()
        return force

    def state_dict(self):
        return dict(schema=self.SCHEMA,nq=self.model.nq,nv=self.model.nv,reference_sha256=self.reference_sha256,
                    phase_s=self.phase_s.tolist(),age_s=self.age_s,speed=self.speed,steering=self.steering,initial=self.initial.tolist(),
                    last_target=self.last_target.tolist(),last_actuator=self.last_actuator.tolist(),last_saturated=self.last_saturated.tolist(),
                    parameters=dict(k_scale=1.,torque_limit_native=.15,kd=.002),
                    units=dict(torque_Nm_per_native=1e-7,speed='reference playback fraction',steering='differential reference rate'))

    def load_state_dict(self,state):
        if state['schema']!=self.SCHEMA or state['nq']!=self.model.nq or state['nv']!=self.model.nv or state['reference_sha256']!=self.reference_sha256:
            raise ValueError('Controller schema/model/reference mismatch')
        if state['parameters']!=dict(k_scale=1.,torque_limit_native=.15,kd=.002):raise ValueError('Controller parameters differ')
        self.set_command(state['speed'],state['steering']);self.age_s=float(state['age_s'])
        for name,shape in [('phase_s',(2,)),('initial',(48,)),('last_target',(48,)),('last_actuator',(48,)),('last_saturated',(48,))]:
            arr=np.asarray(state[name],dtype=bool if name=='last_saturated' else float)
            if arr.shape!=shape or not np.isfinite(arr).all():raise ValueError('Invalid controller state '+name)
            setattr(self,name,arr.copy())
        if self.age_s<0 or np.any(self.phase_s<0) or np.any(self.phase_s>(len(self.reference)-1)*self.reference_dt):raise ValueError('Invalid phase or age')
        return self

    @classmethod
    def from_state(cls,model,data,state):return cls(model,data).load_state_dict(state)

"""Explicit ideal motorized rollers at loaded claw contacts.

Normal floor contacts remain real. Tangential passive friction on six claws is
replaced with bounded active horizontal contact forces applied at the feet.
This is synthetic mobility, not gait or biological muscle physiology.
"""
from pathlib import Path
import copy,importlib.util
import numpy as np
import mujoco as mj

D=Path(__file__).resolve().parent;R=D.parents[1]
_spec=importlib.util.spec_from_file_location('contact_device_recovery',R/'work/stage2_synthetic_prosthesis_20260915/controller.py')
_legacy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_legacy)
_lib=_legacy._lib

def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    return copy.deepcopy(value)

class Controller:
    SCHEMA='synthetic_contact_rollers_v1'
    def __init__(self,model,data):
        self.model=model;self.recovery=_legacy.Controller(model,data)
        self.elapsed_ns=0;self.active=False;self.forward_mm_s=0.;self.yaw_rate_rad_s=0.;self.model_patch=None
        self.activation_reason=None;self.activation_elapsed_ns=None;self.activation_body_time_s=None;self.activation_is_probe=False
        self.geom_names=[f'tarsal_claw_T{i}_{side}_collision' for i in (1,2,3) for side in ('left','right')]
        self.geoms=np.array([model.geom(name).id for name in self.geom_names]);self.geom_to_leg={int(g):i for i,g in enumerate(self.geoms)}
        self.floor=model.geom('floor').id;self.thorax=model.body('thorax').id
        self.mass_kg=float(np.sum(model.body_mass)*.001);self.mu=1.;self.rate_per_s=200.
        self.last_pd_qfrc=np.zeros(model.nv);self.last_traction_qfrc=np.zeros(model.nv)
        self.last_force_N=np.zeros((6,3));self.last_normal_N=np.zeros(6);self.last_point_cm=np.zeros((6,3))
        self.last_velocity_m_s=np.zeros((6,3));self.last_target_m_s=np.zeros((6,3));self.last_contact_active=np.zeros(6,dtype=bool)
        self.last_motor_power_W=0.;self.last_body_power_W=0.;self.last_dissipation_W=0.
        self.energy_motor_J=0.;self.energy_body_J=0.;self.energy_dissipation_J=0.
        self.material_slip_um=np.zeros(6);self.contact_chassis_travel_um=np.zeros(6)
        self._jp=np.zeros((3,model.nv));self._jr=np.zeros((3,model.nv));self._f=np.zeros(6)
        self._data=data;self._resume_observation=None

    def _activate(self,reason='default_after_400ms_recovery',probe=False):
        m=self.model;before={n:dict(friction=m.geom_friction[g].tolist(),condim=int(m.geom_condim[g]),priority=int(m.geom_priority[g])) for n,g in zip(self.geom_names,self.geoms)}
        # Higher priority makes these contacts normal-only; passive tangent
        # friction is replaced by the explicit roller law, never double-counted.
        m.geom_condim[self.geoms]=1;m.geom_priority[self.geoms]=1
        after={n:dict(friction=m.geom_friction[g].tolist(),condim=int(m.geom_condim[g]),priority=int(m.geom_priority[g])) for n,g in zip(self.geom_names,self.geoms)}
        self.model_patch=dict(schema='six_ideal_roller_contacts_v1',geoms=self.geom_names,before=before,after=after,
            law=dict(mu=1.,velocity_gain_per_s=200.,force_z_N=0.),scope='six claw contacts only; keep normal contact, geometry, mass, gravity and other geoms',
            root_generalized_force='nonzero J-transpose foot force is physical and recorded; no direct thorax force')
        self.activation_reason=reason;self.activation_elapsed_ns=self.elapsed_ns;self.activation_body_time_s=float(self._data.time);self.activation_is_probe=bool(probe)
        self.model_patch['activation']=dict(reason=reason,elapsed_ns=self.elapsed_ns,body_time_s=float(self._data.time),diagnostic_probe=bool(probe))
        self.active=True

    def activate_traction_for_probe(self,reason):
        """Explicit diagnostic activation; does not claim recovered preparation.

        Applies the same physical patch, without advancing or changing either
        elapsed clock. Production still activates after the default 400 ms.
        """
        if self.active:raise ValueError('Traction already active')
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Probe activation requires a reason')
        self._activate(reason=reason.strip(),probe=True)

    def set_command(self,forward_mm_s=0.,yaw_rate_rad_s=0.):
        if not np.isfinite([forward_mm_s,yaw_rate_rad_s]).all() or abs(forward_mm_s)>.5 or abs(yaw_rate_rad_s)>np.deg2rad(15):
            raise ValueError('Contact-device command outside declared envelope')
        self.forward_mm_s=float(forward_mm_s);self.yaw_rate_rad_s=float(yaw_rate_rad_s)

    def _contacts(self,data):
        result={}
        for ci,c in enumerate(data.contact):
            if c.efc_address<0:continue
            g1,g2=map(int,c.geom)
            g=g2 if g1==self.floor else (g1 if g2==self.floor else -1)
            if g not in self.geom_to_leg:continue
            mj.mj_contactForce(self.model,data,ci,self._f)
            normal=float(self._f[0])*1e-5
            if normal<=0:continue
            leg=self.geom_to_leg[g]
            if leg not in result:result[leg]=dict(normal=0.,point_sum=np.zeros(3),body=int(self.model.geom_bodyid[g]))
            result[leg]['normal']+=normal;result[leg]['point_sum']+=normal*c.pos
        return result

    def _observation(self,data):
        """The last completed step's contact observation, including its geometry.

        mjSTATE_INTEGRATION excludes contact/Jacobian caches. Persisting this
        observation preserves the very next force after a checkpoint restore.
        Force bases are generated with mj_applyFT at each loaded foot point.
        """
        contacts=self._contacts(data);out=[]
        r=np.empty(9);mj.mju_quat2Mat(r,data.qpos[3:7]);forward=r.reshape(3,3)[:,0].copy();forward[2]=0.;forward/=np.linalg.norm(forward)
        for leg,c in contacts.items():
            point=c['point_sum']/c['normal'];bid=c['body']
            mj.mj_jac(self.model,data,self._jp,self._jr,point,bid)
            basis=np.zeros((self.model.nv,2))
            for axis in range(2):
                f=np.zeros(3);f[axis]=1.;q=np.zeros(self.model.nv)
                mj.mj_applyFT(self.model,data,f,np.zeros(3),point,bid,q);basis[:,axis]=q
            out.append(dict(leg=leg,normal_N=c['normal'],point_cm=point,body=bid,velocity_m_s=self._jp@data.qvel*.01,basis_native=basis))
        return dict(time_s=float(data.time),forward=forward,centre_m=data.subtree_com[self.thorax].copy()*.01,contacts=out)

    def torque(self,data,dt):
        dt=float(dt);ns=round(dt*1e9)
        if not np.isfinite(dt) or not 0<dt<=.001 or abs(ns*1e-9-dt)>1e-15:raise ValueError('Invalid physical timestep')
        self._data=data
        if not self.active and self.elapsed_ns>=400000000:self._activate()
        # Original recovery controller remains a fixed articular support target.
        self.recovery.set_command(0.,0.);self.last_pd_qfrc=self.recovery.torque(data,dt)
        self.last_traction_qfrc[:]=0;self.last_force_N[:]=0;self.last_normal_N[:]=0;self.last_point_cm[:]=0
        self.last_velocity_m_s[:]=0;self.last_target_m_s[:]=0;self.last_contact_active[:]=False
        self.last_motor_power_W=0.;self.last_body_power_W=0.;self.last_dissipation_W=0.
        if self.active:
            obs=self._resume_observation if self._resume_observation is not None else self._observation(data)
            if abs(obs['time_s']-data.time)>1e-12:raise ValueError('Retained contact observation time differs')
            self._resume_observation=None;contacts=obs['contacts'];n=len(contacts)
            if n:
                k=self.mass_kg*self.rate_per_s/n
                v_world=np.asarray(obs['forward'])*self.forward_mm_s*.001;centre=np.asarray(obs['centre_m'])
                for c in contacts:
                    leg=c['leg'];point=np.asarray(c['point_cm']);normal=c['normal_N'];actual=np.asarray(c['velocity_m_s'])
                    target=v_world+self.yaw_rate_rad_s*np.cross(np.array([0.,0.,1.]),point*.01-centre)
                    error=target[:2]-actual[:2];force=np.zeros(3);force[:2]=k*error
                    norm=float(np.linalg.norm(force));cap=self.mu*normal
                    if norm>cap:force*=cap/norm
                    assert force[2]==0. and np.linalg.norm(force)<=cap+1e-20
                    # Linear combination of the two mj_applyFT foot-force bases.
                    self.last_traction_qfrc+=np.asarray(c['basis_native'])@(force[:2]/1e-5)
                    pm=float(force@target);pb=float(force@actual);diss=pm-pb
                    if diss < -1e-24:raise FloatingPointError('Negative roller dissipation')
                    self.last_motor_power_W+=pm;self.last_body_power_W+=pb;self.last_dissipation_W+=diss
                    self.last_force_N[leg]=force;self.last_normal_N[leg]=normal;self.last_point_cm[leg]=point
                    self.last_velocity_m_s[leg]=actual;self.last_target_m_s[leg]=target;self.last_contact_active[leg]=True
                    self.material_slip_um[leg]+=np.linalg.norm(error)*dt*1e6
                    self.contact_chassis_travel_um[leg]+=np.linalg.norm(actual[:2])*dt*1e6
        self.energy_motor_J+=self.last_motor_power_W*dt;self.energy_body_J+=self.last_body_power_W*dt;self.energy_dissipation_J+=self.last_dissipation_W*dt
        self.elapsed_ns+=ns
        out=self.last_pd_qfrc+self.last_traction_qfrc
        assert np.isfinite(out).all() and not np.any(self.last_pd_qfrc[:6])
        assert not np.any(self.last_force_N[~self.last_contact_active])
        return out

    def state_dict(self):
        names=['elapsed_ns','active','forward_mm_s','yaw_rate_rad_s','model_patch','activation_reason','activation_elapsed_ns','activation_body_time_s','activation_is_probe','last_pd_qfrc','last_traction_qfrc','last_force_N','last_normal_N','last_point_cm',
            'last_velocity_m_s','last_target_m_s','last_contact_active','last_motor_power_W','last_body_power_W','last_dissipation_W','energy_motor_J','energy_body_J','energy_dissipation_J',
            'material_slip_um','contact_chassis_travel_um']
        observation=(self._resume_observation if self._resume_observation is not None else self._observation(self._data)) if self.active or self.elapsed_ns>=400000000 else None
        return dict(schema=self.SCHEMA,nq=self.model.nq,nv=self.model.nv,mass_kg=self.mass_kg,
            parameters=dict(mu=self.mu,velocity_gain_per_s=self.rate_per_s,recovery_ns=400000000),
            state={n:(getattr(self,n).tolist() if isinstance(getattr(self,n),np.ndarray) else copy.deepcopy(getattr(self,n))) for n in names},recovery=self.recovery.state_dict(),
            retained_contact_observation=_plain(observation),
            units=dict(force_N_per_native=1e-5,torque_Nm_per_native=1e-7,length_m_per_native=.01),device='ideal motorized rollers, not six-leg gait')

    @classmethod
    def from_state(cls,model,data,state):
        if state['schema']!=cls.SCHEMA or state['nq']!=model.nq or state['nv']!=model.nv or state['parameters']!=dict(mu=1.,velocity_gain_per_s=200.,recovery_ns=400000000):
            raise ValueError('Contact-controller schema/model/parameters mismatch')
        obj=cls(model,data)
        if abs(obj.mass_kg-state['mass_kg'])>1e-18:raise ValueError('Contact-controller mass mismatch')
        if state['state']['active']:obj._activate()
        for name,value in state['state'].items():
            current=getattr(obj,name)
            if isinstance(current,np.ndarray):
                value=np.asarray(value,dtype=bool if name=='last_contact_active' else float)
                if value.shape!=current.shape or not np.isfinite(value).all():raise ValueError('Invalid contact-controller '+name)
                value=value.copy()
            setattr(obj,name,copy.deepcopy(value))
        obj.recovery=_legacy.Controller.from_state(model,data,state['recovery']);obj.set_command(obj.forward_mm_s,obj.yaw_rate_rad_s)
        obj._resume_observation=copy.deepcopy(state['retained_contact_observation'])
        return obj

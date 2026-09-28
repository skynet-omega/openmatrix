"""Reusable active tendon operator for a declared geometry/model prior.

Q=-J.T F and power=-F.T ldot use the same geometry evaluation.  Geometry is
MuJoCo-native cm, while force and power are SI. Free coordinates have mixed
conjugate units; do not label the entire Q vector as a vector of joint torques.
This experimental bank does not advance a body or write a CNS command.
"""
import numpy as np
import mujoco as mj
from coxal_body import tendon_row


class TendonBank:
    def __init__(self,model,rows,prefix='hypothesis_'):
        self.model=model;self.data=mj.MjData(model)
        self.names=tuple(r['name'] for r in rows)
        self.tids=np.array([model.tendon(prefix+n).id for n in self.names],dtype=int)
        self.gains=np.array([r['gainprm'] for r in rows],dtype=float)
        self.dynamics=np.array([r['dynprm_s'] for r in rows],dtype=float)
        self.ranges=np.array([r['lengthrange_mm'] for r in rows],dtype=float)
        for x in [self.tids,self.gains,self.dynamics,self.ranges]:x.setflags(write=False)
        self.activation=np.zeros(len(rows));self.time_ns=0

    def geometry(self,qpos,qvel):
        d=self.data;m=self.model
        d.qpos[:]=qpos;d.qvel[:]=qvel;mj.mj_forward(m,d)
        jac=np.array([tendon_row(m,d,i) for i in self.tids])
        length=d.ten_length[self.tids].copy()*10
        velocity=(jac@qvel)*10
        if not np.isfinite(jac).all() or not np.all(length>0):raise ValueError('Invalid tendon geometry')
        return length,velocity,jac

    def step(self,command,qpos,qvel,dt_ns):
        u=np.asarray(command,dtype=float);a=self.activation;dt=dt_ns*1e-9
        if type(dt_ns) is not int or dt_ns<=0 or u.shape!=a.shape or not np.isfinite(u).all() or np.any((u<0)|(u>1)):
            raise ValueError('Invalid activation command/step')
        def f(v):return np.array([mj.mju_muscleDynamics(float(x),float(y),p) for x,y,p in zip(u,v,self.dynamics)])
        k1=f(a);k2=f(a+.5*dt*k1);k3=f(a+.5*dt*k2);k4=f(a+dt*k3)
        anew=a+dt*(k1+2*k2+2*k3+k4)/6
        if not np.isfinite(anew).all() or np.any((anew<0)|(anew>1)):raise FloatingPointError('Activation out of range')
        l,v,j=self.geometry(qpos,qvel)
        force=-np.array([mj.mju_muscleGain(x,y,r,1.,p)*ac for x,y,r,p,ac in zip(l,v,self.ranges,self.gains,(a+anew)*.5)])*1e-6
        if not np.isfinite(force).all() or np.any(force<0):raise FloatingPointError('Invalid muscle tension')
        generalized=-(j*.01).T@force
        power_by_muscle=-force*v*.001
        self.activation=anew;self.time_ns+=dt_ns
        return dict(force_N=force,generalized_SI=generalized,power_W=float(generalized@qvel),
                    power_by_muscle_W=power_by_muscle,length_mm=l,velocity_mm_s=v,jacobian_cm=j)

    def state_dict(self):
        return dict(names=self.names,activation=self.activation.copy(),time_ns=self.time_ns)

    def restore(self,state):
        a=np.asarray(state['activation'])
        if set(state)!={'names','activation','time_ns'} or tuple(state['names'])!=self.names or a.shape!=self.activation.shape or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any((a<0)|(a>1)) or type(state['time_ns']) is not int or state['time_ns']<0:
            raise ValueError('Invalid tendon bank state')
        self.activation=a.copy();self.time_ns=state['time_ns']

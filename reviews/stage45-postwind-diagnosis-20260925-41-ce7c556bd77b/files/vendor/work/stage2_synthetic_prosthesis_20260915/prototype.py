"""Literal LegServo donor; exact saved reference replaces HDF5 I/O."""
from pathlib import Path
import numpy as np
LEGS=[f"T{i}_{side}" for i in (1,2,3) for side in ("left","right")]

class LegServo:
    def __init__(self,m,k_scale=1.,limit=.15,kd=.002):
        self.m=m;self.names=[];rows=[];self.k=[];self.bounds=[]
        for leg in LEGS:
            for joint in ('coxa_abduct','coxa_twist','coxa','femur_twist','femur','tibia','tarsus','tarsus2'):
                name=f'{joint}_{leg}';row=np.zeros(m.nv)
                if joint=='tarsus2':
                    for part,weight in [(2,1.),(3,.5),(4,.5),(5,.5)]:
                        row[m.joint(f'tarsus{part}_{leg}').dofadr[0]]=weight
                    self.bounds.append((-.9,.9))
                else:
                    j=m.joint(name);row[j.dofadr[0]]=1.;self.bounds.append(tuple(j.range))
                rows.append(row);self.names.append(name);self.k.append((.8 if joint.startswith(('coxa','femur')) else .4)*k_scale)
        self.J=np.array(rows);self.k=np.array(self.k);self.bounds=np.array(self.bounds)
        self.qadr=np.array([m.jnt_qposadr[m.dof_jntid[v]] for v in range(6,m.nv)])
        self.limit=float(limit);self.kd=float(kd)
        assert not np.any(self.J[:,:6])
        self.allowed=np.any(self.J!=0,axis=0)

    def coordinate(self,qpos):return self.J[:,6:]@qpos[self.qadr]

    def force(self,d,target,target_v=None):
        target=np.clip(target,self.bounds[:,0],self.bounds[:,1])
        v=np.zeros(48) if target_v is None else target_v
        raw=self.k*(target-self.coordinate(d.qpos))-self.kd*(self.J@d.qvel-v)
        actuator=np.clip(raw,-self.limit,self.limit)
        return self.J.T@actuator,actuator,np.abs(raw)>self.limit

def load_reference(model,servo):
    with np.load(Path(__file__).resolve().parents[3]/"inputs/reference.npz",allow_pickle=False) as z:
        return z["reference"].copy(),float(z["dt"]),z["roots"].copy()

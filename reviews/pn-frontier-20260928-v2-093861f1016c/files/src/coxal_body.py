"""Seven provisional LF coxal active muscle paths on the existing FlyBody.

Joint passivity and the twelve tibial elements remain explicit approximations.
No extra passive muscle bias, force fitting, learned policy, or new body pose.
"""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from flybody_cns_body import CNSFlyBody
from flybody_torque_port import ASSETS, MAX_TORQUE_NATIVE, FLYGYM_TORQUE_TO_NM, FLYBODY_TORQUE_TO_NM
from passive_body import reject_external_callbacks
from session_io import sha256

ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/coxal_geometry_20260913/prior.json'
PRIOR_SHA256='7d0f73a8887eb992a32616f9f996ec6c47afffd5c8651068c4f9274a528d9bdd'


def tendon_row(model,data,index):
    if not mj.mj_isSparse(model):return data.ten_J[index].copy()
    start=int(data.ten_J_rowadr[index]);end=start+int(data.ten_J_rownnz[index])
    row=np.zeros(model.nv)
    row[data.ten_J_colind.ravel()[start:end]]=data.ten_J.ravel()[start:end]
    return row


class CoxalBody(CNSFlyBody):
    SCHEMA='matrix_flybody_coxal_body_v1'

    def __init__(self):
        super().__init__()
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed coxal prior')
        self.prior=json.loads(PRIOR.read_text())
        rows=self.prior['muscles'];self.coxal_names=[r['name'] for r in rows]
        self.coxal_ids=[r['canonical_ids'] for r in rows]
        self.gain=np.array([r['source_gainprm'] for r in rows])
        self.dynamics=np.array([r['source_dynprm_s'] for r in rows])
        self.lengthrange=np.array([r['source_lengthrange_mm'] for r in rows])
        root=ET.parse(ASSETS/'fruitfly.xml').getroot()
        root.find('compiler').set('meshdir',str(ASSETS));root.remove(root.find('actuator'))
        tendons=root.find('tendon')
        if tendons is None:tendons=ET.SubElement(root,'tendon')
        for row in rows:
            tendon=ET.SubElement(tendons,'spatial',name='matrix_'+row['name'])
            for index,site in enumerate(row['sites']):
                name='matrix_'+row['name']+'_'+str(index)
                owner=next(b for b in root.iter('body') if b.get('name')==site['body'])
                ET.SubElement(owner,'site',name=name,pos=' '.join(format(x,'.17g') for x in site['position_cm']))
                ET.SubElement(tendon,'site',site=name)
        self.geometry=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.gdata=mj.MjData(self.geometry)
        if self.geometry.nv!=self.model.nv or self.geometry.nq!=self.model.nq:
            raise ValueError('Coxal geometry changes generalized coordinates')
        for index in range(self.model.njnt):
            if mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,index)!=mj.mj_id2name(self.geometry,mj.mjtObj.mjOBJ_JOINT,index):
                raise ValueError('Coxal joint order differs')
        self.tendon_ids=[mj.mj_name2id(self.geometry,mj.mjtObj.mjOBJ_TENDON,'matrix_'+n) for n in self.coxal_names]
        self.activation=np.zeros(7);self.pending_coxal=np.zeros(7)
        self.last_tension_N=np.zeros(7);self.last_coxal_torque_Nm=np.zeros(self.model.nv)
        self.coxal_time_ns=0;self.coxal_origin_ns=0
        self.identity.update(coxal_source_sha256=sha256(__file__),coxal_prior_sha256=PRIOR_SHA256,
            coxal_force_scope='Active Hill only; original joint passivity retained, biological force not calibrated',
            coxal_geometry_scope=self.prior['geometry_scope'])

    @classmethod
    def from_parent(cls,parent,command):
        if type(parent) is not CNSFlyBody:raise ValueError('Expected unchanged FlyBody189 body')
        obj=cls();state=parent.state_dict()
        mj.mj_setState(obj.model,obj.data,state['integration'],obj.spec)
        obj.steps=parent.steps;obj.origin_ns=parent.origin_ns
        obj.coxal_time_ns=obj.steps*round(obj.dt*1e9);obj.coxal_origin_ns=obj.coxal_time_ns
        obj.pending_coxal=np.asarray(command,dtype=float).copy();obj.validate_coxa()
        np.testing.assert_array_equal(obj.integration_state(),parent.integration_state())
        return obj

    def validate_coxa(self):
        for a,shape in [(self.activation,(7,)),(self.pending_coxal,(7,)),(self.last_tension_N,(7,)),(self.last_coxal_torque_Nm,(self.model.nv,))]:
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid coxal state')
        if (np.any(self.activation<0) or np.any(self.activation>1) or np.any(self.pending_coxal<0)
                or np.any(self.pending_coxal>1) or np.any(self.last_tension_N<0)
                or type(self.coxal_time_ns) is not int or type(self.coxal_origin_ns) is not int
                or not 0<=self.coxal_origin_ns<=self.coxal_time_ns
                or self.coxal_time_ns!=self.steps*round(self.dt*1e9)):
            raise ValueError('Coxal clock or normalized input invalid')
        if not np.array_equal(self.data.qfrc_applied,self.last_coxal_torque_Nm/FLYBODY_TORQUE_TO_NM):
            raise ValueError('External torque differs from the declared muscles')

    def _muscle_step(self):
        dt=self.dt;a=self.activation;u=self.pending_coxal
        def f(act):return np.array([mj.mju_muscleDynamics(float(x),float(y),p) for x,y,p in zip(u,act,self.dynamics)])
        k1=f(a);k2=f(a+.5*dt*k1);k3=f(a+.5*dt*k2);k4=f(a+dt*k3)
        anew=a+dt*(k1+2*k2+2*k3+k4)/6;amid=.5*(a+anew)
        if np.any(anew<0) or np.any(anew>1):raise FloatingPointError('Coxal activation left its bounds')
        self.gdata.qpos[:]=self.data.qpos;self.gdata.qvel[:]=self.data.qvel
        mj.mj_forward(self.geometry,self.gdata)
        J=np.array([tendon_row(self.geometry,self.gdata,i) for i in self.tendon_ids])
        length=self.gdata.ten_length[self.tendon_ids]*10.;velocity=J@self.data.qvel*10.
        tension=-np.array([mj.mju_muscleGain(l,v,r,1.,p)*ac for l,v,r,p,ac in zip(length,velocity,self.lengthrange,self.gain,amid)])*1e-6
        if not np.isfinite(tension).all() or np.any(tension<0):raise FloatingPointError('Invalid coxal tension')
        self.activation=anew;self.last_tension_N=tension
        self.last_coxal_torque_Nm=-(J*.01).T@tension

    def advance(self,torque_native,nsteps=1):
        torque=np.asarray(torque_native,dtype=float)*FLYGYM_TORQUE_TO_NM
        limit=MAX_TORQUE_NATIVE*FLYGYM_TORQUE_TO_NM
        if torque.shape!=(6,) or not np.isfinite(torque).all() or np.any(abs(torque)>limit*(1+1e-12)) or type(nsteps) is not int or nsteps<0:
            raise ValueError('Invalid tibial torque')
        self.validate_coxa()
        if np.any(self.data.xfrc_applied):raise ValueError('Undeclared body force')
        for _ in range(nsteps):
            reject_external_callbacks();self._muscle_step()
            self.data.qfrc_applied[:]=self.last_coxal_torque_Nm/FLYBODY_TORQUE_TO_NM
            if not np.any(self.last_coxal_torque_Nm):
                super().advance_joint_torque_si(torque,1)
            else:
                self.data.ctrl[:]=torque/FLYBODY_TORQUE_TO_NM
                mj.mj_step(self.model,self.data);self.steps+=1
                if (np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all()
                        or not np.isfinite(self.data.qvel).all() or abs(self.data.time-self.steps*self.dt)>1e-10):
                    raise FloatingPointError('Coxal physical step failed')
            self.coxal_time_ns+=round(self.dt*1e9)

    def state_dict(self):
        self.validate_coxa();out=super().state_dict();out['schema']=self.SCHEMA
        out['coxa']=dict(activation=self.activation.copy(),pending=self.pending_coxal.copy(),last_tension_N=self.last_tension_N.copy(),
            last_torque_Nm=self.last_coxal_torque_Nm.copy(),time_ns=self.coxal_time_ns,origin_ns=self.coxal_origin_ns)
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:
            raise ValueError('Wrong coxal body or identity')
        if (type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0
                or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9)
                or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all()):
            raise ValueError('Wrong coxal body clock/state')
        c=state['coxa']
        if set(c)!={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}:raise ValueError('Incomplete muscle state')
        mj.mj_setState(obj.model,obj.data,state['integration'],obj.spec);obj.steps=state['steps'];obj.origin_ns=state['origin_ns']
        obj.activation=c['activation'].copy();obj.pending_coxal=c['pending'].copy();obj.last_tension_N=c['last_tension_N'].copy()
        obj.last_coxal_torque_Nm=c['last_torque_Nm'].copy();obj.coxal_time_ns=c['time_ns'];obj.coxal_origin_ns=c['origin_ns']
        obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:raise ValueError('Coxal physical time differs')
        return obj

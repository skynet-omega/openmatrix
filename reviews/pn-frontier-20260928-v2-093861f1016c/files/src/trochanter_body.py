"""Add six LF CTr active paths without resetting the seven coxal muscles.

Separate torque sums preserve the parent's floating-point evaluation for the
zero-new-input control. Geometry/kinetics are declared source model priors.
"""
from pathlib import Path
import json,xml.etree.ElementTree as ET
import numpy as np,mujoco as mj
from coxal_body import CoxalBody,tendon_row
from flybody_cns_body import CNSFlyBody
from flybody_torque_port import ASSETS,MAX_TORQUE_NATIVE,FLYGYM_TORQUE_TO_NM,FLYBODY_TORQUE_TO_NM
from passive_body import reject_external_callbacks
from session_io import sha256
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/trochanter_geometry_20260913/prior.json'
PRIOR_SHA256='4a8e8653eec7dcdf205c99aca9c42944961b8b4cf440e16baef9ab1bcb882eb7'

class TrochanterBody(CoxalBody):
    SCHEMA='matrix_flybody_trochanter_body_v1'

    def __init__(self):
        super().__init__()
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed CTr prior')
        self.tr_prior=json.loads(PRIOR.read_text());rows=self.tr_prior['muscles']
        self.tr_names=[r['name'] for r in rows];self.tr_ids=[r['canonical_ids'] for r in rows]
        self.tr_gain=np.array([r['source_gainprm'] for r in rows]);self.tr_dynamics=np.array([r['source_dynprm_s'] for r in rows]);self.tr_lengthrange=np.array([r['source_lengthrange_mm'] for r in rows])
        root=ET.parse(ASSETS/'fruitfly.xml').getroot();root.find('compiler').set('meshdir',str(ASSETS));root.remove(root.find('actuator'));tendons=root.find('tendon')
        if tendons is None:tendons=ET.SubElement(root,'tendon')
        for row in rows:
            tendon=ET.SubElement(tendons,'spatial',name='matrix_'+row['name'])
            for i,site in enumerate(row['sites']):
                name='matrix_'+row['name']+'_'+str(i);owner=next(b for b in root.iter('body') if b.get('name')==site['body'])
                ET.SubElement(owner,'site',name=name,pos=' '.join(format(x,'.17g') for x in site['position_cm']));ET.SubElement(tendon,'site',site=name)
        self.tr_geometry=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));self.tr_data=mj.MjData(self.tr_geometry)
        if self.tr_geometry.nv!=self.model.nv or self.tr_geometry.nq!=self.model.nq:raise ValueError('CTr coordinates changed')
        for i in range(self.model.njnt):
            if mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,i)!=mj.mj_id2name(self.tr_geometry,mj.mjtObj.mjOBJ_JOINT,i):raise ValueError('CTr joint order differs')
        self.tr_tendon_ids=[mj.mj_name2id(self.tr_geometry,mj.mjtObj.mjOBJ_TENDON,'matrix_'+n) for n in self.tr_names]
        self.tr_activation=np.zeros(6);self.pending_tr=np.zeros(6);self.last_tr_tension_N=np.zeros(6);self.last_tr_torque_Nm=np.zeros(self.model.nv);self.tr_time_ns=0;self.tr_origin_ns=0
        self.identity.update(trochanter_source_sha256=sha256(__file__),trochanter_prior_sha256=PRIOR_SHA256,trochanter_scope=self.tr_prior['geometry_scope'])

    def _restore_inherited(self,state):
        mj.mj_setState(self.model,self.data,state['integration'],self.spec);self.steps=state['steps'];self.origin_ns=state['origin_ns'];c=state['coxa']
        self.activation=c['activation'].copy();self.pending_coxal=c['pending'].copy();self.last_tension_N=c['last_tension_N'].copy();self.last_coxal_torque_Nm=c['last_torque_Nm'].copy();self.coxal_time_ns=c['time_ns'];self.coxal_origin_ns=c['origin_ns']

    @classmethod
    def from_parent(cls,parent,command):
        if type(parent) is not CoxalBody:raise ValueError('Expected continuing192 coxal body')
        state=parent.state_dict();obj=cls();obj._restore_inherited(state)
        obj.pending_tr=np.asarray(command,dtype=float).copy();obj.tr_time_ns=obj.coxal_time_ns;obj.tr_origin_ns=obj.tr_time_ns;obj.validate_coxa()
        np.testing.assert_array_equal(obj.integration_state(),parent.integration_state())
        return obj

    def validate_coxa(self):
        # Same inherited checks, with the sum of the two declared active ports.
        for a,n in [(self.activation,7),(self.pending_coxal,7),(self.last_tension_N,7),(self.last_coxal_torque_Nm,self.model.nv),
                    (self.tr_activation,6),(self.pending_tr,6),(self.last_tr_tension_N,6),(self.last_tr_torque_Nm,self.model.nv)]:
            if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError('Invalid CTr/coxal state')
        for a in [self.activation,self.pending_coxal,self.tr_activation,self.pending_tr]:
            if np.any(a<0) or np.any(a>1):raise ValueError('Invalid normalized muscle input/activation')
        if np.any(self.last_tension_N<0) or np.any(self.last_tr_tension_N<0):raise ValueError('Negative muscle tension')
        for t,o in [(self.coxal_time_ns,self.coxal_origin_ns),(self.tr_time_ns,self.tr_origin_ns)]:
            if type(t) is not int or type(o) is not int or not 0<=o<=t or t!=self.steps*round(self.dt*1e9):raise ValueError('Invalid muscle clock')
        torque=self.last_coxal_torque_Nm+self.last_tr_torque_Nm
        if not np.array_equal(self.data.qfrc_applied,torque/FLYBODY_TORQUE_TO_NM):raise ValueError('Undeclared external torque')

    def _tr_step(self):
        dt=self.dt;a=self.tr_activation;u=self.pending_tr
        def f(v):return np.array([mj.mju_muscleDynamics(float(x),float(y),p) for x,y,p in zip(u,v,self.tr_dynamics)])
        k1=f(a);k2=f(a+.5*dt*k1);k3=f(a+.5*dt*k2);k4=f(a+dt*k3);anew=a+dt*(k1+2*k2+2*k3+k4)/6;amid=.5*(a+anew)
        if np.any(anew<0) or np.any(anew>1):raise FloatingPointError('CTr activation out of bounds')
        d=self.tr_data;m=self.tr_geometry;d.qpos[:]=self.data.qpos;d.qvel[:]=self.data.qvel;mj.mj_forward(m,d)
        J=np.array([tendon_row(m,d,i) for i in self.tr_tendon_ids]);length=d.ten_length[self.tr_tendon_ids]*10.;velocity=J@self.data.qvel*10.
        tension=-np.array([mj.mju_muscleGain(l,v,r,1.,p)*ac for l,v,r,p,ac in zip(length,velocity,self.tr_lengthrange,self.tr_gain,amid)])*1e-6
        if not np.isfinite(tension).all() or np.any(tension<0):raise FloatingPointError('Invalid CTr tension')
        self.tr_activation=anew;self.last_tr_tension_N=tension;self.last_tr_torque_Nm=-(J*.01).T@tension

    def advance(self,torque_native,nsteps=1):
        torque=np.asarray(torque_native,dtype=float)*FLYGYM_TORQUE_TO_NM;limit=MAX_TORQUE_NATIVE*FLYGYM_TORQUE_TO_NM
        if torque.shape!=(6,) or not np.isfinite(torque).all() or np.any(abs(torque)>limit*(1+1e-12)) or type(nsteps) is not int or nsteps<0:raise ValueError('Invalid tibial torque')
        self.validate_coxa()
        if np.any(self.data.xfrc_applied):raise ValueError('Undeclared body force')
        for _ in range(nsteps):
            reject_external_callbacks();self._muscle_step();self._tr_step();combined=self.last_coxal_torque_Nm+self.last_tr_torque_Nm
            self.data.qfrc_applied[:]=combined/FLYBODY_TORQUE_TO_NM
            if not np.any(combined):self.advance_joint_torque_si(torque,1)
            else:
                self.data.ctrl[:]=torque/FLYBODY_TORQUE_TO_NM;mj.mj_step(self.model,self.data);self.steps+=1
                if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or abs(self.data.time-self.steps*self.dt)>1e-10:raise FloatingPointError('CTr physical step failed')
            self.coxal_time_ns+=round(self.dt*1e9);self.tr_time_ns+=round(self.dt*1e9)

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA
        out['trochanter']=dict(activation=self.tr_activation.copy(),pending=self.pending_tr.copy(),last_tension_N=self.last_tr_tension_N.copy(),last_torque_Nm=self.last_tr_torque_Nm.copy(),time_ns=self.tr_time_ns,origin_ns=self.tr_origin_ns)
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa','trochanter'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:raise ValueError('Wrong CTr identity')
        if type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0 or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9) or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all():raise ValueError('Invalid body clock/state')
        fields={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}
        if set(state['coxa'])!=fields or set(state['trochanter'])!=fields:raise ValueError('Incomplete muscle state')
        obj._restore_inherited(state);t=state['trochanter'];obj.tr_activation=t['activation'].copy();obj.pending_tr=t['pending'].copy();obj.last_tr_tension_N=t['last_tension_N'].copy();obj.last_tr_torque_Nm=t['last_torque_Nm'].copy();obj.tr_time_ns=t['time_ns'];obj.tr_origin_ns=t['origin_ns'];obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:raise ValueError('Physical clock differs')
        return obj

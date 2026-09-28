"""Declared serial CTr homology, independent from the frozen frontal body.

Geometry and Hill curves are transferred model priors, not measured posterior
motor units. No added passive bias, prescribed movement or root actuation.
"""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from active_tendon_bank import TendonBank
from rf_tarsal_body import RFTarsalBody
from flybody_torque_port import ASSETS,MAX_TORQUE_NATIVE,FLYGYM_TORQUE_TO_NM,FLYBODY_TORQUE_TO_NM
from passive_body import reject_external_callbacks
from session_io import sha256
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/serial_ctr_homology_20260914/prior.json'
PRIOR_SHA256='07fbb064d58e34d468c687fa2ce8dfa9a760b6741d25d2fcb7a7f7b277b66220'


class SerialCTrBody(RFTarsalBody):
    SCHEMA='matrix_flybody_serial_ctr_body_v1'

    def __init__(self):
        super().__init__();self._build_serial()

    def _build_serial(self):
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed serial CTr prior')
        self.serial_prior=json.loads(PRIOR.read_text());rows=self.serial_prior['muscles']
        self.serial_ids=[r['canonical_ids'] for r in rows]
        root=ET.parse(ASSETS/'fruitfly.xml').getroot();root.find('compiler').set('meshdir',str(ASSETS));root.remove(root.find('actuator'))
        for row in rows:
            ten=ET.SubElement(root.find('tendon'),'spatial',name='hypothesis_'+row['name'])
            for i,s in enumerate(row['sites']):
                name='hypothesis_'+row['name']+'_'+str(i);owner=next(b for b in root.iter('body') if b.get('name')==s['body'])
                ET.SubElement(owner,'site',name=name,pos=' '.join(format(x,'.17g') for x in s['position_cm']));ET.SubElement(ten,'site',site=name)
        gm=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        if (gm.nq,gm.nv)!=(self.model.nq,self.model.nv) or [gm.joint(i).name for i in range(gm.njnt)]!=[self.model.joint(i).name for i in range(self.model.njnt)]:raise ValueError('Serial coordinates changed')
        self.serial_bank=TendonBank(gm,rows);self.pending_serial=np.zeros(len(rows))
        self.last_serial_force_N=np.zeros(len(rows));self.last_serial_generalized_SI=np.zeros(self.model.nv)
        self.serial_origin_ns=self.steps*round(self.dt*1e9);self.serial_bank.time_ns=self.serial_origin_ns
        self.identity=copy.deepcopy(self.identity)
        self.identity.update(serial_ctr_source_sha256=sha256(__file__),serial_ctr_prior_sha256=PRIOR_SHA256,
                             active_tendon_bank_sha256=sha256(ROOT/'src/active_tendon_bank.py'))

    @classmethod
    def from_parent(cls,parent):
        if type(parent) is not RFTarsalBody:raise ValueError('Expected continuing208 physical body')
        parent.validate_coxa();before=parent.integration_state().copy()
        obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__);obj._build_serial();obj.validate_coxa()
        np.testing.assert_array_equal(before,obj.integration_state());return obj

    def _inherited_generalized(self):
        result=self.last_coxal_torque_Nm+self.last_tr_torque_Nm+self.last_rf_torque_Nm
        result[self.tarsal_vadr]+=self.last_tarsal_torque_Nm
        return result

    def validate_coxa(self):
        # Retain all frozen201 field/clock checks, changing only the declared sum.
        for activation,pending,tension,torque,n,clock,origin in [
            (self.activation,self.pending_coxal,self.last_tension_N,self.last_coxal_torque_Nm,7,self.coxal_time_ns,self.coxal_origin_ns),
            (self.tr_activation,self.pending_tr,self.last_tr_tension_N,self.last_tr_torque_Nm,6,self.tr_time_ns,self.tr_origin_ns),
            (self.rf_activation,self.pending_rf,self.last_rf_tension_N,self.last_rf_torque_Nm,13,self.rf_time_ns,self.rf_origin_ns),
            (self.serial_bank.activation,self.pending_serial,self.last_serial_force_N,self.last_serial_generalized_SI,24,self.serial_bank.time_ns,self.serial_origin_ns)]:
            for value,shape in [(activation,(n,)),(pending,(n,)),(tension,(n,)),(torque,(self.model.nv,))]:
                if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid serial muscle state')
            if np.any((activation<0)|(activation>1)) or np.any((pending<0)|(pending>1)) or np.any(tension<0):raise ValueError('Invalid serial activation or tension')
            if type(clock) is not int or type(origin) is not int or not 0<=origin<=clock or clock!=self.steps*round(self.dt*1e9):raise ValueError('Invalid serial muscle clock')
        for x in [self.tarsal_activation,self.pending_tarsal]:
            if x.shape!=(2,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Invalid tarsal activation')
        if type(self.tarsal_time_ns) is not int or type(self.tarsal_origin_ns) is not int or not 0<=self.tarsal_origin_ns<=self.tarsal_time_ns or self.tarsal_time_ns!=self.steps*round(self.dt*1e9):raise ValueError('Invalid tarsal clock')
        if not np.isfinite(self.last_tarsal_torque_Nm) or abs(self.last_tarsal_torque_Nm)>self.tarsal_prior['torque_budget_Nm']*(1+1e-12):raise ValueError('Invalid tarsal torque')
        total=self._inherited_generalized()+self.last_serial_generalized_SI
        if not np.array_equal(self.data.qfrc_applied,total/FLYBODY_TORQUE_TO_NM):raise ValueError('Undeclared serial body force')

    def advance(self,torque_native,nsteps=1):
        torque=np.asarray(torque_native,dtype=float)*FLYGYM_TORQUE_TO_NM;limit=MAX_TORQUE_NATIVE*FLYGYM_TORQUE_TO_NM
        if torque.shape!=(6,) or not np.isfinite(torque).all() or np.any(abs(torque)>limit*(1+1e-12)) or type(nsteps) is not int or nsteps<0:raise ValueError('Invalid tibial torque')
        self.validate_coxa()
        if np.any(self.data.xfrc_applied):raise ValueError('Undeclared external body force')
        for _ in range(nsteps):
            reject_external_callbacks();self._muscle_step();self._tr_step();self._rf_step();self._tarsal_step()
            out=self.serial_bank.step(self.pending_serial,self.data.qpos,self.data.qvel,round(self.dt*1e9))
            self.last_serial_force_N=out['force_N'];self.last_serial_generalized_SI=out['generalized_SI']
            combined=self._inherited_generalized()+self.last_serial_generalized_SI
            self.data.qfrc_applied[:]=combined/FLYBODY_TORQUE_TO_NM
            if not np.any(combined):self.advance_joint_torque_si(torque,1)
            else:
                self.data.ctrl[:]=torque/FLYBODY_TORQUE_TO_NM;mj.mj_step(self.model,self.data);self.steps+=1
                if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or abs(self.data.time-self.steps*self.dt)>1e-10:raise FloatingPointError('Serial physical step failed')
            dt_ns=round(self.dt*1e9)
            self.coxal_time_ns+=dt_ns;self.tr_time_ns+=dt_ns;self.rf_time_ns+=dt_ns;self.tarsal_time_ns+=dt_ns

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA
        out['serial_ctr']=dict(bank=self.serial_bank.state_dict(),pending=self.pending_serial.copy(),last_force_N=self.last_serial_force_N.copy(),
            last_generalized_SI=self.last_serial_generalized_SI.copy(),origin_ns=self.serial_origin_ns)
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa','trochanter','right_front','rf_tarsal','serial_ctr'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:raise ValueError('Wrong serial body identity')
        if type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0 or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9) or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all():raise ValueError('Invalid serial integration history')
        fields={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}
        if any(set(state[k])!=fields for k in ['coxa','trochanter','right_front']):raise ValueError('Incomplete proximal state')
        t=state['rf_tarsal'];z=state['serial_ctr']
        if set(t)!={'activation','pending','last_torque_Nm','time_ns','origin_ns'} or set(z)!={'bank','pending','last_force_N','last_generalized_SI','origin_ns'}:raise ValueError('Incomplete serial state')
        obj._restore_bilateral(state)
        obj.tarsal_activation=t['activation'].copy();obj.pending_tarsal=t['pending'].copy();obj.last_tarsal_torque_Nm=t['last_torque_Nm'];obj.tarsal_time_ns=t['time_ns'];obj.tarsal_origin_ns=t['origin_ns']
        obj.serial_bank.restore(z['bank']);obj.pending_serial=z['pending'].copy();obj.last_serial_force_N=z['last_force_N'].copy();obj.last_serial_generalized_SI=z['last_generalized_SI'].copy();obj.serial_origin_ns=z['origin_ns']
        obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:raise ValueError('Serial physical clock differs')
        return obj

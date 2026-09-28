"""Source-model FeTi paths in an explicit, guarded operating domain.

Retains effective activation history and pooled MN inputs. Replaces the old
torque contribution after its already committed interval, never adds to it.
The force scale and registered insertions are provisional model priors.
"""
from pathlib import Path
import copy,json,xml.etree.ElementTree as ET
import numpy as np,mujoco as mj
from serial_ctr_body import SerialCTrBody
from active_tendon_bank import TendonBank
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_torque_port import ASSETS
from session_io import sha256
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/tibial_guarded_prior_20260914/prior.json'
PRIOR_SHA256='e19c9732bb112db973d992bf99badf9f68ae997da167c5858d75bd405235bc1c'
POLICY='guarded_FeTi_source_Hill_keep_activation_withdraw_old_torque_v1'

class GuardedTibiaBody(SerialCTrBody):
    SCHEMA='matrix_guarded_tibia_body_v1'

    def __init__(self):
        super().__init__();self._build_tibia()

    def _build_tibia(self):
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed guarded tibia prior')
        self.ft_prior=json.loads(PRIOR.read_text());rows=copy.deepcopy(self.ft_prior['muscles'])
        root=ET.parse(ASSETS/'fruitfly.xml').getroot();root.find('compiler').set('meshdir',str(ASSETS));root.remove(root.find('actuator'))
        for row in rows:
            row['dynprm_s']=row['source_dynprm_s'] # Geometry/gain only; never bank.step().
            ten=ET.SubElement(root.find('tendon'),'spatial',name='hypothesis_'+row['name'])
            for i,s in enumerate(row['sites']):
                name='hypothesis_'+row['name']+'_'+str(i);owner=next(x for x in root.iter('body') if x.get('name')==s['body'])
                ET.SubElement(owner,'site',name=name,pos=' '.join(format(x,'.17g') for x in s['position_cm']));ET.SubElement(ten,'site',site=name)
        gm=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        if (gm.nq,gm.nv)!=(self.model.nq,self.model.nv) or [gm.joint(i).name for i in range(gm.njnt)]!=[self.model.joint(i).name for i in range(self.model.njnt)]:raise ValueError('FeTi coordinate mapping changed')
        self.ft_bank=TendonBank(gm,rows);self.ft_qadr=np.array([r['qadr'] for r in self.ft_prior['domain']]);self.ft_vadr=np.array([r['vadr'] for r in self.ft_prior['domain']])
        self.ft_bounds=np.array([[r['lower_rad'],r['upper_rad']] for r in self.ft_prior['domain']])
        self.ft_origin_ns=self.steps*round(self.dt*1e9);self.ft_enabled=True
        self.ft_last_force_N=np.zeros(12);self.ft_last_generalized_SI=np.zeros(self.model.nv)
        self._ft_muscles=None
        self.identity=copy.deepcopy(self.identity);self.identity.update(guarded_tibia_source_sha256=sha256(__file__),guarded_tibia_prior_sha256=PRIOR_SHA256,guarded_tibia_policy=POLICY)

    @classmethod
    def from_parent(cls,parent,*,enabled=True):
        if type(parent) is not SerialCTrBody or type(enabled) is not bool:raise ValueError('Expected canonical serial body and explicit mode')
        parent.validate_coxa();before=parent.integration_state().copy();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._build_tibia();obj.ft_enabled=enabled;obj.validate_coxa();np.testing.assert_array_equal(before,obj.integration_state());return obj

    def bind_muscles(self,muscles):
        if type(muscles) is not CNSFlyBodyMuscles or muscles.body is not self:raise ValueError('Wrong effective activation owner')
        self._ft_muscles=muscles

    def _active(self):return self.ft_enabled and self.steps*round(self.dt*1e9)>=self.ft_origin_ns+1000000

    def check_tibia_domain(self):
        q=self.data.qpos[self.ft_qadr]
        if np.any(q<=self.ft_bounds[:,0]) or np.any(q>=self.ft_bounds[:,1]):raise ValueError('FeTi source/sign domain exceeded; no force or pose rescue')

    def _inherited_generalized(self):
        base=super()._inherited_generalized()
        return base+self.ft_last_generalized_SI

    def validate_coxa(self):
        super().validate_coxa()
        if type(self.ft_enabled) is not bool or type(self.ft_origin_ns) is not int or not 0<=self.ft_origin_ns<=self.steps*round(self.dt*1e9):raise ValueError('Invalid FeTi mode/clock')
        if self.ft_last_force_N.shape!=(12,) or self.ft_last_generalized_SI.shape!=(self.model.nv,) or not np.isfinite(self.ft_last_force_N).all() or not np.isfinite(self.ft_last_generalized_SI).all() or np.any(self.ft_last_force_N<0):raise ValueError('Invalid FeTi force state')
        if (not self.ft_enabled or self.steps*round(self.dt*1e9)<=self.ft_origin_ns+1000000) and (np.any(self.ft_last_force_N) or np.any(self.ft_last_generalized_SI)):raise ValueError('FeTi force before its replacement boundary')
        other=np.delete(self.ft_last_generalized_SI,self.ft_vadr)
        if np.max(abs(other))>1e-18:raise ValueError('Unexpected non-FeTi force')

    def _muscle_step(self):
        super()._muscle_step()
        if not self._active():return
        mu=self._ft_muscles
        if mu is None or mu.time_ns!=(self.steps+1)*round(self.dt*1e9):raise ValueError('FeTi activation/physical clocks disagree')
        self.check_tibia_domain();bank=self.ft_bank;l,v,j=bank.geometry(self.data.qpos,self.data.qvel)
        force=-np.array([mj.mju_muscleGain(a,z,r,1.,p)*act for a,z,r,p,act in zip(l,v,bank.ranges,bank.gains,mu.activation.ravel())])*1e-6
        if not np.isfinite(force).all() or np.any(force<0):raise ValueError('Invalid active FeTi tension')
        Q=-(j*.01).T@force
        if not np.isclose(Q@self.data.qvel,-force@(v*.001),atol=1e-20,rtol=1e-12):raise ValueError('FeTi power identity failure')
        self.ft_last_force_N=force;self.ft_last_generalized_SI=Q

    def advance(self,torque_native,nsteps=1):
        if type(nsteps) is not int or nsteps!=1:raise ValueError('Coupled FeTi requires one physical step per activation step')
        if self._ft_muscles is None or not np.array_equal(torque_native,self._ft_muscles.last_torque):raise ValueError('FeTi input is not the retained activation result')
        live=self._active()
        if live:self.check_tibia_domain()
        super().advance(np.zeros(6) if live else torque_native,1)
        if live:
            if np.any(self.data.ctrl):raise ValueError('Previous tibial torque remained active')
            self.check_tibia_domain()

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA
        out['guarded_tibia']=dict(origin_ns=self.ft_origin_ns,enabled=self.ft_enabled,last_force_N=self.ft_last_force_N.copy(),last_generalized_SI=self.ft_last_generalized_SI.copy())
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa','trochanter','right_front','rf_tarsal','serial_ctr','guarded_tibia'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:raise ValueError('Wrong FeTi state or identity')
        if type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0 or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9) or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all():raise ValueError('Invalid FeTi physical history')
        fields={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}
        if any(set(state[k])!=fields for k in ['coxa','trochanter','right_front']):raise ValueError('Incomplete inherited proximal state')
        t=state['rf_tarsal'];z=state['serial_ctr'];f=state['guarded_tibia']
        if set(t)!={'activation','pending','last_torque_Nm','time_ns','origin_ns'} or set(z)!={'bank','pending','last_force_N','last_generalized_SI','origin_ns'} or set(f)!={'origin_ns','enabled','last_force_N','last_generalized_SI'}:raise ValueError('Incomplete inherited or FeTi state')
        obj._restore_bilateral(state)
        obj.tarsal_activation=t['activation'].copy();obj.pending_tarsal=t['pending'].copy();obj.last_tarsal_torque_Nm=t['last_torque_Nm'];obj.tarsal_time_ns=t['time_ns'];obj.tarsal_origin_ns=t['origin_ns']
        obj.serial_bank.restore(z['bank']);obj.pending_serial=z['pending'].copy();obj.last_serial_force_N=z['last_force_N'].copy();obj.last_serial_generalized_SI=z['last_generalized_SI'].copy();obj.serial_origin_ns=z['origin_ns']
        obj.ft_origin_ns=f['origin_ns'];obj.ft_enabled=f['enabled'];obj.ft_last_force_N=f['last_force_N'].copy();obj.ft_last_generalized_SI=f['last_generalized_SI'].copy()
        obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:raise ValueError('FeTi physical clock differs')
        return obj

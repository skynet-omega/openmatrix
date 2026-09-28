"""Provisional canonical RF TiTar antagonist port, with a fixed finite torque budget.

No anatomical moment arms, measured Fmax, support feedback or learned policy.
The old body and proximal implementation remain frozen in their parent modules.
"""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from bilateral_front_body import BilateralFrontBody
from flybody_torque_port import MAX_TORQUE_NATIVE, FLYGYM_TORQUE_TO_NM, FLYBODY_TORQUE_TO_NM
from passive_body import reject_external_callbacks
from session_io import sha256
ROOT=Path(__file__).resolve().parents[1]
PRIOR=ROOT/'data/rf_tarsal_interface_20260913/prior.json'
PRIOR_SHA256='6a9996aad47ce74ecae25772fa5fa8ccd4ecfb565f229bdd25175ca27a5d6449'

class RFTarsalBody(BilateralFrontBody):
    SCHEMA='matrix_flybody_rf_tarsal_body_v1'

    def __init__(self):
        super().__init__()
        if sha256(PRIOR)!=PRIOR_SHA256:raise ValueError('Changed RF tarsal prior')
        self.tarsal_prior=json.loads(PRIOR.read_text())
        self.tarsal_ids=self.tarsal_prior['canonical_ids']
        self.tarsal_joint=self.model.joint(self.tarsal_prior['joint']).id
        self.tarsal_qadr=int(self.model.jnt_qposadr[self.tarsal_joint])
        self.tarsal_vadr=int(self.model.jnt_dofadr[self.tarsal_joint])
        d=mj.MjData(self.model);mj.mj_forward(self.model,d)
        axis=d.xaxis[self.tarsal_joint]
        names=['tibia_T1_right','tarsus_T1_right','tarsus2_T1_right']
        x=d.xpos[[self.model.body(n).id for n in names]]
        u=x[0]-x[1];v=x[2]-x[1]
        u-=u.dot(axis)*axis;v-=v.dot(axis)*axis
        sine=axis.dot(np.cross(u,v))/(np.linalg.norm(u)*np.linalg.norm(v))
        if not np.isfinite(sine) or abs(sine)<1e-10:raise ValueError('Undefined distal sign')
        self.tarsal_extension_sign=float(np.sign(sine))
        self.tarsal_activation=np.zeros(2);self.pending_tarsal=np.zeros(2)
        self.last_tarsal_torque_Nm=0.;self.tarsal_time_ns=0;self.tarsal_origin_ns=0
        self.identity.update(rf_tarsal_source_sha256=sha256(__file__),rf_tarsal_prior_sha256=PRIOR_SHA256,
            rf_tarsal_extension_sign=self.tarsal_extension_sign,rf_tarsal_force_biologically_calibrated=False)

    def _restore_bilateral(self,state):
        self._restore_parent(state)
        rf=state['right_front']
        self.rf_activation=rf['activation'].copy();self.pending_rf=rf['pending'].copy()
        self.last_rf_tension_N=rf['last_tension_N'].copy();self.last_rf_torque_Nm=rf['last_torque_Nm'].copy()
        self.rf_time_ns=rf['time_ns'];self.rf_origin_ns=rf['origin_ns']

    @classmethod
    def from_parent(cls,parent,command):
        if type(parent) is not BilateralFrontBody:raise ValueError('Expected CNS198 body')
        parent.validate_coxa();obj=cls();obj._restore_bilateral(parent.state_dict())
        obj.pending_tarsal=np.asarray(command,dtype=float).copy()
        obj.tarsal_time_ns=obj.rf_time_ns;obj.tarsal_origin_ns=obj.tarsal_time_ns
        obj.validate_coxa();np.testing.assert_array_equal(obj.integration_state(),parent.integration_state())
        return obj

    def _tarsal_step(self):
        a=self.tarsal_activation;u=self.pending_tarsal;p=self.tarsal_prior
        tau=np.where(u>a,p['activation_tau_s'],p['deactivation_tau_s'])
        anew=a+(-np.expm1(-self.dt/tau))*(u-a)
        # Midpoint activation torque, with equal/opposite action across one hinge.
        amid=.5*(a+anew)
        self.last_tarsal_torque_Nm=float(p['torque_budget_Nm']*self.tarsal_extension_sign*(amid[0]-amid[1]))
        self.tarsal_activation=anew

    def validate_coxa(self):
        for activation, pending, tension, torque, n, clock, origin in [
            (self.activation, self.pending_coxal, self.last_tension_N, self.last_coxal_torque_Nm, 7, self.coxal_time_ns, self.coxal_origin_ns),
            (self.tr_activation, self.pending_tr, self.last_tr_tension_N, self.last_tr_torque_Nm, 6, self.tr_time_ns, self.tr_origin_ns),
            (self.rf_activation, self.pending_rf, self.last_rf_tension_N, self.last_rf_torque_Nm, 13, self.rf_time_ns, self.rf_origin_ns),
        ]:
            for value, shape in [(activation,(n,)),(pending,(n,)),(tension,(n,)),(torque,(self.model.nv,))]:
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Invalid proximal muscle state')
            if np.any((activation<0)|(activation>1)) or np.any((pending<0)|(pending>1)) or np.any(tension<0):
                raise ValueError('Invalid proximal activation or tension')
            if type(clock) is not int or type(origin) is not int or not 0 <= origin <= clock or clock != self.steps*round(self.dt*1e9):
                raise ValueError('Invalid proximal clock')
        for x in [self.tarsal_activation, self.pending_tarsal]:
            if x.shape != (2,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
                raise ValueError('Invalid tarsal activation or pending input')
        if type(self.tarsal_time_ns) is not int or type(self.tarsal_origin_ns) is not int or not 0 <= self.tarsal_origin_ns <= self.tarsal_time_ns or self.tarsal_time_ns != self.steps*round(self.dt*1e9):
            raise ValueError('Invalid tarsal clock')
        if not np.isfinite(self.last_tarsal_torque_Nm) or abs(self.last_tarsal_torque_Nm)>self.tarsal_prior['torque_budget_Nm']*(1+1e-12):
            raise ValueError('Invalid tarsal torque')
        torque = self.last_coxal_torque_Nm+self.last_tr_torque_Nm+self.last_rf_torque_Nm
        torque[self.tarsal_vadr] += self.last_tarsal_torque_Nm
        if not np.array_equal(self.data.qfrc_applied, torque/FLYBODY_TORQUE_TO_NM):
            raise ValueError('Undeclared external torque')

    def advance(self, torque_native, nsteps=1):
        torque=np.asarray(torque_native,dtype=float)*FLYGYM_TORQUE_TO_NM
        limit=MAX_TORQUE_NATIVE*FLYGYM_TORQUE_TO_NM
        if torque.shape!=(6,) or not np.isfinite(torque).all() or np.any(abs(torque)>limit*(1+1e-12)) or type(nsteps) is not int or nsteps<0:
            raise ValueError('Invalid tibial torque')
        self.validate_coxa()
        if np.any(self.data.xfrc_applied):
            raise ValueError('Undeclared body force')
        for _ in range(nsteps):
            reject_external_callbacks()
            self._muscle_step(); self._tr_step(); self._rf_step(); self._tarsal_step()
            combined=self.last_coxal_torque_Nm+self.last_tr_torque_Nm+self.last_rf_torque_Nm
            combined[self.tarsal_vadr] += self.last_tarsal_torque_Nm
            self.data.qfrc_applied[:]=combined/FLYBODY_TORQUE_TO_NM
            if not np.any(combined):
                self.advance_joint_torque_si(torque,1)
            else:
                self.data.ctrl[:]=torque/FLYBODY_TORQUE_TO_NM
                mj.mj_step(self.model,self.data)
                self.steps+=1
                if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or abs(self.data.time-self.steps*self.dt)>1e-10:
                    raise FloatingPointError('Bilateral physical step failed')
            dt_ns=round(self.dt*1e9)
            self.coxal_time_ns+=dt_ns; self.tr_time_ns+=dt_ns; self.rf_time_ns+=dt_ns; self.tarsal_time_ns+=dt_ns

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA
        out['rf_tarsal']=dict(activation=self.tarsal_activation.copy(),pending=self.pending_tarsal.copy(),
            last_torque_Nm=self.last_tarsal_torque_Nm,time_ns=self.tarsal_time_ns,origin_ns=self.tarsal_origin_ns)
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa','trochanter','right_front','rf_tarsal'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:
            raise ValueError('Wrong RF tarsal body identity')
        if type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0 or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9) or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all():
            raise ValueError('Invalid RF tarsal integration history')
        fields={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}
        if any(set(state[k])!=fields for k in ['coxa','trochanter','right_front']):raise ValueError('Incomplete proximal history')
        t=state['rf_tarsal']
        if set(t)!={'activation','pending','last_torque_Nm','time_ns','origin_ns'}:raise ValueError('Incomplete distal history')
        obj._restore_bilateral(state)
        obj.tarsal_activation=t['activation'].copy();obj.pending_tarsal=t['pending'].copy()
        obj.last_tarsal_torque_Nm=t['last_torque_Nm'];obj.tarsal_time_ns=t['time_ns'];obj.tarsal_origin_ns=t['origin_ns']
        obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:raise ValueError('Distal physical clock differs')
        return obj

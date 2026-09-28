"""RF proximal muscle prior; preserve all existing CNS195 physical history."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from trochanter_body import TrochanterBody
from coxal_body import tendon_row
from flybody_torque_port import ASSETS, MAX_TORQUE_NATIVE, FLYGYM_TORQUE_TO_NM, FLYBODY_TORQUE_TO_NM
from passive_body import reject_external_callbacks
from session_io import sha256

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'data/bilateral_front_geometry_20260913/prior.json'
PRIOR_SHA256 = '683831c8e7a54d000c6eddf863cbb606b81b3d99903da3920fb26ed9503dfa69'


class BilateralFrontBody(TrochanterBody):
    SCHEMA = 'matrix_flybody_bilateral_front_body_v1'

    def __init__(self):
        super().__init__()
        if sha256(PRIOR) != PRIOR_SHA256:
            raise ValueError('Changed bilateral prior')
        self.rf_prior = json.loads(PRIOR.read_text())
        rows = self.rf_prior['muscles']
        self.rf_names = [r['name'] for r in rows]
        self.rf_ids = [r['canonical_ids'] for r in rows]
        self.rf_gain = np.array([r['source_gainprm'] for r in rows])
        self.rf_dynamics = np.array([r['source_dynprm_s'] for r in rows])
        self.rf_lengthrange = np.array([r['source_lengthrange_mm'] for r in rows])
        root = ET.parse(ASSETS/'fruitfly.xml').getroot()
        root.find('compiler').set('meshdir', str(ASSETS))
        root.remove(root.find('actuator'))
        tendons = root.find('tendon')
        for row in rows:
            tendon = ET.SubElement(tendons, 'spatial', name='matrix_'+row['name'])
            for i, site in enumerate(row['sites']):
                name = 'matrix_'+row['name']+'_'+str(i)
                owner = next(b for b in root.iter('body') if b.get('name') == site['body'])
                ET.SubElement(owner, 'site', name=name,
                              pos=' '.join(format(x, '.17g') for x in site['position_cm']))
                ET.SubElement(tendon, 'site', site=name)
        self.rf_geometry = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.rf_data = mj.MjData(self.rf_geometry)
        if (self.rf_geometry.nv, self.rf_geometry.nq) != (self.model.nv, self.model.nq):
            raise ValueError('RF generalized coordinates changed')
        if [self.model.joint(i).name for i in range(self.model.njnt)] != [self.rf_geometry.joint(i).name for i in range(self.rf_geometry.njnt)]:
            raise ValueError('RF joint order changed')
        self.rf_tendon_ids = [self.rf_geometry.tendon('matrix_'+n).id for n in self.rf_names]
        self.rf_activation = np.zeros(13)
        self.pending_rf = np.zeros(13)
        self.last_rf_tension_N = np.zeros(13)
        self.last_rf_torque_Nm = np.zeros(self.model.nv)
        self.rf_time_ns = 0
        self.rf_origin_ns = 0
        self.identity.update(bilateral_front_source_sha256=sha256(__file__),
                             bilateral_front_prior_sha256=PRIOR_SHA256,
                             bilateral_front_scope=self.rf_prior['geometry_scope'])

    def _restore_parent(self, state):
        self._restore_inherited(state)
        tr = state['trochanter']
        self.tr_activation = tr['activation'].copy()
        self.pending_tr = tr['pending'].copy()
        self.last_tr_tension_N = tr['last_tension_N'].copy()
        self.last_tr_torque_Nm = tr['last_torque_Nm'].copy()
        self.tr_time_ns = tr['time_ns']
        self.tr_origin_ns = tr['origin_ns']

    @classmethod
    def from_parent(cls, parent, command):
        if type(parent) is not TrochanterBody:
            raise ValueError('Expected continuing CNS195 body')
        obj = cls()
        obj._restore_parent(parent.state_dict())
        obj.pending_rf = np.asarray(command, dtype=float).copy()
        obj.rf_time_ns = obj.tr_time_ns
        obj.rf_origin_ns = obj.rf_time_ns
        obj.validate_coxa()
        np.testing.assert_array_equal(obj.integration_state(), parent.integration_state())
        return obj

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
        torque = self.last_coxal_torque_Nm+self.last_tr_torque_Nm+self.last_rf_torque_Nm
        if not np.array_equal(self.data.qfrc_applied, torque/FLYBODY_TORQUE_TO_NM):
            raise ValueError('Undeclared external torque')

    def _rf_step(self):
        a, u, dt = self.rf_activation, self.pending_rf, self.dt
        def f(v):
            return np.array([mj.mju_muscleDynamics(float(x),float(y),p) for x,y,p in zip(u,v,self.rf_dynamics)])
        k1=f(a); k2=f(a+.5*dt*k1); k3=f(a+.5*dt*k2); k4=f(a+dt*k3)
        anew=a+dt*(k1+2*k2+2*k3+k4)/6
        if np.any((anew<0)|(anew>1)):
            raise FloatingPointError('RF activation out of bounds')
        gd, gm = self.rf_data, self.rf_geometry
        gd.qpos[:] = self.data.qpos
        gd.qvel[:] = self.data.qvel
        mj.mj_forward(gm,gd)
        jac=np.array([tendon_row(gm,gd,i) for i in self.rf_tendon_ids])
        length=gd.ten_length[self.rf_tendon_ids]*10
        velocity=jac@self.data.qvel*10
        tension=-np.array([mj.mju_muscleGain(l,v,r,1.,p)*ac for l,v,r,p,ac in
                          zip(length,velocity,self.rf_lengthrange,self.rf_gain,(a+anew)*.5)])*1e-6
        if not np.isfinite(tension).all() or np.any(tension<0):
            raise FloatingPointError('Invalid RF tension')
        self.rf_activation=anew
        self.last_rf_tension_N=tension
        self.last_rf_torque_Nm=-(jac*.01).T@tension

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
            self._muscle_step(); self._tr_step(); self._rf_step()
            combined=self.last_coxal_torque_Nm+self.last_tr_torque_Nm+self.last_rf_torque_Nm
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
            self.coxal_time_ns+=dt_ns; self.tr_time_ns+=dt_ns; self.rf_time_ns+=dt_ns

    def state_dict(self):
        out=super().state_dict()
        out['schema']=self.SCHEMA
        out['right_front']=dict(activation=self.rf_activation.copy(),pending=self.pending_rf.copy(),
            last_tension_N=self.last_rf_tension_N.copy(),last_torque_Nm=self.last_rf_torque_Nm.copy(),
            time_ns=self.rf_time_ns,origin_ns=self.rf_origin_ns)
        return out

    @classmethod
    def from_state(cls,state):
        obj=cls()
        if set(state)!={'schema','identity','steps','integration','origin_ns','coxa','trochanter','right_front'} or state['schema']!=cls.SCHEMA or state['identity']!=obj.identity:
            raise ValueError('Wrong bilateral body identity')
        if type(state['steps']) is not int or type(state['origin_ns']) is not int or state['steps']<0 or not 0<=state['origin_ns']<=state['steps']*round(obj.dt*1e9) or state['integration'].shape!=obj.integration_state().shape or not np.isfinite(state['integration']).all():
            raise ValueError('Invalid bilateral body state')
        fields={'activation','pending','last_tension_N','last_torque_Nm','time_ns','origin_ns'}
        if any(set(state[k])!=fields for k in ['coxa','trochanter','right_front']):
            raise ValueError('Incomplete muscle history')
        obj._restore_parent(state)
        rf=state['right_front']
        obj.rf_activation=rf['activation'].copy(); obj.pending_rf=rf['pending'].copy()
        obj.last_rf_tension_N=rf['last_tension_N'].copy(); obj.last_rf_torque_Nm=rf['last_torque_Nm'].copy()
        obj.rf_time_ns=rf['time_ns']; obj.rf_origin_ns=rf['origin_ns']
        obj.validate_coxa()
        if abs(obj.data.time-obj.steps*obj.dt)>1e-10:
            raise ValueError('Bilateral physical clock differs')
        return obj

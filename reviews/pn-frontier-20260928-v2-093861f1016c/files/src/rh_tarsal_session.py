"""RH800358 effective motor port; numerical capacity is distinct from recruitment."""
from pathlib import Path
import copy
import numpy as np
from guarded_tibia_session import GuardedTibiaSession,SOURCES as PARENT_SOURCES
from guarded_tibia_body import PRIOR_SHA256 as FT_PRIOR_SHA256,POLICY as FT_POLICY
from rh_tarsal_body import RHTarsalBody,PRIOR_SHA256 as RH_PRIOR_SHA256
from rh_tarsal_storage import load_rh_tarsal_session
from rf_tarsal_identity_session import POLICY as RF_POLICY
from serial_ctr_body import PRIOR_SHA256 as SERIAL_PRIOR_SHA256
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyEye,FlyBodyProprioception
from pn_graph_solver_session import PnGraphSolverSession
from coxal_body import PRIOR_SHA256 as COXAL_PRIOR_SHA256
from trochanter_body import PRIOR_SHA256 as TROCHANTER_PRIOR_SHA256
from bilateral_front_body import PRIOR_SHA256 as RIGHT_FRONT_PRIOR_SHA256
from rf_tarsal_body import PRIOR_SHA256 as TARSAL_PRIOR_SHA256
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies

SOURCES=tuple(PARENT_SOURCES)+('rh_tarsal_body.py','rh_tarsal_storage.py','rh_tarsal_session.py')
ENTRYPOINT='rh_tarsal_session.py'
POLICY='RH800358_effective_depressor_keep_committed_ms_v1'

class RHTarsalSession(GuardedTibiaSession):
    SCHEMA='matrix_rh_tarsal_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,connected=True,diagnostic_pulse_ms=0):
        if type(connected) is not bool or type(diagnostic_pulse_ms) is not int or diagnostic_pulse_ms not in (0,1):
            raise ValueError('Explicit canonical mode or declared one-ms capacity pulse required')
        if diagnostic_pulse_ms and not connected:raise ValueError('Pulse requires connected experimental port')
        parent=GuardedTibiaSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','body'}
            before=_fingerprints(parent,excluded);old=parent.body.integration_state().copy()
            obj.body=RHTarsalBody.from_parent(parent.body,enabled=connected)
            obj.muscles.body=obj.body;obj.body.bind_muscles(obj.muscles)
            obj.proprioception.body=obj.body;obj.eyes.body=obj.body;obj._index_rh()
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA,rh_tarsal_policy=dict(policy=POLICY,canonical_id=800358,
                connected=connected,origin_ns=obj.time_ns,prior_sha256=RH_PRIOR_SHA256,
                diagnostic_pulse_ms=diagnostic_pulse_ms,pulse_amplitude=1.0 if diagnostic_pulse_ms else 0.0,
                pulse_scope='declared_motor_capacity_fixture_not_neural_recruitment' if diagnostic_pulse_ms else 'canonical_release_only'),
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)))
            obj.body.rh_pending=obj.rh_command()
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('RH admission changed inherited non-body state')
            np.testing.assert_array_equal(old,obj.body.integration_state())
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation=POLICY,
                parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,previous_active_RH_TiTa_contribution=0,new_activation_zero=True,
                committed_interval_preserved_ns=1000000,neural_parameters_changed=False,canonical_neurons_removed=0,
                biological_force_calibration=False,diagnostic_motor_pulse_ms=diagnostic_pulse_ms)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();obj._validate_rh_pending();return obj
        except BaseException:obj.close();raise

    def _index_rh(self):
        self.rh_index=int(np.searchsorted(self.brain.node_ids,800358))
        if self.rh_index>=len(self.brain.node_ids) or self.brain.node_ids[self.rh_index]!=800358:raise ValueError('Missing RH800358')

    def rh_command(self):
        p=self.config['rh_tarsal_policy']
        if not p['connected']:return 0.0
        age=self.time_ns-p['origin_ns']
        if p['diagnostic_pulse_ms'] and 1000000<=age<2000000:return 1.0
        return float(self.hybrid.release()[self.rh_index])

    def _validate_rh_pending(self):
        if self.body.rh_pending!=self.rh_command():raise ValueError('RH pending differs from its explicit input policy')

    def _validate(self):
        # Frozen parent requires its exact body class. Retain its substantive
        # policy checks here, with the explicitly versioned extended body.
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not RHTarsalBody or type(self.muscles) is not CNSFlyBodyMuscles
            or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
            or self.output_connected or self.pending_cyborg_command!=0.
            or self.config['flybody_clock_origin_ns']!=self.body.origin_ns or self.body._ft_muscles is not self.muscles):raise ValueError('Invalid RH physical family')
        for key,digest,origin,required in [('coxal',COXAL_PRIOR_SHA256,self.body.coxal_origin_ns,True),('trochanter',TROCHANTER_PRIOR_SHA256,self.body.tr_origin_ns,True),('right_front',RIGHT_FRONT_PRIOR_SHA256,self.body.rf_origin_ns,False),('rf_tarsal',TARSAL_PRIOR_SHA256,self.body.tarsal_origin_ns,False)]:
            connected=self.config[key+'_input_connected']
            if type(connected) is not bool or (required and not connected) or self.config[key+'_prior_sha256']!=digest or self.config[key+'_origin_ns']!=origin:raise ValueError('Inherited motor policy changed')
        p=self.config['motor_input_intervention']
        if p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator']) or p['policy']!='zero_one_future_release_keep_pool_denominator_v1' or p['inherited_pending_interval_preserved'] is not True:raise ValueError('RH requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.rf_time_ns!=self.time_ns or self.body.tarsal_time_ns!=self.time_ns or self.body.serial_bank.time_ns!=self.time_ns:raise ValueError('Inherited motor clocks differ')
        if type(self.config['serial_ctr_connected']) is not bool or self.config['serial_ctr_prior_sha256']!=SERIAL_PRIOR_SHA256 or self.config['serial_ctr_origin_ns']!=self.body.serial_origin_ns:raise ValueError('Serial policy differs')
        p=self.config.get('rf_tarsal_identity_policy',{});fixed=dict(policy=RF_POLICY,removed_id=820896,role='levator',position=2,denominator=3,inherited_pending_interval_preserved=True,muscle_reassigned=False)
        if (set(p)!=set(fixed)|{'withdraw','start_ns'} or any(p.get(k)!=v or type(p.get(k)) is not type(v) for k,v in fixed.items()) or type(p['withdraw']) is not bool or type(p['start_ns']) is not int
            or not 0<=p['start_ns']<=self.time_ns or (self.time_ns-p['start_ns'])%self.CONTROL_NS or self.body.tarsal_ids!=[[825721],[817210,820110,820896]]):raise ValueError('Changed RF identity quarantine')
        if self.config['guarded_tibia_policy']!=FT_POLICY or self.config['guarded_tibia_prior_sha256']!=FT_PRIOR_SHA256 or type(self.config['guarded_tibia_enabled']) is not bool or self.config['guarded_tibia_enabled']!=self.body.ft_enabled or self.config['guarded_tibia_origin_ns']!=self.body.ft_origin_ns:raise ValueError('FeTi policy/state disagree')
        p=self.config['rh_tarsal_policy'];required={'policy','canonical_id','connected','origin_ns','prior_sha256','diagnostic_pulse_ms','pulse_amplitude','pulse_scope'}
        if set(p)!=required or p['policy']!=POLICY or type(p['canonical_id']) is not int or p['canonical_id']!=800358 or p['prior_sha256']!=RH_PRIOR_SHA256:raise ValueError('Invalid RH policy')
        if type(p['connected']) is not bool or p['connected']!=self.body.rh_enabled or type(p['origin_ns']) is not int or p['origin_ns']!=self.body.rh_origin_ns or self.body.rh_time_ns!=self.time_ns:raise ValueError('RH clock/mode mismatch')
        pulse=p['diagnostic_pulse_ms']
        if type(pulse) is not int or pulse not in (0,1) or (pulse and not p['connected']) or p['pulse_amplitude']!=(1.0 if pulse else 0.0) or p['pulse_scope']!=('declared_motor_capacity_fixture_not_neural_recruitment' if pulse else 'canonical_release_only'):raise ValueError('Undeclared RH input fixture')

    def step(self):
        self._validate_rh_pending();used=self.body.rh_pending;natural=float(self.hybrid.release()[self.rh_index])
        row=super().step();self.body.rh_pending=self.rh_command();self._validate_rh_pending()
        row.update(rh_tarsal_command_used=used,rh_tarsal_command_pending=self.body.rh_pending,rh_tarsal_canonical_release_before=natural,
            rh_tarsal_activation=self.body.rh_activation,rh_tarsal_torque_Nm=self.body.rh_last_torque_Nm,
            rh_tarsal_input_policy=copy.deepcopy(self.config['rh_tarsal_policy']))
        return row

    def _manifest_fields(self):
        x=super()._manifest_fields();x.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),physical_body_schema=RHTarsalBody.SCHEMA,
            rh_tarsal_policy=copy.deepcopy(self.config['rh_tarsal_policy']),rh_tarsal_biological_force_calibration=False,rh_tarsal_neural_recruitment_demonstrated=False)
        return x

    def save(self,path):
        self._validate_rh_pending();self._validate_tarsal_pending();self._validate_serial_pending();self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('RH source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_rh_tarsal_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)

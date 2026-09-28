"""Continuing CNS with bounded-domain FeTi model prior; no neural change."""
from pathlib import Path
import copy,numpy as np
from rf_tarsal_identity_session import RFTarsalIdentitySession,SOURCES as PARENT_SOURCES,POLICY as RF_POLICY
from guarded_tibia_body import GuardedTibiaBody,PRIOR_SHA256,POLICY
from guarded_tibia_storage import load_guarded_tibia_session
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
SOURCES=tuple(PARENT_SOURCES)+('guarded_tibia_body.py','guarded_tibia_storage.py','guarded_tibia_session.py')
ENTRYPOINT='guarded_tibia_session.py'

class GuardedTibiaSession(RFTarsalIdentitySession):
    SCHEMA='matrix_guarded_tibia_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,enabled=True):
        parent=RFTarsalIdentitySession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','body'};before=_fingerprints(parent,excluded)
            oldbody=parent.body.state_dict();obj.body=GuardedTibiaBody.from_parent(parent.body,enabled=enabled)
            obj.muscles.body=obj.body;obj.body.bind_muscles(obj.muscles);obj.proprioception.body=obj.body;obj.eyes.body=obj.body
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA,guarded_tibia_policy=POLICY,guarded_tibia_prior_sha256=PRIOR_SHA256,
                guarded_tibia_enabled=enabled,guarded_tibia_origin_ns=obj.time_ns,
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('FeTi adoption changed inherited non-body state')
            np.testing.assert_array_equal(oldbody['integration'],obj.body.integration_state())
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation=POLICY,parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),
                time_ns=obj.time_ns,preserved_state_checks=checks,first_committed_interval_preserved=True,old_tibia_torque_withdrawn_once_after_ms=1,
                canonical_neurons_removed=0,neural_parameters_changed=False,biological_force_calibration=False,anatomy_is_transferred_model_prior=True,whole_articular_range_qualified=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        # Keep parent serial/RF contracts explicit. The frozen parent requires
        # its exact body type, so it cannot validate this new physical family.
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not GuardedTibiaBody or type(self.muscles) is not CNSFlyBodyMuscles or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
            or self.output_connected or self.pending_cyborg_command!=0. or self.config['flybody_clock_origin_ns']!=self.body.origin_ns or self.body._ft_muscles is not self.muscles):raise ValueError('Invalid guarded FeTi physical family')
        for key,digest,origin,required in [('coxal',COXAL_PRIOR_SHA256,self.body.coxal_origin_ns,True),('trochanter',TROCHANTER_PRIOR_SHA256,self.body.tr_origin_ns,True),('right_front',RIGHT_FRONT_PRIOR_SHA256,self.body.rf_origin_ns,False),('rf_tarsal',TARSAL_PRIOR_SHA256,self.body.tarsal_origin_ns,False)]:
            connected=self.config[key+'_input_connected']
            if type(connected) is not bool or (required and not connected) or self.config[key+'_prior_sha256']!=digest or self.config[key+'_origin_ns']!=origin:raise ValueError('Inherited motor policy changed')
        p=self.config['motor_input_intervention']
        if p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator']) or p['policy']!='zero_one_future_release_keep_pool_denominator_v1' or p['inherited_pending_interval_preserved'] is not True:raise ValueError('FeTi requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.rf_time_ns!=self.time_ns or self.body.tarsal_time_ns!=self.time_ns or self.body.serial_bank.time_ns!=self.time_ns:raise ValueError('Inherited serial session clocks differ')
        if type(self.config['serial_ctr_connected']) is not bool or self.config['serial_ctr_prior_sha256']!=SERIAL_PRIOR_SHA256 or self.config['serial_ctr_origin_ns']!=self.body.serial_origin_ns:raise ValueError('Inherited serial policy differs')
        p=self.config.get('rf_tarsal_identity_policy',{});fixed=dict(policy=RF_POLICY,removed_id=820896,role='levator',position=2,denominator=3,inherited_pending_interval_preserved=True,muscle_reassigned=False)
        if (set(p)!=set(fixed)|{'withdraw','start_ns'} or any(p.get(k)!=v or type(p.get(k)) is not type(v) for k,v in fixed.items()) or type(p['withdraw']) is not bool or type(p['start_ns']) is not int
            or not 0<=p['start_ns']<=self.time_ns or (self.time_ns-p['start_ns'])%self.CONTROL_NS or self.body.tarsal_ids!=[[825721],[817210,820110,820896]]):raise ValueError('Changed RF identity quarantine')
        if self.config['guarded_tibia_policy']!=POLICY or self.config['guarded_tibia_prior_sha256']!=PRIOR_SHA256 or type(self.config['guarded_tibia_enabled']) is not bool or self.config['guarded_tibia_enabled']!=self.body.ft_enabled or self.config['guarded_tibia_origin_ns']!=self.body.ft_origin_ns:raise ValueError('FeTi policy/state disagree')

    def step(self):
        live=self.body._active()
        row=super().step()
        if live:
            # Parent appends this same row to history. Label the retired
            # constitutive calculation, and expose the force actually used.
            row.update(retired_effective_force_native=row['muscle_force_native'],
                retired_effective_torque_native=row['torque_native'],
                muscle_force_native=(self.body.ft_last_force_N.reshape(6,2)/1e-6).tolist(),
                torque_native=(self.body.ft_last_generalized_SI[self.body.ft_vadr]/1e-9).tolist(),
                guarded_tibia_force_N=self.body.ft_last_force_N.tolist(),
                guarded_tibia_generalized_SI=self.body.ft_last_generalized_SI.tolist(),
                guarded_tibia_prior=True,guarded_tibia_old_torque_removed=True)
        return row

    def _manifest_fields(self):
        x=super()._manifest_fields();x.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),physical_body_schema=GuardedTibiaBody.SCHEMA,
            guarded_tibia_prior_sha256=PRIOR_SHA256,guarded_tibia_enabled=self.body.ft_enabled,guarded_tibia_origin_ns=self.body.ft_origin_ns,
            guarded_tibia_biologically_calibrated=False,guarded_tibia_whole_range_validated=False)
        return x

    def save(self,path):
        self._validate();self._validate_tarsal_pending();self._validate_serial_pending();self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('FeTi source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_guarded_tibia_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)

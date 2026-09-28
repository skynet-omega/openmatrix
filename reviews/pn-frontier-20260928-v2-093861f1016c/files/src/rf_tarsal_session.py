"""Add a declared effective RF TiTar antagonist pair to the continuing CNS198."""
from pathlib import Path
import copy
import numpy as np
from bilateral_front_session import BilateralFrontSession,SOURCES as PARENT_SOURCES
from bilateral_front_body import PRIOR_SHA256
from trochanter_body import PRIOR_SHA256 as TROCHANTER_PRIOR_SHA256
from coxal_body import PRIOR_SHA256 as COXAL_PRIOR_SHA256
from rf_tarsal_body import RFTarsalBody,PRIOR_SHA256 as TARSAL_PRIOR_SHA256
from rf_tarsal_storage import load_rf_tarsal_session
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyEye,FlyBodyProprioception
from pn_graph_solver_session import PnGraphSolverSession
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
SOURCES=tuple(PARENT_SOURCES)+('rf_tarsal_body.py','rf_tarsal_storage.py','rf_tarsal_session.py')
ENTRYPOINT='rf_tarsal_session.py'

class RFTarsalSession(BilateralFrontSession):
    SCHEMA='matrix_flybody_rf_tarsal_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,connected=True):
        if type(connected) is not bool:raise ValueError('Explicit distal connection required')
        parent=BilateralFrontSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            if not parent.config['right_front_input_connected']:raise ValueError('Expected connected CNS198')
            excluded={'schema','config','source_identity','intervention','body'}
            before=_fingerprints(parent,excluded);old_body=parent.body
            obj.body=RFTarsalBody.from_parent(old_body,np.zeros(2));obj._index_tarsal()
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(rf_tarsal_input_connected=connected,rf_tarsal_prior_sha256=TARSAL_PRIOR_SHA256,
                rf_tarsal_origin_ns=obj.time_ns,candidate=cls.SCHEMA,
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)))
            obj.body.pending_tarsal=obj.tarsal_command()
            obj.muscles.body=obj.body;obj.proprioception.body=obj.body;obj.eyes.body=obj.body
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Distal adoption changed inherited neural/sensory history')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Add RF effective TiTar depressor/levator from four canonical MN, no previous active distal contribution',
                parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),
                time_ns=obj.time_ns,preserved_state_checks=checks,physical_integration_state_preserved=True,
                new_activation_zero=True,inherited_activation_and_pending_retained=True,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_coxal_pending();obj._validate_tr_pending();obj._validate_rf_pending();obj._validate_tarsal_pending();obj._validate_afferent_pending()
            old_body.close();return obj
        except BaseException:obj.close();raise

    def _index_tarsal(self):
        self.tarsal_indices=[np.searchsorted(self.brain.node_ids,ids) for ids in self.body.tarsal_ids]
        for idx,ids in zip(self.tarsal_indices,self.body.tarsal_ids):
            if np.any(idx>=len(self.brain.node_ids)) or not np.array_equal(self.brain.node_ids[idx],ids):raise ValueError('Missing canonical tarsal MN')

    def tarsal_command(self):
        if not self.config['rf_tarsal_input_connected']:return np.zeros(2)
        q=self.hybrid.release();return np.array([q[idx].mean() for idx in self.tarsal_indices])

    def _validate_tarsal_pending(self):
        if not np.array_equal(self.body.pending_tarsal,self.tarsal_command()):raise ValueError('Tarsal pending differs from canonical release')

    def step(self):
        self._validate_tarsal_pending();used=self.body.pending_tarsal.copy()
        row=super().step();self.body.pending_tarsal=self.tarsal_command();self._validate_tarsal_pending()
        row.update(rf_tarsal_command_used=used.tolist(),rf_tarsal_command_pending=self.body.pending_tarsal.tolist(),
            rf_tarsal_activation=self.body.tarsal_activation.tolist(),rf_tarsal_torque_Nm=self.body.last_tarsal_torque_Nm,
            rf_tarsal_prior_sha256=TARSAL_PRIOR_SHA256,rf_tarsal_input_connected=self.config['rf_tarsal_input_connected'])
        return row

    def _manifest_fields(self):
        out=super()._manifest_fields()
        out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),physical_body_schema=RFTarsalBody.SCHEMA,
            rf_tarsal_prior_sha256=TARSAL_PRIOR_SHA256,rf_tarsal_effective_activations=2,rf_tarsal_canonical_MN=4,
            rf_tarsal_input_connected=self.config['rf_tarsal_input_connected'],rf_tarsal_force_biologically_calibrated=False)
        return out

    def save(self,path):
        self._validate_tarsal_pending();self._validate_rf_pending();self._validate_tr_pending();self._validate_coxal_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Distal source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_rf_tarsal_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)

    def _validate(self):
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not RFTarsalBody or type(self.muscles) is not CNSFlyBodyMuscles or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
            or self.output_connected or self.pending_cyborg_command!=0. or self.config['flybody_clock_origin_ns']!=self.body.origin_ns
            or type(self.config['coxal_input_connected']) is not bool or not self.config['coxal_input_connected'] or self.config['coxal_prior_sha256']!=COXAL_PRIOR_SHA256 or self.config['coxal_origin_ns']!=self.body.coxal_origin_ns
            or type(self.config['trochanter_input_connected']) is not bool or not self.config['trochanter_input_connected'] or self.config['trochanter_prior_sha256']!=TROCHANTER_PRIOR_SHA256 or self.config['trochanter_origin_ns']!=self.body.tr_origin_ns
            or type(self.config['right_front_input_connected']) is not bool or self.config['right_front_prior_sha256']!=PRIOR_SHA256 or self.config['right_front_origin_ns']!=self.body.rf_origin_ns):raise ValueError('Invalid RF proximal family/body policy')
        p=self.config['motor_input_intervention']
        if p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator']) or p['policy']!='zero_one_future_release_keep_pool_denominator_v1' or p['inherited_pending_interval_preserved'] is not True:raise ValueError('RF proximal family requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.rf_time_ns!=self.time_ns:raise ValueError('RF proximal/session clocks differ')

        if (type(self.config['rf_tarsal_input_connected']) is not bool or self.config['rf_tarsal_prior_sha256']!=TARSAL_PRIOR_SHA256 or self.config['rf_tarsal_origin_ns']!=self.body.tarsal_origin_ns or self.body.tarsal_time_ns!=self.time_ns):
            raise ValueError('Invalid distal interface clock/policy')


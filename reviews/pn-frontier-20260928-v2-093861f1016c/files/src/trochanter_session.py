"""Continuing CNS192 plus six declared coxa–trochanter muscle paths."""
from pathlib import Path
import copy
import numpy as np
from coxal_session import CoxalSession,SOURCES as PARENT_SOURCES
from coxal_body import PRIOR_SHA256 as COXAL_PRIOR_SHA256
from trochanter_body import TrochanterBody,PRIOR_SHA256
from trochanter_storage import load_trochanter_session
from flybody_cns_body import CNSFlyBodyMuscles
from flybody_cns_sensors import FlyBodyEye,FlyBodyProprioception
from pn_graph_solver_session import PnGraphSolverSession
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
SOURCES=tuple(PARENT_SOURCES)+('trochanter_body.py','trochanter_storage.py','trochanter_session.py')
ENTRYPOINT='trochanter_session.py'

class TrochanterSession(CoxalSession):
    SCHEMA='matrix_flybody_trochanter_session_v1'

    @classmethod
    def from_checkpoint(cls,path,*,connected=True):
        if type(connected) is not bool:raise ValueError('Explicit CTr connection required')
        parent=CoxalSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            if not parent.config['coxal_input_connected']:raise ValueError('Adopt the connected coxal reference')
            excluded={'schema','config','source_identity','intervention','body'};before=_fingerprints(parent,excluded);old_body=parent.body
            obj.body=TrochanterBody.from_parent(old_body,np.zeros(6));obj._index_tr()
            obj.config=copy.deepcopy(parent.config);obj.config.update(trochanter_input_connected=connected,trochanter_prior_sha256=PRIOR_SHA256,trochanter_origin_ns=obj.time_ns,candidate=cls.SCHEMA)
            obj.body.pending_tr=obj.trochanter_command();obj.muscles.body=obj.body;obj.proprioception.body=obj.body;obj.eyes.body=obj.body
            closure=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES);obj.config['runtime_source_contract']=dict(entrypoint=ENTRYPOINT,static_local_imports=closure)
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('CTr adoption changed inherited history: '+str(checks))
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Add six provisional LF CTr active paths to13concordant MN; no earlier active CTr contribution',parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,physical_integration_state_preserved=True,new_activation_zero=True,inherited_coxal_activation_and_pending_retained=True,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();obj._validate_pending();obj._validate_coxal_pending();obj._validate_tr_pending();obj._validate_afferent_pending();old_body.close();return obj
        except BaseException:obj.close();raise

    def _index_tr(self):
        self.tr_indices=[np.searchsorted(self.brain.node_ids,ids) for ids in self.body.tr_ids]
        for idx,ids in zip(self.tr_indices,self.body.tr_ids):
            if np.any(idx>=len(self.brain.node_ids)) or not np.array_equal(self.brain.node_ids[idx],ids):raise ValueError('Missing canonical CTr MN')

    def trochanter_command(self):
        if not self.config['trochanter_input_connected']:return np.zeros(6)
        q=self.hybrid.release();return np.array([q[idx].mean() for idx in self.tr_indices])

    def _validate(self):
        PnGraphSolverSession._validate(self)
        if (type(self.body) is not TrochanterBody or type(self.muscles) is not CNSFlyBodyMuscles or type(self.eyes) is not FlyBodyEye or type(self.proprioception) is not FlyBodyProprioception
            or self.output_connected or self.pending_cyborg_command!=0. or self.config['flybody_clock_origin_ns']!=self.body.origin_ns
            or type(self.config['coxal_input_connected']) is not bool or not self.config['coxal_input_connected'] or self.config['coxal_prior_sha256']!=COXAL_PRIOR_SHA256 or self.config['coxal_origin_ns']!=self.body.coxal_origin_ns
            or type(self.config['trochanter_input_connected']) is not bool or self.config['trochanter_prior_sha256']!=PRIOR_SHA256 or self.config['trochanter_origin_ns']!=self.body.tr_origin_ns):raise ValueError('Invalid CTr family/body policy')
        p=self.config['motor_input_intervention']
        if p['removed_id'] is not None or any(p[k] is not None for k in ['leg','role','position','denominator']) or p['policy']!='zero_one_future_release_keep_pool_denominator_v1' or p['inherited_pending_interval_preserved'] is not True:raise ValueError('CTr family requires reference tibial inputs')
        self.body.validate_coxa()
        if self.body.tr_time_ns!=self.time_ns:raise ValueError('CTr/session clocks differ')

    def _validate_tr_pending(self):
        if not np.array_equal(self.body.pending_tr,self.trochanter_command()):raise ValueError('CTr pending input differs from current canonical release')

    def step(self):
        self._validate_tr_pending();used=self.body.pending_tr.copy();row=super().step();self.body.pending_tr=self.trochanter_command();self._validate_tr_pending()
        row.update(trochanter_command_used=used.tolist(),trochanter_command_pending=self.body.pending_tr.tolist(),trochanter_activation=self.body.tr_activation.tolist(),trochanter_force_N=self.body.last_tr_tension_N.tolist(),trochanter_prior_sha256=PRIOR_SHA256,trochanter_input_connected=self.config['trochanter_input_connected']);return row

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),physical_body_schema=TrochanterBody.SCHEMA,trochanter_prior_sha256=PRIOR_SHA256,trochanter_muscles=6,trochanter_canonical_MN=13,trochanter_input_connected=self.config['trochanter_input_connected'],trochanter_force_biologically_calibrated=False);return out

    def save(self,path):
        self._validate_tr_pending();self._validate_coxal_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('CTr source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_trochanter_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)

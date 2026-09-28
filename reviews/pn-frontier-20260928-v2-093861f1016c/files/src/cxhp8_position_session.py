"""Anatomical hip-position input on a continuing166700-neuron CNS and FlyBody."""
from pathlib import Path
import copy
import numpy as np
from rf_tarsal_session import RFTarsalSession,SOURCES as PARENT_SOURCES
from rf_tarsal_storage import load_rf_tarsal_session
from cxhp8_position_brain import GpuCxHP8PositionBrain
from cxhp8_position_field import CxHP8PositionField,IDS,POLICY,PRIOR_SHA256,AMPLITUDE
from kc_session_storage import save_session
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
SOURCES=tuple(PARENT_SOURCES)+('cxhp8_geometry.py','cxhp8_position_field.py','cxhp8_position_brain.py','cxhp8_position_session.py')
ENTRYPOINT='cxhp8_position_session.py'


class CxHP8PositionSession(RFTarsalSession):
    SCHEMA='matrix_cxhp8_position_session_v1'
    BRAIN=GpuCxHP8PositionBrain

    @classmethod
    def from_checkpoint(cls,path,*,mode='position'):
        parent=RFTarsalSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.cxhp8_field=CxHP8PositionField(obj.body.model);sample=obj.cxhp8_field.sample(obj.body)
            if set(IDS)&set(obj.proprioception.node_ids.tolist()):raise ValueError('Duplicated preexisting proprioceptive drive')
            obj.hybrid=cls.BRAIN.adopt(parent.hybrid,sample,mode=mode)
            obj.config=copy.deepcopy(parent.config);obj.config.update(candidate=cls.SCHEMA,cxhp8_policy=POLICY,cxhp8_mode=mode,cxhp8_prior_sha256=PRIOR_SHA256,electrical_backend_identity=obj.hybrid.backend_identity(),runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('CxHP8 adoption changed inherited body/history')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation=POLICY,parent_checkpoint=str(Path(path).resolve()),parent_manifest_sha256=sha256(Path(path)/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,initial_sample=sample,neural_parameters_changed=False,new_canonical_neurons=0,scope='Seven organ candidates, empirical calcium position index with engineering current80; no previous direct hip input to remove; all recurrent connections andFeCO retained')
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_cxhp8_pending();return obj
        except BaseException:obj.close();raise

    def _validate_cxhp8_pending(self):
        if not hasattr(self,'cxhp8_field') or self.cxhp8_field.model is not self.body.model:self.cxhp8_field=CxHP8PositionField(self.body.model)
        self.hybrid.validate_cxhp8();m=self.hybrid.cxhp8_manifest;p=self.hybrid.cxhp8_pending
        if self.config['cxhp8_policy']!=POLICY or self.config['cxhp8_mode']!=m['mode'] or self.config['cxhp8_prior_sha256']!=PRIOR_SHA256 or p['sample_time_ns']!=self.time_ns:raise ValueError('CxHP8 session policy/clock differs')
        sample=self.cxhp8_field.sample(self.body)
        if not np.array_equal(sample['angles_deg'],p['angles_deg']) or sample['index']!=p['index']:raise ValueError('CxHP8 sample differs from physical posture')

    def step(self):
        self._validate_cxhp8_pending();used=copy.deepcopy(self.hybrid.cxhp8_pending)
        row=super().step();self.hybrid.set_cxhp8_sample(self.cxhp8_field.sample(self.body));self._validate_cxhp8_pending()
        row.update(cxhp8_used=used,cxhp8_pending=copy.deepcopy(self.hybrid.cxhp8_pending),cxhp8_current_used_model_units=AMPLITUDE*used['applied_index'],cxhp8_mode=self.config['cxhp8_mode']);return row

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),cxhp8_policy=POLICY,cxhp8_mode=self.config['cxhp8_mode'],cxhp8_prior_sha256=PRIOR_SHA256,cxhp8_anatomical_organ_candidates=IDS,cxhp8_electrical_calibration=False);return out

    def save(self,path):
        self._validate_cxhp8_pending()
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Changed CxHP8 source contract')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES);obj=load_rf_tarsal_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)
        try:obj._validate_cxhp8_pending();return obj
        except BaseException:obj.close();raise

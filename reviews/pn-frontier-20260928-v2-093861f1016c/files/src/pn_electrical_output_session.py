"""Persist the next partial PN output increment without editing its parent."""
from pathlib import Path
import copy
from pn_cholinergic_cns_session import PnCholinergicSession,SOURCES as PARENT_SOURCES
from pn_electrical_output_brain import GpuPnElectricalOutputBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session
from kcgamma_regional_brain import _record_hash

SCHEMA='matrix_pn_electrical_output_partial_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_electrical_output_port.py','pn_electrical_output_source.py','pn_electrical_output_brain.py','pn_electrical_output_session.py')

class PnElectricalOutputSession(PnCholinergicSession):
    @classmethod
    def from_checkpoint(cls,path,spec,*,enabled=True,coupling_ns=62500):
        path=Path(path).resolve();parent=PnCholinergicSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded);oldsource=parent.hybrid._online_source.state_dict()
            obj.hybrid=GpuPnElectricalOutputBrain.adopt(parent.hybrid,spec,enabled=enabled,coupling_ns=coupling_ns)
            if _record_hash(oldsource)!=_record_hash(obj.hybrid._online_source.state_dict()['base']):raise ValueError('New output altered previous PN/Ca/input histories')
            closure=require_covered_dependencies(ROOT/'src','pn_electrical_output_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config);obj.config.update(candidate='PN_ELECTRICAL_OUTPUT_PARTIAL_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_electrical_output_session.py',static_local_imports=closure),PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Output adoption changed body or pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Replace365KCand1APL future PN contributions locally',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                new_canonical_neurons=0,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnElectricalOutputBrain):raise ValueError('Wrong additional-output backend')

    def state_dict(self):out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_output_pairs_replaced=466,PN_output_pairs_legacy=629,
            PN_additional_outputs_enabled=self.hybrid.pn_online_manifest['electrical_outputs']['enabled']);return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_electrical_output_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Output source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_electrical_output_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnElectricalOutputBrain)

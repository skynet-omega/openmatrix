"""Persist a partial PN input extension without changing its parent archive."""
from pathlib import Path
import copy
from pn_online_cns_session import PnOnlineSession,SOURCES as PARENT_SOURCES
from pn_cholinergic_cns_brain import GpuPnCholinergicBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_cholinergic_partial_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_cholinergic_rate_prior.py','pn_cholinergic_online_source.py','pn_cholinergic_cns_brain.py','pn_cholinergic_cns_session.py')

class PnCholinergicSession(PnOnlineSession):
    @classmethod
    def from_checkpoint(cls,path,spec,*,connected=True,coupling_ns=62500):
        path=Path(path).resolve();parent=PnOnlineSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            oldsource=parent.hybrid._online_source.state_dict()
            obj.hybrid=GpuPnCholinergicBrain.adopt(parent.hybrid,spec,connected=connected,coupling_ns=coupling_ns)
            from kcgamma_regional_brain import _record_hash
            if _record_hash(oldsource)!=_record_hash(obj.hybrid._online_source.state_dict()['base']):raise ValueError('Input extension changed inherited PN/Ca/tails')
            closure=require_covered_dependencies(ROOT/'src','pn_cholinergic_cns_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config);obj.config.update(candidate='PN_CHOLINERGIC_PARTIAL_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_cholinergic_cns_session.py',static_local_imports=closure),PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('ACh adoption changed body/pending input')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Enable217localAChpair priors on continuing finePN',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                new_canonical_neurons=0,additional_receptor_coordinates=434,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnCholinergicBrain):raise ValueError('Wrong PN/ACh backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_ACh_pairs=217,
            PN_ACh_current_connected=self.hybrid.pn_online_manifest['cholinergic']['connected'],PN_nonORN_inputs_pending=202);return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_cholinergic_cns_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_cholinergic_cns_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnCholinergicBrain)

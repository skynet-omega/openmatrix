"""Candidate CNS containing every originally nonzero PN input route."""
from pathlib import Path
import copy
from pn_apl_graded_session import PnAplGradedSession,SOURCES as PARENT_SOURCES
from pn_inhibitory_closure_brain import GpuPnInhibitoryClosureBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_inhibitory_closure_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_delayed_cascade_receptor.py','pn_inhibitory_closure_source.py','pn_inhibitory_closure_brain.py','pn_inhibitory_closure_session.py')


class PnInhibitoryClosureSession(PnAplGradedSession):
    @classmethod
    def from_checkpoint(cls,path,spec,*,mode,connected=True,coupling_ns=15625):
        path=Path(path).resolve();parent=PnAplGradedSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuPnInhibitoryClosureBrain.adopt(parent.hybrid,spec,mode=mode,connected=connected,coupling_ns=coupling_ns)
            closure=require_covered_dependencies(ROOT/'src','pn_inhibitory_closure_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config);obj.config.update(candidate='PN_INHIBITORY_CLOSURE_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_inhibitory_closure_session.py',static_local_imports=closure),PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Inhibitory closure activation changed body/pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Add explicit provisional GABA/Glu local closures',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                previous_PN_outputs_already_withdrawn=True,old_PN_q_diagnostic_only=True,new_canonical_neurons=0,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnInhibitoryClosureBrain):raise ValueError('Wrong inhibitory closure backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();m=self.hybrid.pn_online_manifest['inhibitory_closure']
        out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_inhibitory_closure_mode=m['mode'],PN_inhibitory_closure_connected=m['connected'],
            PN_input_pairs=450 if m['connected'] else 292,PN_nonORN_inputs_pending=43 if m['connected'] else 201,
            PN_nonzero_input_routes_complete=m['connected'],PN_native_input_efficacies_identified=False)
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_inhibitory_closure_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Inhibitory closure source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_inhibitory_closure_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnInhibitoryClosureBrain)

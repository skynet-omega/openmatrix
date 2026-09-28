"""Persist an explicitly selected Krylov solver without altering PN biology."""
from pathlib import Path
import copy
from pn_inhibitory_closure_session import PnInhibitoryClosureSession,SOURCES as PARENT_SOURCES
from pn_mass_krylov_brain import GpuPnMassKrylovBrain
from pn_mass_krylov_backend import POLICY
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_mass_krylov_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_mass_krylov_backend.py','pn_mass_krylov_brain.py','pn_mass_krylov_session.py')

class PnMassKrylovSession(PnInhibitoryClosureSession):
    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=PnInhibitoryClosureSession.load(path)
        return cls.adopt(parent,parent_checkpoint=path)

    @classmethod
    def adopt(cls,parent,*,parent_checkpoint):
        if type(parent) is not PnInhibitoryClosureSession:raise ValueError('Exact173 session required')
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuPnMassKrylovBrain.adopt(parent.hybrid)
            closure=require_covered_dependencies(ROOT/'src','pn_mass_krylov_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config);obj.config.update(candidate='PN_MASS_KRYLOV_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_mass_krylov_session.py',static_local_imports=closure))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Solver migration changed body or pending inputs')
            path=Path(parent_checkpoint).resolve()
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Replace full-mass linear solver only',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,physical_model_and_histories_preserved=True,solver_policy=POLICY,new_canonical_neurons=0)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnMassKrylovBrain):raise ValueError('Wrong numerical backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_linear_solver_policy=POLICY);return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_mass_krylov_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Krylov runtime source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_mass_krylov_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnMassKrylovBrain)

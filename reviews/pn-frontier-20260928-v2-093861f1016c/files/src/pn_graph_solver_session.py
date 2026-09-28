"""Save the same179 physical history with an explicitly versioned factorization."""
from pathlib import Path
import copy
from orn_peripheral_terminal_session import OrnPeripheralTerminalSession,SOURCES as PARENT_SOURCES
from pn_graph_solver_brain import GpuPnGraphSolverBrain
from pn_graph_elimination_backend import POLICY
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SOURCES=tuple(PARENT_SOURCES)+('pn_graph_elimination_backend.py','pn_graph_solver_brain.py','pn_graph_solver_session.py')
ENTRYPOINT='pn_graph_solver_session.py'


class PnGraphSolverSession(OrnPeripheralTerminalSession):
    SCHEMA='matrix_pn_graph_solver_session_v1'
    BRAIN=GpuPnGraphSolverBrain

    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=OrnPeripheralTerminalSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded);obj.hybrid=cls.BRAIN.adopt(parent.hybrid)
            closure=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA,electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=closure))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Graph adoption changed body or pending input')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Replace linear factorization only',parent_checkpoint=str(path),
                parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                all_neural_and_chemical_history_preserved=True,new_canonical_neurons=0)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),PN_graph_solver_policy=POLICY);return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Graph source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)

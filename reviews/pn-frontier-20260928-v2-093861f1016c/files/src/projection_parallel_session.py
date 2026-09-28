"""Version an execution-only CNS improvement; retain the original source archive."""
from pathlib import Path
import copy
from olfactory_endogenous_session import OlfactoryEndogenousSession,SOURCES as PARENT_SOURCES
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session
from projection_parallel_brain import GpuProjectionParallelBrain

SCHEMA='matrix_projection_parallel_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('kc_projection_parallel.py','projection_parallel_brain.py','projection_parallel_session.py')


class ProjectionParallelSession(OlfactoryEndogenousSession):
    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=OlfactoryEndogenousSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            obj.hybrid=GpuProjectionParallelBrain.adopt(parent.hybrid)
            closure=require_covered_dependencies(ROOT/'src','projection_parallel_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate='PROJECTION_PARALLEL_CNS_v1',
                electrical_backend_identity=obj.hybrid.backend_identity(),projection_threads=obj.hybrid.THREADS,
                runtime_source_contract=dict(entrypoint='projection_parallel_session.py',static_local_imports=closure))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Execution adoption changed organism state')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Parallelize independent KC input assembly; preserve APL accumulation order',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,neural_equations_changed=False,new_state_coordinates=0,
                biological_validation=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._validate();obj._validate_pending();obj._validate_afferent_pending();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuProjectionParallelBrain) or self.config['projection_threads']!=self.hybrid.THREADS:
            raise ValueError('Parallel execution policy differs')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),
            neural_equations_changed=False,projection_parallel_enabled=True,projection_threads=self.hybrid.THREADS)
        return out

    def save(self,path):
        closure=require_covered_dependencies(ROOT/'src','projection_parallel_session.py',SOURCES)
        if closure!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Source import contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','projection_parallel_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuProjectionParallelBrain)

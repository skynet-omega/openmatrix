"""Versioned mass/fine CNS branches for a bounded ORN terminal approximation."""
from pathlib import Path
import copy
from pn_mass_krylov_session import PnMassKrylovSession,SOURCES as PARENT_SOURCES
from pn_inhibitory_closure_session import PnInhibitoryClosureSession
from orn_peripheral_terminal_brain import GpuOrnPeripheralTerminalBrain,GpuOrnPeripheralTerminalFineBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SOURCES=tuple(PARENT_SOURCES)+('orn_peripheral_terminal.py','orn_peripheral_terminal_brain.py','orn_peripheral_terminal_session.py')
ENTRYPOINT='orn_peripheral_terminal_session.py'


class OrnPeripheralTerminalSessionMixin:
    @classmethod
    def from_checkpoint(cls,path,*,recurrent_connected=True):
        path=Path(path).resolve();parent=cls.PARENT.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'}
            before=_fingerprints(parent,excluded)
            obj.hybrid=cls.BRAIN.adopt(parent.hybrid,recurrent_connected=recurrent_connected)
            closure=require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate=cls.SCHEMA,electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint=ENTRYPOINT,static_local_imports=closure))
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('ORN migration changed body or pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Separate peripheral forcing and bounded terminal input modulation',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),
                time_ns=obj.time_ns,preserved_state_checks=checks,new_canonical_neurons=0,
                biological_validation=False,old_ORN_target_replaced_once=True)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if type(self.hybrid) is not self.BRAIN:raise ValueError('Wrong ORN terminal backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=self.SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields()
        out.update(schema=self.SCHEMA,runtime_source_files=len(SOURCES),
            ORN_peripheral_terminal_policy=self.hybrid.pn_online_manifest['orn_peripheral_terminal']['policy'],
            ORN_recurrent_connected=self.hybrid.pn_online_manifest['orn_peripheral_terminal']['recurrent_connected'])
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('ORN runtime source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src',ENTRYPOINT,SOURCES)
        return load_session(path,cls,cls.SCHEMA,SOURCES,cls.BRAIN)


class OrnPeripheralTerminalSession(OrnPeripheralTerminalSessionMixin,PnMassKrylovSession):
    SCHEMA='matrix_orn_peripheral_terminal_mass_session_v1'
    PARENT=PnMassKrylovSession
    BRAIN=GpuOrnPeripheralTerminalBrain


class OrnPeripheralTerminalFineSession(OrnPeripheralTerminalSessionMixin,PnInhibitoryClosureSession):
    SCHEMA='matrix_orn_peripheral_terminal_fine_session_v1'
    PARENT=PnInhibitoryClosureSession
    BRAIN=GpuOrnPeripheralTerminalFineBrain

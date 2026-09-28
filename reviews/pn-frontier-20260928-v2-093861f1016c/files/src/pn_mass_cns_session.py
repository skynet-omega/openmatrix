"""Save/reload the continuing CNS and reduced PN without fine-volume assembly."""
from pathlib import Path
import copy
from pn_general_output_session import PnGeneralOutputSession,SOURCES as PARENT_SOURCES
from pn_mass_cns_brain import GpuPnMassBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_mass_partial_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_mass_backend.py','pn_mass_coupled_step.py','pn_mass_calcium.py',
    'pn_mass_source_bridge.py','pn_mass_runtime.py','pn_mass_cns_brain.py','pn_mass_cns_session.py')


class PnMassSession(PnGeneralOutputSession):
    @classmethod
    def from_checkpoint(cls,path,contract,*,coupling_ns=15625):
        path=Path(path).resolve();parent=PnGeneralOutputSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuPnMassBrain.adopt(parent.hybrid,contract,coupling_ns=coupling_ns)
            closure=require_covered_dependencies(ROOT/'src','pn_mass_cns_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate='PN_MASS_PARTIAL_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_mass_cns_session.py',static_local_imports=closure),PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Mass migration changed body or pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Project only passive fine volume; retain exterior and every active mechanism',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                new_canonical_neurons=0,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnMassBrain):raise ValueError('Wrong mass CNS backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();pn=self.hybrid._online_source.pn
        out.pop('fine_PN_time_ns',None)
        out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_time_ns=pn.time_ns,
            PN_computational_coordinates=len(pn.voltage),PN_exterior_coordinates=178818,PN_internal_charge_coordinates=20,PN_model_level='working_full_mass')
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_mass_cns_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:
            raise ValueError('Mass source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_mass_cns_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnMassBrain)

"""Continuing CNS with one explicit graded APL feedback route into PN10208."""
from pathlib import Path
import json,copy
from pn_general_output_session import PnGeneralOutputSession
from pn_mass_cns_session import PnMassSession,SOURCES as PARENT_SOURCES
from pn_apl_graded_brain import GpuPnAplGradedBrain,AplPnCnsPorts
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_apl_graded_partial_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_apl_graded_receptor.py','pn_apl_graded_source.py','pn_apl_graded_brain.py','pn_apl_graded_session.py')


class PnAplGradedSession(PnGeneralOutputSession):
    @classmethod
    def from_checkpoint(cls,path,spec,*,connected=True,coupling_ns=15625):
        path=Path(path).resolve();schema=json.loads((path/'manifest.json').read_text())['schema']
        family={'matrix_pn_general_output_session_v1':PnGeneralOutputSession,'matrix_pn_mass_partial_session_v1':PnMassSession}
        if schema not in family:raise ValueError('Qualified PN parent family required')
        parent=family[schema].load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuPnAplGradedBrain.adopt(parent.hybrid,spec,connected=connected,coupling_ns=coupling_ns)
            closure=require_covered_dependencies(ROOT/'src','pn_apl_graded_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate='PN_APL_GRADED_PARTIAL_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_apl_graded_session.py',static_local_imports=closure),PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('APL route activation changed body or pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Add graded regional APL receptor on native PN membrane contacts',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,preserved_state_checks=checks,
                previous_PN_outputs_already_withdrawn=True,old_PN_q_diagnostic_only=True,new_canonical_neurons=0,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def pn_ports(self,pn_id=10208):return AplPnCnsPorts(self.hybrid,pn_id)

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnAplGradedBrain):raise ValueError('Wrong APL feedback backend')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();connected=self.hybrid.pn_online_manifest['apl_feedback']['connected'];mass='mass' in self.hybrid.pn_online_manifest
        out.pop('fine_PN_time_ns',None)
        out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),PN_APL_graded_connected=connected,PN_input_pairs=292 if connected else 291,
            PN_nonORN_inputs_pending=201 if connected else 202,PN_time_ns=self.hybrid._online_source.pn.time_ns,
            PN_model_level='working_full_mass' if mass else 'fine',PN_APL_pair_full_scale_nS=self.hybrid._online_source.apl_spec['pair_full_scale_nS'])
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_apl_graded_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('APL source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_apl_graded_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnAplGradedBrain)

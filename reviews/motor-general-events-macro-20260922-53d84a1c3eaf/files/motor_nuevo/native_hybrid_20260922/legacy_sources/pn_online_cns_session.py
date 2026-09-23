"""Persist the continuing CNS together with its partial live fine-PN source.

Only74ORN inputs and100KCgamma outputs use the fine PN; remaining legacy
routes are retained explicitly. This is not a full PN membrane replacement.
"""
from pathlib import Path
import copy
from pn_port_session import PnPortSession,SOURCES as PARENT_SOURCES
from pn_online_cns_brain import GpuPnOnlineBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session

SCHEMA='matrix_pn_online_partial_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('compensated_csr.py', 'dm1_kc_event_receiver.py', 'dm1_kc_replacement_receiver.py', 'dm1_native_passive.py', 'dm1_pn_membrane.py', 'neck_connected_domain.py', 'neck_distributed_conductor.py', 'neck_dynamic_handoff.py', 'neck_gpu_multigrid.py', 'neck_gpu_transient.py', 'neck_joint_csr.py', 'neck_joint_multigrid.py', 'neck_joint_transient.py', 'neck_multiport.py', 'neck_selected_conductor.py', 'neck_surface_ports.py', 'pn_calcium_coupled.py', 'pn_calcium_port.py', 'pn_calcium_release.py', 'pn_cns_orn_stages.py', 'pn_coupled_ionic.py', 'pn_fine_ionic.py', 'pn_local_conductance.py', 'pn_online_cns_brain.py', 'pn_online_cns_session.py', 'pn_online_orn_source.py', 'pn_prepared_kc_boundary.py', 'pn_reference_runtime.py', 'pn_site_kc_boundary.py', 'pn_spatial_orn.py', 'synaptic_event_filter.py')

class PnOnlineSession(PnPortSession):
    @classmethod
    def from_checkpoint(cls,path,preparation,*,coupling_ns=62500,orn_connected=True,context=None):
        path=Path(path).resolve();parent=PnPortSession.load(path);obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention','hybrid'};before=_fingerprints(parent,excluded)
            obj.hybrid=GpuPnOnlineBrain.adopt(parent.hybrid,preparation,coupling_ns=coupling_ns,orn_connected=orn_connected,context=context)
            closure=require_covered_dependencies(ROOT/'src','pn_online_cns_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate='PN_ONLINE_PARTIAL_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_online_cns_session.py',static_local_imports=closure),
                PN_online_scope=obj.hybrid.pn_online_manifest['scope'])
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Online adoption changed body or pending inputs')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),operation='Partial live PN10208/Ca source to100KCgamma with canonical ORN filters',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,new_canonical_neurons=0,native_PN_geometry_replaced=False,
                source_feedback=True,full_PN_replacement=False,biological_validation=False)
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnOnlineBrain):raise ValueError('Wrong online backend')
        self.hybrid.validate_online()

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),neural_equations_changed=True,
            PN_online_partial_replacement=True,source_feedback=True,native_PN_geometry_replaced=False,
            fine_PN_time_ns=self.hybrid._online_source.pn.time_ns,coupling_ns=self.hybrid.pn_online_manifest['coupling_ns'])
        return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_online_cns_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_online_cns_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnOnlineBrain)

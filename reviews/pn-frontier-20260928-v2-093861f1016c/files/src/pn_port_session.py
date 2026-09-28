"""Version the live PN input/output readout in a complete CNS checkpoint.

This adds a read-only integration API and archives its exact implementation.
It does not replace the PN membrane or any synaptic contribution.
"""
from pathlib import Path
import copy
from projection_parallel_session import ProjectionParallelSession,SOURCES as PARENT_SOURCES
from projection_parallel_brain import GpuProjectionParallelBrain
from kc_audited_session import ROOT,_fingerprints,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session
from pn_cns_ports import PnCnsPorts

SCHEMA='matrix_pn_port_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_cns_ports.py','pn_port_session.py')


class PnPortSession(ProjectionParallelSession):
    @classmethod
    def from_checkpoint(cls,path):
        path=Path(path).resolve();parent=ProjectionParallelSession.load(path)
        obj=cls();obj.__dict__.update(parent.__dict__)
        try:
            excluded={'schema','config','source_identity','intervention'}
            before=_fingerprints(parent,excluded)
            ports=PnCnsPorts(obj.hybrid,10208)
            if len(ports.source_ids)!=493 or len(ports.target_ids)!=1095:raise ValueError('PN10208 canonical ports incomplete')
            closure=require_covered_dependencies(ROOT/'src','pn_port_session.py',SOURCES)
            obj.config=copy.deepcopy(parent.config)
            obj.config.update(candidate='PN_CANONICAL_PORTS_v1',pn_ports_readout=True,
                runtime_source_contract=dict(entrypoint='pn_port_session.py',static_local_imports=closure),
                pn_port_scope='Read existing canonical source/filter states; no fine PN replacement or invented conductance.')
            after=_fingerprints(obj,excluded);checks={k:v==after[k] for k,v in before.items()}
            if not all(checks.values()):raise ValueError('Readout adoption changed CNS/body history')
            obj.intervention=dict(previous_intervention=copy.deepcopy(parent.intervention),
                operation='Expose read-only PN canonical ports and archive their implementation',
                parent_checkpoint=str(path),parent_manifest_sha256=sha256(path/'manifest.json'),time_ns=obj.time_ns,
                preserved_state_checks=checks,neural_equations_changed=False,new_state_coordinates=0,
                native_PN_geometry_replaced=False,biological_validation=False)
            obj.source_identity={name:sha256(ROOT/'src'/name) for name in SOURCES}
            obj._validate();return obj
        except BaseException:obj.close();raise

    def pn_ports(self,pn_id=10208):return PnCnsPorts(self.hybrid,pn_id)

    def _validate(self):
        super()._validate()
        if self.config.get('pn_ports_readout') is not True:raise ValueError('Missing PN readout policy')

    def state_dict(self):
        out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),
            pn_ports_readout=True,native_PN_geometry_replaced=False,neural_equations_changed=False)
        return out

    def save(self,path):
        closure=require_covered_dependencies(ROOT/'src','pn_port_session.py',SOURCES)
        if closure!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Source closure changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_port_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuProjectionParallelBrain)

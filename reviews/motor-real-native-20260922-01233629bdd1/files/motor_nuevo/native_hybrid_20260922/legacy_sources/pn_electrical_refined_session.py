"""Explicit temporal refinement of the same partial PN electrical outputs.

The153-source trial is preserved. No geometry, efficacy, gate or chemistry
parameter changes; this version admits31.25/15.625us coupling for the observed
single-cell event-timing discrepancy instead of weakening its acceptance test.
"""
import copy
from pathlib import Path
from pn_electrical_output_session import PnElectricalOutputSession,SOURCES as PARENT_SOURCES
from pn_electrical_output_brain import GpuPnElectricalOutputBrain
from kc_audited_session import ROOT,sha256,require_covered_dependencies
from kc_session_storage import save_session,load_session
from kcgamma_regional_brain import _record_hash

SCHEMA='matrix_pn_electrical_refined_session_v1'
SOURCES=tuple(PARENT_SOURCES)+('pn_electrical_refined_session.py',)
SUPPORTED_NS=(62500,31250,15625)

class GpuPnElectricalRefinedBrain(GpuPnElectricalOutputBrain):
    SCHEMA='matrix_pn_electrical_refined_brain_v1'

    @classmethod
    def reconfigure(cls,base,coupling_ns):
        if type(base) is not GpuPnElectricalOutputBrain or type(coupling_ns) is not int or coupling_ns not in SUPPORTED_NS:
            raise ValueError('Explicit finer coupling and exact153-output parent required')
        before=base.state_dict();obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        m=copy.deepcopy(base.pn_online_manifest);m['coupling_ns']=coupling_ns
        m['electrical_outputs']['temporal_refinement']=dict(previous_coupling_ns=base.pn_online_manifest['coupling_ns'],coupling_ns=coupling_ns,
            unchanged='Geometry, receptor scales, membrane, gates, chemistry, sources, targets, all nonzero histories; only coupling interval changes.')
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        # Copied views/closures otherwise retain the previous object's clock
        # and state array when the continuing GPU solver replaces that array.
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for field in ('schema','pn_online_manifest'):before.pop(field);after.pop(field)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Temporal refinement changed state')
        return obj

    def validate_refinement(self):
        self.validate_online();m=self.pn_online_manifest
        r=m['electrical_outputs'].get('temporal_refinement',{})
        if m['coupling_ns'] not in SUPPORTED_NS or r.get('coupling_ns')!=m['coupling_ns']:raise ValueError('Incomplete declared temporal refinement')

    @classmethod
    def from_state(cls,brain,saved):
        obj=super().from_state(brain,saved);obj.validate_refinement();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnElectricalOutputBrain.backend_identity();out['PN_output_refinement']='Same153-source mechanism; explicit62500/31250/15625ns coupling. No efficacy or anatomy adjustment.';return out

class PnElectricalRefinedSession(PnElectricalOutputSession):
    @classmethod
    def from_checkpoint(cls,path,spec,*,enabled=True,coupling_ns=31250):
        if type(coupling_ns) is not int or coupling_ns not in SUPPORTED_NS:raise ValueError('Unsupported refined coupling')
        base=PnElectricalOutputSession.from_checkpoint(path,spec,enabled=enabled,coupling_ns=62500)
        obj=cls();obj.__dict__.update(base.__dict__)
        try:
            obj.hybrid=GpuPnElectricalRefinedBrain.reconfigure(base.hybrid,coupling_ns)
            obj.config=copy.deepcopy(base.config);closure=require_covered_dependencies(ROOT/'src','pn_electrical_refined_session.py',SOURCES)
            obj.config.update(candidate='PN_ELECTRICAL_OUTPUT_REFINED_v1',electrical_backend_identity=obj.hybrid.backend_identity(),
                runtime_source_contract=dict(entrypoint='pn_electrical_refined_session.py',static_local_imports=closure))
            obj.intervention=copy.deepcopy(base.intervention);obj.intervention['temporal_refinement_ns']=coupling_ns
            obj.source_identity={n:sha256(ROOT/'src'/n) for n in SOURCES};obj._validate();return obj
        except BaseException:obj.close();raise

    def _validate(self):
        super()._validate()
        if not isinstance(self.hybrid,GpuPnElectricalRefinedBrain):raise ValueError('Wrong refined backend')
        self.hybrid.validate_refinement()

    def state_dict(self):out=super().state_dict();out['schema']=SCHEMA;return out

    def _manifest_fields(self):
        out=super()._manifest_fields();out.update(schema=SCHEMA,runtime_source_files=len(SOURCES),temporal_refinement_ns=self.hybrid.pn_online_manifest['coupling_ns']);return out

    def save(self,path):
        if require_covered_dependencies(ROOT/'src','pn_electrical_refined_session.py',SOURCES)!=self.config['runtime_source_contract']['static_local_imports']:raise ValueError('Refined source contract changed')
        return save_session(self,path,SOURCES)

    @classmethod
    def load(cls,path):
        require_covered_dependencies(ROOT/'src','pn_electrical_refined_session.py',SOURCES)
        return load_session(path,cls,SCHEMA,SOURCES,GpuPnElectricalRefinedBrain)

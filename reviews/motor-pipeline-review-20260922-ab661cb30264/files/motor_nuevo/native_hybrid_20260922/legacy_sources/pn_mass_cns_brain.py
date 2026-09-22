"""CNS branch with the same PN mechanism on guarded generalized coordinates."""
import copy
from pn_general_output_brain import GpuPnGeneralOutputBrain
from pn_cholinergic_cns_brain import KEYS
from projection_parallel_brain import GpuProjectionParallelBrain
from pn_mass_runtime import build_source
from pn_mass_source_bridge import MassPnSourceBridge
from kcgamma_regional_brain import _record_hash


class GpuPnMassBrain(GpuPnGeneralOutputBrain):
    SCHEMA='matrix_pn_mass_partial_brain_v1'

    @classmethod
    def adopt(cls,parent,contract,*,coupling_ns=15625):
        if type(parent) is not GpuPnGeneralOutputBrain or coupling_ns not in (31250,15625):
            raise ValueError('Exact158 parent and explicit comparison step required')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        if _record_hash(before['pn_online_state'])!=contract['parent_source_sha256']:
            raise ValueError('Projection seed does not belong to this live PN source')
        m=copy.deepcopy(parent.pn_online_manifest);m['mass']=copy.deepcopy(contract)
        if obj.time_ns!=contract['CNS_origin_ns']:
            raise ValueError('Mass projection is not at this CNS checkpoint')
        obj._online_source=build_source(contract,m);obj._reference_context=None
        m['adoption_time_ns']=contract['CNS_origin_ns']
        m['coupling_ns']=coupling_ns;m['electrical_outputs']['temporal_refinement']['coupling_ns']=coupling_ns
        m['scope']='Live291inputs and1095local outputs; PN exterior178818 plus20internal charge coordinates.202inputs absent. Same provisional biological mechanisms; finite reduction qualification only.'
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for key in ('schema','pn_online_manifest','pn_online_state'):before.pop(key);after.pop(key)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Mass migration changed inherited CNS state')
        return obj

    def validate_online(self):
        super().validate_online()
        if not isinstance(self._online_source.pn,MassPnSourceBridge):raise ValueError('Mass branch requires explicit PN bridge')
        m=self.pn_online_manifest['mass'];pn=self._online_source.pn
        if (pn.identity!=m['pn_identity'] or len(pn.voltage)!=m['exterior_coordinates']+m['internal_coordinates']
            or self._online_source.source_origin_ns!=m['PN_origin_ns'] or self._online_source.cns_origin_ns!=m['CNS_origin_ns']):
            raise ValueError('Mass coordinate or origin contract changed')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete mass CNS state')
        parent={k:v for k,v in saved.items() if k not in ('pn_online_manifest','pn_online_state')}
        parent['schema']=GpuProjectionParallelBrain.SCHEMA
        base=GpuProjectionParallelBrain.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.pn_online_manifest=copy.deepcopy(saved['pn_online_manifest'])
        obj._online_source=build_source(obj.pn_online_manifest['mass'],obj.pn_online_manifest)
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._reference_context=None
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnGeneralOutputBrain.backend_identity()
        out['PN_mass']='CPU full Galerkin mass;178818physical exterior plus20internal charge coordinates. Fine source remains reference; no biological parameter fitted.'
        return out

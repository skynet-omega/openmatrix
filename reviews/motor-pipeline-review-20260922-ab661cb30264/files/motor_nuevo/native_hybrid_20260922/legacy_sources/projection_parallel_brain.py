"""Execution-only KC input assembly on the unchanged continuing CNS state."""
import numba
import numpy as np
from olfactory_endogenous_brain import GpuOlfactoryEndogenousBrain,KEYS
from kc_electrical_scales_brain import RECEPTOR_START,PARENT_STATE_SIZE
from kc_projection_parallel import projection_conductances_parallel
from kcgamma_regional_brain import _record_hash
from kc_visual_ports import VisualComponents


class GpuProjectionParallelBrain(GpuOlfactoryEndogenousBrain):
    SCHEMA='matrix_projection_parallel_brain_fp64_cuda_v1'
    THREADS=14

    @classmethod
    def adopt(cls,parent):
        if type(parent) is not GpuOlfactoryEndogenousBrain:raise ValueError('Exact endogenous CNS parent required')
        before=parent.state_dict();before.pop('schema')
        obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__);obj._rebind_continuing_inputs()
        after=obj.state_dict();after.pop('schema')
        if _record_hash(before)!=_record_hash(after):raise ValueError('Execution adoption changed neural state')
        return obj

    def _conductance_values(self,caps=None):
        c=self._dynamic_cache;y=self.state;a=self.routes.arrays;b=self.brain
        _,edge=self.routes.outgoing_release(self.kc_apl_dynamic_state['apl_transmission'])
        sax=y[self.regional_parent_state_size:self.retinal_parent_state_size].reshape(3,-1)[2]
        previous=numba.get_num_threads()
        try:
            numba.set_num_threads(self.THREADS)
            return projection_conductances_parallel(c['rows'],c['local_ptr'],c['mode'],c['fraction'],c['slot'],c['pn_slot'],
                c['apl_edge_slot'],b.W.indptr,b.W.indices,self.weights64,y[self.transmission_start:self.inherited_state_size],
                sax,self.caps if caps is None else caps,self.visual_mask,self.visual_output_connected,y[RECEPTOR_START:PARENT_STATE_SIZE],edge,
                self._dynamic_scales,self.kc_apl_dynamic_manifest['group'],c['pair_route_slot'],a['region_fractions'],c['route_apl'],
                self.routes.shape[1],self._pn_gain_local,self.kc_electrical_scales_state['filters'][1],self._apl_gain_local)
        finally:numba.set_num_threads(previous)

    def conductance_components(self):
        # The six numerical arrays also carry the inherited visual spatial labels.
        values=self._conductance_values();ge=np.zeros_like(values[0]);gi=ge.copy()
        for name in self.kc_visual_manifest['groups']:
            a,b=self._visual_pathways.visual_conductances(name);ge+=a;gi+=b
        return VisualComponents(values,ge,gi,self._visual_gamma_mask)

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete parallel projection CNS state')
        parent=dict(saved);parent['schema']=GpuOlfactoryEndogenousBrain.SCHEMA
        return cls.adopt(GpuOlfactoryEndogenousBrain.from_state(brain,parent))

    @staticmethod
    def backend_identity():
        out=GpuOlfactoryEndogenousBrain.backend_identity()
        out['projection_assembly']='Independent KC rows in14 CPU threads; every APL regional sum serial in original order; FP64 without fastmath; thread setting restored after call.'
        return out

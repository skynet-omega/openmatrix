"""Partial fine-PN output replacement for365additional KC and one regional APL.

Selected legacy conductances are replaced locally; the shared PN transmission
and629general consumers remain untouched. APL updates its actual regional age.
"""
import copy
import numpy as np
from pn_cholinergic_cns_brain import GpuPnCholinergicBrain,KEYS
from pn_electrical_output_source import ElectricalOutputSource
from kc_visual_ports import VisualComponents
from kcgamma_regional_brain import _record_hash

class GpuPnElectricalOutputBrain(GpuPnCholinergicBrain):
    SCHEMA='matrix_pn_electrical_output_partial_brain_v1'

    @classmethod
    def adopt(cls,parent,spec,*,enabled=True,coupling_ns=62500):
        if type(parent) is not GpuPnCholinergicBrain or type(enabled) is not bool or coupling_ns not in (125000,62500):raise ValueError('Exact149parent and explicit output intervention required')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._online_source=ElectricalOutputSource.adopt(parent._online_source,spec)
        m=copy.deepcopy(parent.pn_online_manifest);m['coupling_ns']=coupling_ns
        m['electrical_outputs']=dict(spec=copy.deepcopy(spec),enabled=enabled,adoption_time_ns=obj.time_ns,
            new_output_pairs=366,total_output_pairs=466,remaining_output_pairs=629,
            scope='365additionalKCand1APL; local release normalized by its pulse area, inherited class conductance-per-weight-Hz scale retained as provisional. GenericPNtransmission retained for629others.')
        m['scope']='Live291inputs to finePN/Ca,100KCgamma plus365KCand1APL outputs;202inputs absent,629outputs andPNq legacy. Additional-output scale is an inherited provisional interface, not measured efficacy.'
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m;obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for field in ('schema','pn_online_manifest','pn_online_state'):before.pop(field);after.pop(field)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Output adoption changed inherited CNS state')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes()
        if not isinstance(self._online_source,ElectricalOutputSource):return
        h=self;port=h._online_source.extra_output;c=h._dynamic_cache;W=h.brain.W
        targets=np.searchsorted(h.brain.node_ids,port.route.target_ids);j=np.searchsorted(c['rows'],targets)
        np.testing.assert_array_equal(h.brain.node_ids[targets],port.route.target_ids);np.testing.assert_array_equal(c['rows'][j],targets)
        groups=h.kc_apl_dynamic_manifest['group'][j]
        if len(j)!=366 or np.sum(groups<2)!=365 or np.sum(groups==3)!=1 or np.any(groups==2):raise ValueError('Exact365additionalKCand1APL consumer selection required')
        positions=[];local=[];pn=h._online_ports.pn_row
        for row,index in zip(targets,j):
            edges=np.arange(W.indptr[row],W.indptr[row+1]);pos=edges[W.indices[edges]==pn]
            if len(pos)!=1:raise ValueError('Missing unique PN consumer edge')
            positions.append(pos[0]);local.append(c['local_ptr'][index]+pos[0]-W.indptr[row])
        positions=np.asarray(positions);local=np.asarray(local)
        if np.any(c['mode'][local]!=0) or np.any(c['pn_slot'][local]!=-1) or np.any(c['apl_edge_slot'][local]!=-1) or np.any(h.weights64[positions]<=0):raise ValueError('Unexpected specialized PN consumer')
        gain=h.weights64[positions]*h.caps[pn]*h._dynamic_scales[groups,0]
        np.testing.assert_array_equal(gain,port.legacy_gain)
        h._extra_output_slots=j;h._extra_group=groups
        a=np.flatnonzero(groups==3);route=c['pair_route_slot'][local[a]]
        h._extra_apl_local=int(a[0]);h._extra_apl_cell=int(c['route_apl'][route[0]])
        h._extra_apl_fraction=h.routes.arrays['region_fractions'][route[0]].copy()
        if not np.isclose(h._extra_apl_fraction.sum(),1.,atol=1e-12,rtol=0):raise ValueError('Incomplete APL regional allocation')

    def validate_online(self):
        super().validate_online()
        if isinstance(self._online_source,ElectricalOutputSource):
            self._online_source.additional_output_nS()
            if self._online_source.extra_output.origin_ns!=self.pn_online_manifest['electrical_outputs']['adoption_time_ns']:raise ValueError('Changed output activation origin')

    def conductance_components(self):
        old=super().conductance_components()
        if not hasattr(self,'_extra_output_slots') or not self.pn_online_manifest['electrical_outputs']['enabled']:return old
        port=self._online_source.extra_output
        inherited=self.state[self.transmission_start+self._online_ports.pn_row]*port.legacy_gain
        delta=self._online_source.additional_output_nS()-inherited
        values=list(old);values[0]=values[0].copy();values[2]=values[2].copy()
        values[0][self._extra_output_slots]+=delta
        values[2][self._extra_apl_cell]+=delta[self._extra_apl_local]*self._extra_apl_fraction
        if np.any(values[0]<0) or np.any(values[2]<0):raise ValueError('Negative receiver conductance after partial replacement')
        return VisualComponents(values,old.visual_ge,old.visual_gi,old.gamma_mask)

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete extended output state')
        parent=dict(saved);parent['schema']=GpuPnCholinergicBrain.SCHEMA;parent['pn_online_state']=saved['pn_online_state']['base']
        base=GpuPnCholinergicBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj._online_source=ElectricalOutputSource.adopt(base._online_source,obj.pn_online_manifest['electrical_outputs']['spec'])
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnCholinergicBrain.backend_identity();out['additional_PN_outputs']='365KCand1regionalAPL local replacement with legacy-tail preservation and declared release-area normalization;629general consumers remain legacy.';return out

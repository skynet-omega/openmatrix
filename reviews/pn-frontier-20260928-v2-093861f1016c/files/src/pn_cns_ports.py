"""Read the actual continuing CNS ports for a future fine-PN replacement.

No second neuron, source filter or membrane is created. Values named rate
are inherited rate proxies, not measured firing or receptor conductances.
Reading these ports does not couple the fine PN output into the CNS.
"""
import numpy as np


class PnCnsPorts:
    def __init__(self,hybrid,pn_id):
        self.hybrid=hybrid;b=hybrid.brain;ids=b.node_ids
        match=np.flatnonzero(ids==pn_id)
        if len(match)!=1:raise ValueError('PN must occur once in the canonical graph')
        self.pn_row=int(match[0]);self.pn_id=int(pn_id);W=b.W
        self.in_positions=np.arange(W.indptr[self.pn_row],W.indptr[self.pn_row+1],dtype=np.int64)
        self.source_rows=W.indices[self.in_positions].copy();self.source_ids=ids[self.source_rows].copy()
        self.out_positions=np.flatnonzero(W.indices==self.pn_row)
        self.target_rows=np.searchsorted(W.indptr,self.out_positions,side='right')-1
        self.target_ids=ids[self.target_rows].copy()
        if len(np.unique(self.source_ids))!=len(self.source_ids) or len(np.unique(self.target_ids))!=len(self.target_ids):
            raise ValueError('Canonical CSR must have one aggregate per pair')
        om=hybrid.olfactory_endogenous_manifest;self.orn_rows=om['source_rows'].copy();self.orn_ids=om['source_ids'].copy()
        if not np.isin(self.orn_rows,self.source_rows).all():raise ValueError('ORN source omitted from PN input row')
        pm=hybrid.kc_electrical_scales_manifest
        self.pn_filter_slot=int(np.flatnonzero(pm['source_ids']==pn_id)[0])
        self._graph=(W.indptr,W.indices,ids)

    def observe(self):
        h=self.hybrid;b=h.brain
        current=(b.W.indptr,b.W.indices,b.node_ids)
        if any(a is not c for a,c in zip(self._graph,current)):
            # Validating/saving a CNS can rewrap an identical CSR. Compare
            # anatomical values before rebinding; never accept a new graph.
            if any(not np.array_equal(a,c) for a,c in zip(self._graph,current)):
                raise ValueError('Graph anatomy changed during port capture')
            self._graph=current
        state=h.state;begin,end=h.parent_state_size,h.regional_parent_state_size
        filters=state[begin:end].reshape(4,-1)
        if filters.shape[1]!=len(self.orn_rows):raise ValueError('ORN filter layout changed')
        return dict(time_ns=h.time_ns,source_ids=self.source_ids.copy(),target_ids=self.target_ids.copy(),
            source_q=state[self.source_rows].copy(),
            source_legacy_rate_proxy_hz=(state[self.source_rows]*h.caps[self.source_rows]).copy(),
            source_base_transmission=state[h.transmission_start+self.source_rows].copy(),
            ORN_source_ids=self.orn_ids.copy(),ORN_filters=filters.copy(),
            PN_q=float(state[self.pn_row]),PN_base_transmission=float(state[h.transmission_start+self.pn_row]),
            PN_KC_filters=h.kc_electrical_scales_state['filters'][:,self.pn_filter_slot].copy(),
            APL_voltage_mV=h.kc_apl_dynamic_state['apl_voltage_mV'].copy(),
            APL_regional_transmission=h.kc_apl_dynamic_state['apl_transmission'].copy(),
            incoming_weights=h.weights64[self.in_positions].copy(),outgoing_weights=h.weights64[self.out_positions].copy())

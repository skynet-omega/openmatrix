"""Partial live fine-PN->100KC coupling within the canonical continuing CNS.

CNS advances with the beginning PN release, then finePN reads the interval's
canonical ORN filters. This partitioned coupling has its own timestep error.
The other419inputs remain on the legacy PN and995outputs retain legacy routes;
it is explicitly NOT a full replacement of the canonical PN membrane.
"""
import copy
import numpy as np
from projection_parallel_brain import GpuProjectionParallelBrain,KEYS as PARENT_KEYS
from pn_reference_runtime import online_source
from pn_cns_ports import PnCnsPorts
from kc_apl_dynamics import cascade
from kc_visual_ports import VisualComponents
from kcgamma_regional_brain import _record_hash

KEYS=PARENT_KEYS|{'pn_online_manifest','pn_online_state'}

class GpuPnOnlineBrain(GpuProjectionParallelBrain):
    SCHEMA='matrix_pn_online_partial_brain_v1'

    @classmethod
    def adopt(cls,parent,preparation,*,coupling_ns=62500,orn_connected=True,context=None):
        if type(parent) is not GpuProjectionParallelBrain or coupling_ns not in (125000,62500,31250,15625) or type(orn_connected) is not bool:
            raise ValueError('Exact parent, supported coupling and explicit ORN current switch required')
        before=parent.state_dict();before.pop('schema');obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._online_source,obj._reference_context=online_source(preparation,context=context)
        m=dict(preparation=copy.deepcopy(preparation),coupling_ns=coupling_ns,orn_connected=orn_connected,
            adoption_time_ns=obj.time_ns,parent_record_sha256=_record_hash(before),
            scope='Live fine-PN/Ca fed only by74 canonical ORN filters; post-handoff release replaces only100KCgamma paths.419otherinputs/995outputs andPNq remain on legacy paths.',
            full_PN_replacement=False,biological_validation=False,
            source_history='Single owner canonical ORN filters; independent prepared PN source origin explicitly declared.')
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m;obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict();after.pop('schema');after.pop('pn_online_manifest');after.pop('pn_online_state')
        if _record_hash(after)!=m['parent_record_sha256']:raise ValueError('Online source adoption changed inherited CNS fields')
        return obj

    def _bind_online_routes(self):
        source=self._online_source;pm=self.kc_electrical_scales_manifest;W=self.brain.W
        slots=np.flatnonzero(pm['source_ids']==10208)
        if len(slots)!=1:raise ValueError('Unique PN10208 source required')
        self._online_slot=int(slots[0]);row=int(pm['source_rows'][self._online_slot])
        p=pm['pn_positions'];selected=p[W.indices[p]==row];targets=np.searchsorted(W.indptr,selected,side='right')-1
        targets=targets[np.argsort(self.brain.node_ids[targets])]
        np.testing.assert_array_equal(self.brain.node_ids[targets],source.route.target_ids)
        if len(targets)!=100 or not self.kc_spatial_manifest['enabled'] or not self.kc_spatial_manifest['pn_enabled']:
            raise ValueError('Enabled100KCgamma route required')
        self._online_dynamic_slot=np.searchsorted(self._dynamic_cache['rows'],targets)
        np.testing.assert_array_equal(self._dynamic_cache['rows'][self._online_dynamic_slot],targets)
        self._online_ports=PnCnsPorts(self,10208)
        om=self.olfactory_endogenous_manifest;order=np.argsort(om['source_ids'])
        np.testing.assert_array_equal(om['source_ids'][order],source.allocation.source_ids)
        np.testing.assert_array_equal(self.caps[om['source_rows']][order],source.caps)

    def validate_online(self):
        m=self.pn_online_manifest
        if m['record_sha256']!=_record_hash(m):raise ValueError('Online policy changed')
        if self._online_source.time_ns!=self.time_ns:raise ValueError('CNS and source clocks disagree')
        if self._online_source.pn.time_ns-self._online_source.source_origin_ns!=self.time_ns-m['adoption_time_ns']:
            raise ValueError('Prepared PN clock offset changed')
        self._online_source.output_nS()

    def conductance_components(self):
        old=super().conductance_components()
        if not hasattr(self,'_online_source'):return old
        if self._online_source.time_ns!=self.time_ns:raise ValueError('Partial interval cannot expose source output')
        g=self._online_source.output_nS();values=list(old);values[0]=values[0].copy();values[4]=values[4].copy()
        values[0][self._online_dynamic_slot]+=g;values[4][self._online_dynamic_slot]+=g
        return VisualComponents(values,old.visual_ge,old.visual_gi,old.gamma_mask)

    def _restore_joint(self,parent_state,source_state,published):
        m=self.pn_online_manifest;source=self._online_source;context=self._reference_context
        source.load_state_dict(source_state)
        parent=dict(parent_state);parent['schema']=GpuProjectionParallelBrain.SCHEMA
        # The strict parent loader validates the published float32 view and
        # its clock before restoring detailed state. Restore both atomically.
        self.brain.time_ns=published[0];self.brain.rates[:]=published[1]
        base=GpuProjectionParallelBrain.from_state(self.brain,parent)
        self.__dict__.clear();self.__dict__.update(base.__dict__)
        self.pn_online_manifest=m;self._online_source=source;self._reference_context=context
        self._bind_online_routes();self._rebind_continuing_inputs();self.publish_rates();self.validate_online()

    def advance(self,dt_ns,drive,light):
        self._validated_inputs(dt_ns,drive,light);self.validate_online();left=dt_ns
        p=self.kc_electrical_scales_manifest['source']['PN_KCgamma'];m=self.pn_online_manifest
        while left:
            ns=min(left,m['coupling_ns']);before=super().state_dict();source_before=self._online_source.state_dict()
            published=(self.brain.time_ns,self.brain.rates.copy())
            first=self._online_ports.observe();old=self.kc_electrical_scales_state['filters'][:,self._online_slot].copy()
            try:
                super().advance(ns,drive,light)
                x,y=cascade(old[0],old[1],0.,p['cascade_fast_tau_s'],p['cascade_slow_tau_s'],ns*1e-9)
                self.kc_electrical_scales_state['filters'][:,self._online_slot]=[x,y]
                second=self._online_ports.observe()
                r=self._online_source.advance(ns,first,second,connected=m['orn_connected'],
                    rtol=1e-9,atol=5e-6,maxiter=220,max_newton=8,gate_atol=1e-12,stage_predictor='linear')
                if not r['accepted']:raise RuntimeError('FinePN step rejected: '+str(r.get('reason')))
                self.validate_online();self.validate_scales()
            except BaseException:
                self._restore_joint(before,source_before,published);raise
            left-=ns

    def state_dict(self):
        self.validate_online();out=super().state_dict();out['pn_online_manifest']=copy.deepcopy(self.pn_online_manifest)
        out['pn_online_state']=self._online_source.state_dict();return out

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete online PN/CNS state')
        parent={k:v for k,v in saved.items() if k not in ('pn_online_manifest','pn_online_state')};parent['schema']=GpuProjectionParallelBrain.SCHEMA
        base=GpuProjectionParallelBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        obj.pn_online_manifest=copy.deepcopy(saved['pn_online_manifest'])
        obj._online_source,obj._reference_context=online_source(obj.pn_online_manifest['preparation'])
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuProjectionParallelBrain.backend_identity();out['pn_online']='Prepared finePN/Ca coupled with canonical ORN filters and100KCgamma release; explicit partitioned coupling and remaining legacy PN paths.';return out

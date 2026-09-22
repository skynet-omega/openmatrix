"""Continuing CNS with explicit fast/slow inhibitory PN closure candidates."""
import copy
import numpy as np
from pn_apl_graded_brain import GpuPnAplGradedBrain
from pn_inhibitory_closure_source import InhibitoryClosureSource
from pn_cholinergic_cns_brain import KEYS
from kcgamma_regional_brain import _record_hash


class GpuPnInhibitoryClosureBrain(GpuPnAplGradedBrain):
    SCHEMA='matrix_pn_inhibitory_closure_brain_v1'

    @classmethod
    def adopt(cls,parent,spec,*,mode,connected,coupling_ns=15625):
        if type(parent) is not GpuPnAplGradedBrain or coupling_ns not in (31250,15625):raise ValueError('Exact APL parent and declared interval required')
        if parent.time_ns!=spec['origin_ns']:raise ValueError('New receptor activation must match CNS clock')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._online_source=InhibitoryClosureSource.adopt(parent._online_source,spec,mode=mode,connected=connected)
        if _record_hash(obj._online_source.state_dict()['base'])!=_record_hash(before['pn_online_state']):raise ValueError('Inhibitory adoption changed existing source histories')
        m=copy.deepcopy(parent.pn_online_manifest);m['coupling_ns']=coupling_ns;m['electrical_outputs']['temporal_refinement']['coupling_ns']=coupling_ns
        m['inhibitory_closure']=dict(spec=copy.deepcopy(spec),mode=mode,connected=connected,origin_ns=obj.time_ns,
            new_pairs=158,remaining_zero_direct_pairs=43,scale_measured=False,
            scope='Engineering1nS maximum pertransmitterfamily, equalperpair thennativecontacts. GABA fast/slow alternatives; GluCl including CA/LH is unverified. Sourceq dimensionless proxy.')
        m['scope']='450nonzero input routes when closureconnected,1095localoutputs.43zero-direct anatomical routes and native efficacy/receptor mechanisms unresolved. Provisional engineering closure, not identified PN.'
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for k in ('schema','pn_online_manifest','pn_online_state'):before.pop(k);after.pop(k)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Closure activation changed inherited CNS state')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes()
        if not isinstance(self._online_source,InhibitoryClosureSource):return
        s=self._online_source;allids=np.concatenate([s.allocation.source_ids,s.ach.source_ids,np.array([s.apl_spec['source_id']]),*[m.source_ids for m in s.inh_maps.values()]])
        if len(allids)!=450 or len(np.unique(allids))!=450 or not np.isin(allids,self._online_ports.source_ids).all():
            raise ValueError('Canonical nonzero input routes incomplete or duplicated')
        frame=self._online_ports.observe();zero=frame['incoming_weights']==0
        if int(zero.sum())!=43 or not np.array_equal(np.sort(allids),np.sort(frame['source_ids'][~zero])):
            raise ValueError('Closure must cover exactly the pre-existing nonzero source pairs')

    def validate_online(self):
        super().validate_online();s=self._online_source;m=self.pn_online_manifest['inhibitory_closure']
        if not isinstance(s,InhibitoryClosureSource):raise ValueError('Inhibitory closure source missing')
        s._assert_inh()
        if s.inh_mode!=m['mode'] or s.inh_connected!=m['connected'] or _record_hash(s.inh_spec)!=_record_hash(m['spec']):raise ValueError('Inhibitory closure policy changed')
        if any(r.time_ns!=self.time_ns for r in s.inh_receptors.values()):raise ValueError('Inhibitory receptor clocks diverged')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete inhibitory closure state')
        parent=dict(saved,schema=GpuPnAplGradedBrain.SCHEMA,pn_online_state=saved['pn_online_state']['base'])
        base=GpuPnAplGradedBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        m=obj.pn_online_manifest['inhibitory_closure']
        obj._online_source=InhibitoryClosureSource.adopt(base._online_source,m['spec'],mode=m['mode'],connected=m['connected'])
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnAplGradedBrain.backend_identity();out['PN_inhibitory_closure']='Explicit GABAfast/slow and separate GluCl proxy, declared family conductance budget and delayed histories; no native efficacy inferred.';return out

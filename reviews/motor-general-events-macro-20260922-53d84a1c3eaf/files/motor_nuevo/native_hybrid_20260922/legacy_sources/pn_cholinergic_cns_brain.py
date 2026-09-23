"""Partial canonical PN input expansion using an explicit ACh receptor prior."""
import copy
import numpy as np
from pn_online_cns_brain import GpuPnOnlineBrain,KEYS
from pn_cholinergic_online_source import CholinergicOnlineSource
from kcgamma_regional_brain import _record_hash


class GpuPnCholinergicBrain(GpuPnOnlineBrain):
    SCHEMA='matrix_pn_cholinergic_partial_brain_v1'

    @classmethod
    def adopt(cls,parent,spec,*,connected=True,coupling_ns=62500):
        if type(parent) is not GpuPnOnlineBrain or coupling_ns not in (125000,62500) or type(connected) is not bool:
            raise ValueError('Exact livePN parent, explicit current switch and supported coupling required')
        if len(spec['mapping']['source_ids'])!=217:raise ValueError('Complete217canonical ACh-pair selection required')
        if np.intersect1d(spec['mapping']['source_ids'],parent.visual_ids).size:raise ValueError('A graded visual coordinate cannot be interpreted as an event-rate proxy')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._online_source=CholinergicOnlineSource.adopt(parent._online_source,**spec,connected=connected)
        m=copy.deepcopy(parent.pn_online_manifest);m['coupling_ns']=coupling_ns
        m['cholinergic']=dict(spec=copy.deepcopy(spec),connected=connected,adoption_time_ns=obj.time_ns,parent_state_sha256=_record_hash(before),
            new_pairs=217,remaining_nonORN_pairs=202,history='Additional receptor filters zero at explicit route activation, not prior equilibrium.',
            amplitude='One published modeled .1nS pulse per pair, not percontact or W-scaled; proxy rate convolution conserves pulse area.')
        m['scope']='Live74ORNplus217ACh-pair-prior inputs to finePN/Ca->100KCgamma;202otherinputs not supplied and995outputs/PNq remain legacy.'
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m;obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for field in ('schema','pn_online_manifest','pn_online_state'):before.pop(field);after.pop(field)
        if _record_hash(before)!=_record_hash(after):raise ValueError('Receptor adoption changed inherited CNS state')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes();m=self.pn_online_manifest.get('cholinergic')
        if m is None or not isinstance(self._online_source,CholinergicOnlineSource):raise ValueError('Missing cholinergic source policy')
        ids=self._online_source.ach.source_ids;positions=np.searchsorted(self.brain.node_ids,ids)
        np.testing.assert_array_equal(self.brain.node_ids[positions],ids)
        np.testing.assert_array_equal(self.caps[positions],self._online_source.ach_caps)
        if not np.isin(ids,self._online_ports.source_ids).all():raise ValueError('ACh source is absent from canonical PN input anatomy')

    def validate_online(self):
        super().validate_online();m=self.pn_online_manifest['cholinergic'];s=self._online_source
        if (s.ach.time_ns!=self.time_ns or s.ach.origin_ns!=m['adoption_time_ns'] or s.ach_connected!=m['connected']
            or len(s.ach.source_ids)!=217 or not np.isfinite([s.ach_charge_pC,s.orn_charge_pC]).all()):raise ValueError('ACh receptor clocks/history/policy mismatch')
        s.ach._validated(s.ach.state_dict())

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete PN/ACh state')
        parent=dict(saved);parent['schema']=GpuPnOnlineBrain.SCHEMA;parent['pn_online_state']=saved['pn_online_state']['base']
        base=GpuPnOnlineBrain.from_state(brain,parent);obj=cls.__new__(cls);obj.__dict__.update(base.__dict__)
        m=obj.pn_online_manifest['cholinergic']
        obj._online_source=CholinergicOnlineSource.adopt(base._online_source,**m['spec'],connected=m['connected'],origin_ns=m['adoption_time_ns'])
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnOnlineBrain.backend_identity();out['pn_cholinergic']='217explicit ACh pair pulse priors, analytic rate convolution, unique stage ownership and separate receptor charge;202otherinputs and995outputs remain pending.';return out

"""Local APL feedback for either the fine or the admitted working PN branch."""
import copy
import numpy as np
from pn_general_output_brain import GpuPnGeneralOutputBrain
from pn_mass_cns_brain import GpuPnMassBrain
from pn_cholinergic_cns_brain import KEYS
from pn_cns_ports import PnCnsPorts
from pn_apl_graded_source import AplGradedSource
from kcgamma_regional_brain import _record_hash


class AplPnCnsPorts(PnCnsPorts):
    def observe(self):
        out=super().observe();h=self.hybrid
        out.update(APL_source_ids=h.routes.arrays['apl_ids'].copy(),APL_region_names=h.routes.arrays['region_names'].copy(),
            APL_graded_q=h.kc_apl_dynamic_state['apl_q'].copy())
        return out


class GpuPnAplGradedBrain(GpuPnGeneralOutputBrain):
    SCHEMA='matrix_pn_apl_graded_partial_brain_v1'

    @classmethod
    def adopt(cls,parent,spec,*,connected,coupling_ns=15625):
        if type(parent) not in (GpuPnGeneralOutputBrain,GpuPnMassBrain) or coupling_ns not in (31250,15625):
            raise ValueError('Exact qualified parent and declared coupling required')
        if parent.time_ns!=spec['origin_ns']:raise ValueError('APL receptor activation must match current CNS clock')
        before=parent.state_dict();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
        obj._online_source=AplGradedSource.adopt(parent._online_source,spec,connected=connected)
        if _record_hash(obj._online_source.state_dict()['base'])!=_record_hash(before['pn_online_state']):
            raise ValueError('APL adoption changed previous source histories')
        m=copy.deepcopy(parent.pn_online_manifest);m['coupling_ns']=coupling_ns;m['electrical_outputs']['temporal_refinement']['coupling_ns']=coupling_ns
        m['apl_feedback']=dict(spec=copy.deepcopy(spec),connected=connected,origin_ns=obj.time_ns,
            new_pairs=1,missing_inputs=201 if connected else 202,
            scope='Local graded APL→PN GABA_A-model conductance, no spiking-rate conversion.1nS pair full-scale is engineering prior, not measured efficacy; kinetics transferred.')
        m['scope']='291previous inputs plus1graded APL route when connected;1095local PN outputs.201other inputs unresolved when connected. Biological scales and mechanisms provisional.'
        m['record_sha256']=_record_hash(m);obj.pn_online_manifest=m
        obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online()
        after=obj.state_dict()
        for k in ('schema','pn_online_manifest','pn_online_state'):before.pop(k);after.pop(k)
        if _record_hash(before)!=_record_hash(after):raise ValueError('APL activation changed inherited CNS fields')
        return obj

    def _bind_online_routes(self):
        super()._bind_online_routes()
        if not isinstance(self._online_source,AplGradedSource):return
        self._online_ports=AplPnCnsPorts(self,10208);s=self._online_source
        ids=self._online_ports.source_ids
        if np.sum(ids==10977)!=1 or np.isin(10977,s.ach.source_ids) or np.isin(10977,s.allocation.source_ids):
            raise ValueError('APL anatomy absent or duplicated in another receptor group')
        np.testing.assert_array_equal(self.routes.arrays['region_names'],s.apl_spec['region_names'])

    def validate_online(self):
        super().validate_online();s=self._online_source;m=self.pn_online_manifest['apl_feedback']
        if not isinstance(s,AplGradedSource):raise ValueError('APL feedback source missing')
        s._assert_apl()
        if (s.apl_receptor.time_ns!=self.time_ns or s.apl_receptor.origin_ns!=m['origin_ns'] or s.apl_connected!=m['connected']
            or _record_hash(s.apl_spec)!=_record_hash(m['spec'])):raise ValueError('APL receptor history/policy mismatch')
        s.apl_receptor.validated(s.apl_receptor.state_dict())
        if 'mass' in self.pn_online_manifest:
            mass=self.pn_online_manifest['mass']
            if s.pn.identity!=mass['pn_identity'] or len(s.pn.voltage)!=mass['exterior_coordinates']+mass['internal_coordinates']:
                raise ValueError('Mass representation changed within APL branch')

    @classmethod
    def from_state(cls,brain,saved):
        if set(saved)!=KEYS or saved['schema']!=cls.SCHEMA:raise ValueError('Incomplete APL feedback CNS state')
        parent=dict(saved);parent['pn_online_state']=saved['pn_online_state']['base']
        base_class=GpuPnMassBrain if 'mass' in saved['pn_online_manifest'] else GpuPnGeneralOutputBrain
        parent['schema']=base_class.SCHEMA;base=base_class.from_state(brain,parent)
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);m=obj.pn_online_manifest['apl_feedback']
        obj._online_source=AplGradedSource.adopt(base._online_source,m['spec'],connected=m['connected'])
        obj._online_source.load_state_dict(saved['pn_online_state']);obj._bind_online_routes();obj._rebind_continuing_inputs();obj.validate_online();return obj

    @staticmethod
    def backend_identity():
        out=GpuPnGeneralOutputBrain.backend_identity()
        out['APL_PN_graded']='Explicit delayed receptor driven by regional APL_q before generic transmission; fine or guarded mass PN selected by saved coordinate contract. Pair efficacy provisional.'
        return out

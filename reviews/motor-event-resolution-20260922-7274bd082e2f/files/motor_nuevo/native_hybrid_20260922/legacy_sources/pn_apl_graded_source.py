"""One graded APL→PN receptor route around the existing291-input source.

The pair efficacy is an explicit engineering scale, not an EM count, Hz rate,
miniature amplitude, or identified native conductance. All local PN outputs
share the same chemistry, updated once per electrical interval.
"""
import hashlib,copy
import numpy as np
from pn_general_output_source import GeneralOutputSource,SCHEMA as BASE_SCHEMA
from pn_apl_graded_receptor import DelayedGradedReceptor
from pn_cns_orn_stages import canonical_orn_stages
from pn_local_conductance import combine_conductance_stages
from pn_cholinergic_rate_prior import REVERSAL_MV
from pn_prepared_kc_boundary import zero_release_tail
from pn_coupled_ionic import GAMMA
from kcgamma_regional_brain import _record_hash

SCHEMA='PN_APL_graded_source_v1'


class AplGradedSource(GeneralOutputSource):
    @classmethod
    def adopt(cls,base,spec,*,connected):
        if type(base) is not GeneralOutputSource or type(connected) is not bool:raise ValueError('Exact general source and explicit APL connection required')
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);spec=copy.deepcopy(spec)
        nodes=np.asarray(spec['contact_nodes']);fractions=np.asarray(spec['contact_fractions'],dtype=float);regions=np.asarray(spec['contact_region_slots'])
        if (nodes.shape!=(242,) or nodes.dtype.kind not in 'iu' or np.any(nodes<0) or np.any(nodes>=len(base.pn.voltage))
            or fractions.shape!=nodes.shape or not np.isfinite(fractions).all() or np.any(fractions<=0) or abs(fractions.sum()-1)>1e-12
            or regions.shape!=nodes.shape or regions.dtype.kind not in 'iu' or np.any(regions<0) or np.any(regions>=len(spec['region_names']))
            or not np.isfinite(spec['pair_full_scale_nS']) or spec['pair_full_scale_nS']<=0 or not np.isfinite(spec['reversal_mV'])
            or spec['source_id']!=10977 or spec['target_id']!=10208 or np.any(base.pn.C[nodes]<=0)):
            raise ValueError('Native242contact membrane map and explicit whole-pair scale required')
        obj.apl_spec=spec;obj.apl_connected=connected
        obj.apl_receptor=DelayedGradedReceptor(spec['region_names'],initial_driver=spec['initial_driver'],time_ns=spec['origin_ns'],
            tau_s=spec['tau_s'],delay_ns=spec['delay_ns'],provenance=spec['provenance'])
        obj.apl_nodes,obj.apl_contact_slots=np.unique(nodes,return_inverse=True);obj.apl_charge_pC=0.;obj._general_identity=base.identity
        obj.apl_nodes=np.frombuffer(obj.apl_nodes.tobytes(),dtype=obj.apl_nodes.dtype)
        obj.apl_contact_slots=np.frombuffer(obj.apl_contact_slots.tobytes(),dtype=obj.apl_contact_slots.dtype)
        obj._apl_definition=(_record_hash(spec),connected,obj.apl_receptor,obj.apl_nodes,obj.apl_contact_slots)
        obj.identity=hashlib.sha256((SCHEMA+base.identity+_record_hash(spec)+str(connected)).encode()).hexdigest()
        return obj

    def _assert_apl(self):
        d=self._apl_definition
        if (_record_hash(self.apl_spec)!=d[0] or self.apl_connected!=d[1] or self.apl_receptor is not d[2]
            or self.apl_nodes is not d[3] or self.apl_contact_slots is not d[4]):raise ValueError('APL receptor definition changed')

    def _apl_driver(self,frame):
        ids=np.asarray(frame['APL_source_ids']);q=np.asarray(frame['APL_graded_q'])
        names=tuple(frame['APL_region_names'])
        if q.shape!=(len(ids),len(names)) or names!=tuple(self.apl_spec['region_names']):raise ValueError('Wrong named APL regional frame')
        slots=np.flatnonzero(ids==self.apl_spec['source_id'])
        if len(slots)!=1:raise ValueError('APL driver identity missing')
        return self.apl_receptor._driver(q[slots[0]])

    def apl_stage(self,occupancy):
        p=self.apl_spec;q=np.asarray(occupancy)
        contact=p['pair_full_scale_nS']*np.asarray(p['contact_fractions'])*q[np.asarray(p['contact_region_slots'])]
        if not self.apl_connected:contact=np.zeros_like(contact)
        return dict(nodes=self.apl_nodes,conductance_nS=np.bincount(self.apl_contact_slots,weights=contact,minlength=len(self.apl_nodes)),reversal_mV=p['reversal_mV'])

    def state_dict(self):
        self._assert_apl()
        base=GeneralOutputSource.state_dict(self);base['identity']=self._general_identity
        return dict(schema=SCHEMA,identity=self.identity,base=base,apl_receptor=self.apl_receptor.state_dict(),apl_charge_pC=self.apl_charge_pC)

    def load_state_dict(self,saved):
        if (set(saved)!={'schema','identity','base','apl_receptor','apl_charge_pC'} or saved['schema']!=SCHEMA or saved['identity']!=self.identity
            or saved['base']['schema']!=BASE_SCHEMA or saved['base']['identity']!=self._general_identity
            or saved['apl_receptor']['time_ns']!=saved['base']['general_output']['time_ns'] or not np.isscalar(saved['apl_charge_pC']) or not np.isfinite(saved['apl_charge_pC'])):
            raise ValueError('Wrong joint APL source identity, charge or clock')
        validated=self.apl_receptor.validated(saved['apl_receptor'])
        GeneralOutputSource.load_state_dict(self,dict(saved['base'],identity=self.identity))
        self.apl_receptor.occupancy,self.apl_receptor.last_driver,self.apl_receptor.history=validated
        self.apl_receptor.time_ns=saved['apl_receptor']['time_ns'];self.apl_charge_pC=float(saved['apl_charge_pC'])

    def advance(self,dt_ns,before,after,*,connected=True,**solver_options):
        self._assert_apl()
        if (before.get('time_ns')!=self.time_ns or self.ach.time_ns!=self.time_ns or self.apl_receptor.time_ns!=self.time_ns
            or self.pn.time_ns-self.source_origin_ns!=self.time_ns-self.cns_origin_ns):raise ValueError('CNS/APL/source interval disagreement')
        if 'stage_observation_nodes' in solver_options:raise ValueError('Receptor source owns charge observation nodes')
        orn=canonical_orn_stages(self.allocation,self.caps,before,after,dt_ns,connected=connected)
        g,ach_next=self.ach.preview(dt_ns,self._frame_rates(before),self._frame_rates(after))
        ach=[self.ach_map.sample(x if self.ach_connected else np.zeros_like(x),REVERSAL_MV) for x in g]
        occupancy,apl_next=self.apl_receptor.preview(dt_ns,self._apl_driver(before),self._apl_driver(after))
        apl=[self.apl_stage(x) for x in occupancy];stages=[combine_conductance_stages(v) for v in zip(orn,ach,apl)];nodes=stages[0]['nodes']
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise,self.decay,dt_ns);previous=self.state_dict()
        try:
            report=self.pn.advance(dt_ns,self.pn.cp.zeros_like(self.pn.voltage),synaptic_stages=stages,stage_observation_nodes=nodes,**solver_options)
            if not report['accepted']:return report
            voltages=np.asarray(report['stage_observations']['voltage_mV']);charges=[]
            for group in (orn,ach,apl):
                current=[float(np.sum(x['conductance_nS']*(v[np.searchsorted(nodes,x['nodes'])]-x['reversal_mV']))) for x,v in zip(group,voltages)]
                charges.append(dt_ns*1e-9*((1-GAMMA)*current[0]+GAMMA*current[1]))
            error=abs(sum(charges)-report['synaptic_outward_charge_increment_pC'])
            if not np.isfinite(charges).all() or error>1e-11+1e-12*sum(abs(x) for x in charges):raise ValueError('Local ORN/ACh/APL charge decomposition failed')
            self.fast=fast;self.slow=slow;self.time_ns+=dt_ns;self.ach.load_state_dict(ach_next);self.apl_receptor.load_state_dict(apl_next)
            self.orn_charge_pC+=charges[0];self.ach_charge_pC+=charges[1];self.apl_charge_pC+=charges[2]
            self.extra_output.advance(dt_ns);self.general_output.advance(dt_ns)
            self.output_nS();self.additional_output_nS();self.general_transmission()
            report['local_receptor_charge']=dict(orn_pC=charges[0],ach_pC=charges[1],apl_pC=charges[2],sum_error_pC=error)
            return report
        except BaseException:self.load_state_dict(previous);raise

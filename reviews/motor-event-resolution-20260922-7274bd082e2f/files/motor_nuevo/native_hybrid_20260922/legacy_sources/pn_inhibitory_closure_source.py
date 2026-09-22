"""Explicit provisional GABA/Glu closure at all remaining nonzero PN inputs.

Driver coordinates are inherited dimensionless source activity, not measured
release or spikes. Each family has a mandatory engineering conductance budget.
GABA fast/slow are alternatives; glutamate is a separate receptor hypothesis.
"""
import copy,hashlib
import numpy as np
from pn_apl_graded_source import AplGradedSource,SCHEMA as BASE_SCHEMA
from pn_apl_graded_receptor import DelayedGradedReceptor
from pn_delayed_cascade_receptor import DelayedCascadeReceptor
from pn_cns_orn_stages import canonical_orn_stages
from pn_local_conductance import LocalPairConductance,combine_conductance_stages
from pn_cholinergic_rate_prior import REVERSAL_MV
from pn_prepared_kc_boundary import zero_release_tail
from pn_coupled_ionic import GAMMA
from kcgamma_regional_brain import _record_hash

SCHEMA='PN_inhibitory_closure_source_v1'


class InhibitoryClosureSource(AplGradedSource):
    @classmethod
    def adopt(cls,base,spec,*,mode,connected):
        if type(base) is not AplGradedSource or type(connected) is not bool or mode not in ('fast','slow'):
            raise ValueError('Exact APL source, explicit connection and GABA mechanism required')
        obj=cls.__new__(cls);obj.__dict__.update(base.__dict__);obj.inh_spec=copy.deepcopy(spec)
        obj.inh_mode=mode;obj.inh_connected=connected;obj.inh_maps={};obj.inh_receptors={};obj.inh_charge_pC={}
        existing=np.r_[base.allocation.source_ids,base.ach.source_ids,base.apl_spec['source_id']]
        if set(spec['families'])!={'GABA','Glu'}:raise ValueError('Separate complete GABA/Glu families required')
        for name,expected in [('GABA',86),('Glu',72)]:
            p=spec['families'][name];mapping=LocalPairConductance(**p['mapping']);ids=mapping.source_ids
            if (len(ids)!=expected or np.intersect1d(existing,ids).size or mapping.full_size!=len(base.pn.voltage)
                or np.any(base.pn.C[mapping.nodes]<=0) or not np.isfinite(p['family_full_scale_nS']) or p['family_full_scale_nS']<=0):
                raise ValueError('Complete disjoint native membrane routes and declared family budget required')
            existing=np.r_[existing,ids];obj.inh_maps[name]=mapping
            kwargs=dict(initial_driver=p['initial_driver'],time_ns=spec['origin_ns'],delay_ns=p['delay_ns'],provenance=p['provenance'])
            names=[str(i) for i in ids]
            if name=='GABA' and mode=='slow':
                receptor=DelayedCascadeReceptor(names,**kwargs,tau_input_s=p['slow_tau_input_s'],tau_effector_s=p['slow_tau_effector_s'])
            else:receptor=DelayedGradedReceptor(names,**kwargs,tau_s=p['fast_tau_s'])
            obj.inh_receptors[name]=receptor;obj.inh_charge_pC[name]=0.
        obj._apl_identity=base.identity
        obj._inh_definition=(_record_hash(spec),mode,connected,dict(obj.inh_maps),dict(obj.inh_receptors))
        obj.identity=hashlib.sha256((SCHEMA+base.identity+_record_hash(spec)+mode+str(connected)).encode()).hexdigest()
        return obj

    def _assert_inh(self):
        d=self._inh_definition
        if (_record_hash(self.inh_spec)!=d[0] or self.inh_mode!=d[1] or self.inh_connected!=d[2]
            or any(self.inh_maps[n] is not d[3][n] or self.inh_receptors[n] is not d[4][n] for n in ('GABA','Glu'))):
            raise ValueError('Inhibitory closure definition changed')

    def _inh_driver(self,name,frame):
        ids=np.asarray(frame['source_ids']);q=np.asarray(frame['source_q']);target=self.inh_maps[name].source_ids
        if ids.ndim!=1 or q.shape!=ids.shape or len(np.unique(ids))!=len(ids):raise ValueError('Named unique source coordinates required')
        order=np.argsort(ids);pos=np.searchsorted(ids[order],target)
        if np.any(pos>=len(ids)) or not np.array_equal(ids[order][pos],target):raise ValueError('Inhibitory source missing')
        x=q[order][pos]
        if not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Bounded dimensionless source driver required')
        return x.copy()

    def inh_stage(self,name,occupancy):
        p=self.inh_spec['families'][name];m=self.inh_maps[name]
        g=p['family_full_scale_nS']/len(m.source_ids)*np.asarray(occupancy)
        if not self.inh_connected:g=np.zeros_like(g)
        E=p['slow_reversal_mV'] if name=='GABA' and self.inh_mode=='slow' else p['fast_reversal_mV']
        return m.sample(g,E)

    def state_dict(self):
        self._assert_inh();base=AplGradedSource.state_dict(self);base['identity']=self._apl_identity
        return dict(schema=SCHEMA,identity=self.identity,base=base,receptors={k:v.state_dict() for k,v in self.inh_receptors.items()},charges_pC=dict(self.inh_charge_pC))

    def load_state_dict(self,state):
        self._assert_inh()
        if (set(state)!={'schema','identity','base','receptors','charges_pC'} or state['schema']!=SCHEMA or state['identity']!=self.identity
            or state['base']['schema']!=BASE_SCHEMA or state['base']['identity']!=self._apl_identity
            or set(state['receptors'])!={'GABA','Glu'} or set(state['charges_pC'])!={'GABA','Glu'}):raise ValueError('Wrong inhibitory source identity')
        # Validate a small copy before touching PN/CNS state; failed input loads are atomic.
        receptors=copy.deepcopy(self.inh_receptors)
        for name,r in receptors.items():
            r.load_state_dict(state['receptors'][name]);charge=state['charges_pC'][name]
            if r.time_ns!=state['base']['apl_receptor']['time_ns'] or not np.isscalar(charge) or not np.isfinite(charge):
                raise ValueError('Receptor clocks and finite signed charges required')
        AplGradedSource.load_state_dict(self,dict(state['base'],identity=self.identity))
        for name,r in self.inh_receptors.items():r.load_state_dict(state['receptors'][name])
        self.inh_charge_pC={n:float(v) for n,v in state['charges_pC'].items()}

    def advance(self,dt_ns,before,after,*,connected=True,**solver_options):
        self._assert_apl();self._assert_inh()
        if (before.get('time_ns')!=self.time_ns or self.ach.time_ns!=self.time_ns or self.apl_receptor.time_ns!=self.time_ns
            or any(r.time_ns!=self.time_ns for r in self.inh_receptors.values())
            or self.pn.time_ns-self.source_origin_ns!=self.time_ns-self.cns_origin_ns):raise ValueError('CNS/receptor/source interval disagreement')
        if 'stage_observation_nodes' in solver_options:raise ValueError('Receptor source owns charge observation nodes')
        orn=canonical_orn_stages(self.allocation,self.caps,before,after,dt_ns,connected=connected)
        g,ach_next=self.ach.preview(dt_ns,self._frame_rates(before),self._frame_rates(after))
        ach=[self.ach_map.sample(x if self.ach_connected else np.zeros_like(x),REVERSAL_MV) for x in g]
        occupancy,apl_next=self.apl_receptor.preview(dt_ns,self._apl_driver(before),self._apl_driver(after))
        groups=[orn,ach,[self.apl_stage(x) for x in occupancy]];next_states={}
        for name,r in self.inh_receptors.items():
            y,next_states[name]=r.preview(dt_ns,self._inh_driver(name,before),self._inh_driver(name,after))
            groups.append([self.inh_stage(name,x) for x in y])
        stages=[combine_conductance_stages(v) for v in zip(*groups)];nodes=stages[0]['nodes']
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise,self.decay,dt_ns);previous=self.state_dict()
        try:
            report=self.pn.advance(dt_ns,self.pn.cp.zeros_like(self.pn.voltage),synaptic_stages=stages,stage_observation_nodes=nodes,**solver_options)
            if not report['accepted']:return report
            voltages=np.asarray(report['stage_observations']['voltage_mV']);charges=[]
            for group in groups:
                current=[float(np.sum(x['conductance_nS']*(v[np.searchsorted(nodes,x['nodes'])]-x['reversal_mV']))) for x,v in zip(group,voltages)]
                charges.append(dt_ns*1e-9*((1-GAMMA)*current[0]+GAMMA*current[1]))
            error=abs(sum(charges)-report['synaptic_outward_charge_increment_pC'])
            if not np.isfinite(charges).all() or error>1e-11+1e-12*sum(abs(x) for x in charges):raise ValueError('Local receptor charge decomposition failed')
            self.fast=fast;self.slow=slow;self.time_ns+=dt_ns;self.ach.load_state_dict(ach_next);self.apl_receptor.load_state_dict(apl_next)
            self.orn_charge_pC+=charges[0];self.ach_charge_pC+=charges[1];self.apl_charge_pC+=charges[2]
            for i,(name,r) in enumerate(self.inh_receptors.items()):r.load_state_dict(next_states[name]);self.inh_charge_pC[name]+=charges[i+3]
            self.extra_output.advance(dt_ns);self.general_output.advance(dt_ns)
            self.output_nS();self.additional_output_nS();self.general_transmission()
            report['local_receptor_charge']=dict(orn_pC=charges[0],ach_pC=charges[1],apl_pC=charges[2],GABA_pC=charges[3],Glu_pC=charges[4],sum_error_pC=error)
            return report
        except BaseException:self.load_state_dict(previous);raise

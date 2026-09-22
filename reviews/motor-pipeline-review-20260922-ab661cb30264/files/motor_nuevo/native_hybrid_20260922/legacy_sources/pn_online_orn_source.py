"""Prepared fine-PN/Ca source driven by single-owner canonical ORN filters.

Only74ORN inputs are supplied here. Other419input mechanisms are not silently
assigned units. The calling CNS may retain those routes in its legacy PN; this
component is not a complete PN replacement. Source and CNS origins are explicit.
"""
import hashlib
import numpy as np
from pn_cns_orn_stages import canonical_orn_stages
from pn_prepared_kc_boundary import zero_release_tail
from pn_calcium_release import SiteToTargetConductance

SCHEMA='PN_online_canonical_ORN_source_v1'

class OnlineOrnPnSource:
    def __init__(self,pn,allocation,caps_hz,mapping,*,cns_time_ns,preparation,gain_provenance):
        if (type(cns_time_ns) is not int or cns_time_ns<0 or not isinstance(preparation,str) or not preparation
            or not isinstance(gain_provenance,str) or not gain_provenance):raise ValueError('Explicit CNS origin and preparations required')
        self.pn=pn;self.allocation=allocation;self.caps=np.asarray(caps_hz,dtype=float).copy()
        self.route=SiteToTargetConductance(**mapping);chem=pn.calcium_port.chemistry
        np.testing.assert_array_equal(chem.ids,self.route.site_ids)
        allocation.sample_filters(np.zeros((4,len(self.caps))),self.caps)
        self.rise=chem.parameters['rise_s'].copy();self.decay=chem.parameters['decay_s'].copy();self.norm=chem.norm.copy()
        self.fast=chem.fast.copy();self.slow=chem.slow.copy()
        self.source_origin_ns=pn.time_ns;self.cns_origin_ns=cns_time_ns;self.time_ns=cns_time_ns
        h=hashlib.sha256((SCHEMA+pn.identity+preparation+gain_provenance+repr((pn.time_ns,cns_time_ns))).encode())
        for v in [allocation.source_ids,allocation.contact_pre_ids,allocation.contact_nodes,allocation.pair_shares,self.caps,
                  self.route.site_ids,self.route.slot,self.route.target_ids,self.route.target_slot,self.route.gain,self.fast,self.slow]:h.update(v.tobytes())
        self.identity=h.hexdigest();self.preparation=preparation
        if np.any(self.output_nS()!=0):raise ValueError('New output at handoff must be exactly zero')

    def output_nS(self):
        generated=self.pn.calcium_port.chemistry.activation()-(self.slow-self.fast)/self.norm
        if np.any(generated<0):raise ValueError('Generated release below inherited source tail; no clipping')
        return self.route.evaluate(generated)

    def advance(self,dt_ns,before,after,*,connected=True,**solver_options):
        if (before.get('time_ns')!=self.time_ns or self.pn.time_ns-self.source_origin_ns!=self.time_ns-self.cns_origin_ns):
            raise ValueError('Source and CNS clocks do not share the declared interval')
        stages=canonical_orn_stages(self.allocation,self.caps,before,after,dt_ns,connected=connected)
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise,self.decay,dt_ns)
        previous=self.state_dict()
        try:
            report=self.pn.advance(dt_ns,self.pn.cp.zeros_like(self.pn.voltage),synaptic_stages=stages,**solver_options)
            if report['accepted']:
                self.fast=fast;self.slow=slow;self.time_ns+=dt_ns;self.output_nS()
            return report
        except BaseException:
            self.load_state_dict(previous);raise

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,pn=self.pn.state_dict(),fast=self.fast.copy(),slow=self.slow.copy())

    def load_state_dict(self,state):
        if (set(state)!={'schema','identity','time_ns','pn','fast','slow'} or state['schema']!=SCHEMA or state['identity']!=self.identity
            or type(state['time_ns']) is not int or state['time_ns']<self.cns_origin_ns
            or state['pn']['pn']['time_ns']-self.source_origin_ns!=state['time_ns']-self.cns_origin_ns):
            raise ValueError('Wrong online source identity or clock')
        a,b=np.asarray(state['fast']),np.asarray(state['slow'])
        if (a.shape!=self.fast.shape or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a<0) or np.any(b<a)):
            raise ValueError('Invalid inherited source tail')
        chem=state['pn']['calcium']['chemistry'];generated=(chem['slow']-chem['fast'])/self.norm-(b-a)/self.norm
        if np.any(generated<0):raise ValueError('Persisted output below inherited tail')
        a=a.astype(float,copy=True);b=b.astype(float,copy=True)
        self.pn.load_state_dict(state['pn']);self.fast=a;self.slow=b;self.time_ns=state['time_ns']

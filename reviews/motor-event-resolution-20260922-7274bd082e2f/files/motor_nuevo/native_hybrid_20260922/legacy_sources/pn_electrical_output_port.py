"""Additional KC/APL local release with the inherited class scale explicit.

The old generic PN transmission remains shared by other consumers. This port
keeps only its homogeneous tail for selected pairs and maps NEW local Ca-site
release to their conductances. One expected site release is transferred to one
old rate-proxy event; this is a hypothesis, not a measured vesicle efficacy.
"""
import hashlib
import numpy as np
from pn_calcium_release import SiteToTargetConductance
from pn_prepared_kc_boundary import zero_release_tail

SCHEMA='PN_additional_electrical_output_port_v1'

class AdditionalElectricalOutput:
    def __init__(self,mapping,*,site_fast,site_slow,rise_s,decay_s,legacy_transmission,
                 legacy_tau_s,legacy_gain_nS,time_ns,provenance):
        self.route=SiteToTargetConductance(**mapping);n=len(self.route.site_ids)
        arrays=[np.asarray(x,dtype=float) for x in (site_fast,site_slow,rise_s,decay_s)]
        if (any(a.shape!=(n,) or not np.isfinite(a).all() for a in arrays) or np.any(arrays[0]<0) or np.any(arrays[1]<arrays[0])
            or np.any(arrays[2]<=0) or np.any(arrays[3]<=arrays[2]) or type(time_ns) is not int or time_ns<0
            or not isinstance(provenance,str) or not provenance.strip() or not np.isfinite(legacy_tau_s) or legacy_tau_s<=0
            or not np.isscalar(legacy_transmission) or not 0<=legacy_transmission<=1):raise ValueError('Explicit release preparation, legacy tail and units required')
        gain=np.asarray(legacy_gain_nS,dtype=float)
        if gain.shape!=self.route.target_ids.shape or not np.isfinite(gain).all() or np.any(gain<=0):raise ValueError('Positive inherited conductance gain for every selected consumer required')
        self.fast,self.slow=[a.copy() for a in arrays[:2]]
        self.rise,self.decay=[np.frombuffer(a.tobytes(),dtype=float) for a in arrays[2:]]
        peak=np.log(self.decay/self.rise)/(1/self.rise-1/self.decay);self.norm=np.exp(-peak/self.decay)-np.exp(-peak/self.rise)
        self.legacy_gain=np.frombuffer(gain.tobytes(),dtype=float);self.legacy=float(legacy_transmission);self.tau=float(legacy_tau_s)
        self.time_ns=time_ns;self.origin_ns=time_ns;self.provenance=provenance
        h=hashlib.sha256((SCHEMA+provenance+repr((time_ns,legacy_transmission,legacy_tau_s))).encode())
        for a in (*arrays,gain,self.route.site_ids,self.route.slot,self.route.target_ids,self.route.target_slot,self.route.gain):h.update(a.tobytes())
        self.identity=h.hexdigest()

    def output_nS(self,site_activation):
        a=np.asarray(site_activation,dtype=float)
        generated=a-(self.slow-self.fast)/self.norm
        if a.shape!=self.fast.shape or not np.isfinite(a).all() or np.any(generated<0):raise ValueError('Local release is below its preserved prehistory')
        return self.legacy*self.legacy_gain+self.route.evaluate(generated)

    def advance(self,dt_ns):
        if type(dt_ns) is not int or not 0<dt_ns<=125000:raise ValueError('Bounded positive electrical coupling interval required')
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise,self.decay,dt_ns)
        self.fast=fast;self.slow=slow;self.legacy*=np.exp(-dt_ns*1e-9/self.tau);self.time_ns+=dt_ns

    def state_dict(self):return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,fast=self.fast.copy(),slow=self.slow.copy(),legacy=self.legacy)

    def validate_state(self,s):
        if (set(s)!={'schema','identity','time_ns','fast','slow','legacy'} or s['schema']!=SCHEMA or s['identity']!=self.identity
            or type(s['time_ns']) is not int or s['time_ns']<self.origin_ns or not np.isscalar(s['legacy']) or not 0<=s['legacy']<=1):raise ValueError('Wrong electrical output identity or clock')
        a,b=np.asarray(s['fast']),np.asarray(s['slow'])
        if (a.shape!=self.fast.shape or b.shape!=a.shape or a.dtype.kind not in 'fiu' or b.dtype.kind not in 'fiu'
            or not np.isfinite([a,b]).all() or np.any(a<0) or np.any(b<a)):raise ValueError('Invalid preserved output tail')
        return a.astype(float,copy=True),b.astype(float,copy=True)

    def load_state_dict(self,s):
        self.fast,self.slow=self.validate_state(s);self.legacy=float(s['legacy']);self.time_ns=s['time_ns']

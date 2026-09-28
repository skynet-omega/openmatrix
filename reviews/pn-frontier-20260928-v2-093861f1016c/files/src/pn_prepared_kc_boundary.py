"""Route release generated after handoff, retaining the old KC tail once.

Prepared Ca filters may have different kinetics from the inherited KC input.
Track their zero-future-release SDIRK solution separately and subtract it
from the observed source activation. Only newly generated release drives
the additional KC term. No existing Ca state or KC tail is reset or fitted.
This external boundary has explicit source/receiver origins; it is not a
live CNS migration or reconstruction of the original calcium history.
"""
import hashlib
import numpy as np
from pn_site_kc_boundary import PnSiteKcBoundary
from pn_calcium_port import GAMMA,METHOD

SCHEMA='PN_prepared_release_window_KC_v1'

def zero_release_tail(fast,slow,rise_s,decay_s,dt_ns):
    """Homogeneous part of the exact same two discrete chemical stages."""
    q=GAMMA*dt_ns*1e-9;ratio=(1-GAMMA)/GAMMA;out=[]
    for old,tau in [(fast,rise_s),(slow,decay_s)]:
        first=old/(1+q/tau);base=old+ratio*(first-old)
        out.append(base/(1+q/tau))
    if any(not np.isfinite(a).all() or np.any(a<0) for a in out) or np.any(out[1]<out[0]):
        raise ValueError('Step does not preserve a nonnegative homogeneous release tail')
    return out

class PreparedPnSiteKcBoundary:
    def __init__(self,*args,site_fast,site_slow,site_rise_s,site_decay_s,source_time_ns,
                 source_identity,source_method,preparation,**kwargs):
        self.boundary=PnSiteKcBoundary(*args,**kwargs);n=len(self.boundary.route.site_ids)
        values=[np.asarray(v) for v in (site_fast,site_slow,site_rise_s,site_decay_s)]
        if (any(v.shape!=(n,) or v.dtype.kind not in 'fiu' or not np.isfinite(v).all() for v in values)
                or np.any(values[0]<0) or np.any(values[1]<values[0])
                or np.any(values[2]<=0) or np.any(values[3]<=values[2])
                or type(source_time_ns) is not int or source_time_ns<0
                or not isinstance(source_identity,str) or not source_identity
                or source_method!=METHOD or not isinstance(preparation,str) or not preparation.strip()):
            raise ValueError('Explicit Ca state, identity, SDIRK method, source clock and preparation required')
        f,s,r,d=[np.asarray(v,dtype=float).copy() for v in values]
        self.rise_s=np.frombuffer(r.tobytes(),dtype=float);self.decay_s=np.frombuffer(d.tobytes(),dtype=float)
        peak=np.log(d/r)/(1/r-1/d);norm=np.exp(-peak/d)-np.exp(-peak/r)
        self.norm=np.frombuffer(norm.tobytes(),dtype=float)
        self.fast=f;self.slow=s;self.source_activation=(s-f)/self.norm
        self.source_origin_ns=source_time_ns;self.source_time_ns=source_time_ns
        self.receiver_origin_ns=self.boundary.time_ns
        h=hashlib.sha256((SCHEMA+self.boundary.identity+source_identity+source_method+preparation+
            repr((source_time_ns,self.receiver_origin_ns))).encode())
        for v in values:h.update(np.asarray(v,dtype=float).tobytes())
        self.identity=h.hexdigest()

    @property
    def time_ns(self):return self.boundary.time_ns

    @property
    def route(self):return self.boundary.route

    def advance(self,dt_ns,source_activation_endpoint,*,source_endpoint_time_ns):
        a=np.asarray(source_activation_endpoint)
        if (type(dt_ns) is not int or not 0<dt_ns<=125000 or type(source_endpoint_time_ns) is not int
                or source_endpoint_time_ns!=self.source_time_ns+dt_ns
                or self.source_time_ns-self.source_origin_ns!=self.time_ns-self.receiver_origin_ns
                or a.shape!=self.fast.shape or a.dtype.kind not in 'fiu' or not np.isfinite(a).all() or np.any(a<0)):
            raise ValueError('Matching source endpoint clock and finite site activation required')
        fast,slow=zero_release_tail(self.fast,self.slow,self.rise_s,self.decay_s,dt_ns)
        new0=self.source_activation-(self.slow-self.fast)/self.norm;new1=a-(slow-fast)/self.norm
        if np.any(new0<0) or np.any(new1<0):raise ValueError('Source activation below its inherited tail; no clipping')
        # Piecewise-linear interpolation is assessed by paired time refinement.
        midpoint=(new0+new1)/2;next_activation=a.astype(float,copy=True)
        self.boundary.advance(dt_ns,midpoint)
        self.fast=fast;self.slow=slow;self.source_activation=next_activation;self.source_time_ns=source_endpoint_time_ns

    def state_dict(self):
        return dict(schema=SCHEMA,identity=self.identity,source_time_ns=self.source_time_ns,
            fast=self.fast.copy(),slow=self.slow.copy(),source_activation=self.source_activation.copy(),
            boundary=self.boundary.state_dict())

    def load_state_dict(self,state):
        if (set(state)!={'schema','identity','source_time_ns','fast','slow','source_activation','boundary'}
                or state['schema']!=SCHEMA or state['identity']!=self.identity
                or type(state['source_time_ns']) is not int or state['source_time_ns']<self.source_origin_ns
                or state['source_time_ns']-self.source_origin_ns!=state['boundary']['time_ns']-self.receiver_origin_ns):
            raise ValueError('Prepared release identity or clock mismatch')
        arrays=[np.asarray(state[k]) for k in ['fast','slow','source_activation']]
        if (any(v.shape!=self.fast.shape or v.dtype.kind not in 'fiu' or not np.isfinite(v).all() or np.any(v<0) for v in arrays)
                or np.any(arrays[1]<arrays[0]) or np.any(arrays[2]<(arrays[1]-arrays[0])/self.norm)):
            raise ValueError('Invalid prepared release tail state')
        arrays=[v.astype(float,copy=True) for v in arrays]
        self.boundary.load_state_dict(state['boundary'])
        self.fast,self.slow,self.source_activation=arrays;self.source_time_ns=state['source_time_ns']

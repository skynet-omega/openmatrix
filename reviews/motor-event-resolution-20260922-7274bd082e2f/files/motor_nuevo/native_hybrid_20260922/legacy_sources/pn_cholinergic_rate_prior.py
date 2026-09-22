"""Explicit published-pulse prior for additional PN cholinergic input pairs.

Gouwens2009's SINGLE-SITE modeled pulse: peak0.1nS, rise10us, decay600us,
E=-10mV. Assigning this pulse once per canonical presynaptic pair is a new,
unidentified efficacy hypothesis, not measured g/contact. No W/count scaling.
Canonical rates are proxies. Their deterministic convolution is the expected
conductance of this pulse model, not sampled vesicles or observed spikes.
"""
import hashlib,json
import numpy as np
from pn_coupled_ionic import GAMMA

SCHEMA='PN_cholinergic_pair_rate_prior_v1'
RISE_S=1e-5
DECAY_S=6e-4
PEAK_NS=.1
REVERSAL_MV=-10.
PEAK_TIME_S=np.log(DECAY_S/RISE_S)/(1/RISE_S-1/DECAY_S)
NORMALIZATION=np.exp(-PEAK_TIME_S/DECAY_S)-np.exp(-PEAK_TIME_S/RISE_S)
EVENT_AREA_NS_S=PEAK_NS*(DECAY_S-RISE_S)/NORMALIZATION


def linear_rate_filter(state,rate_start_hz,rate_end_hz,duration_s,tau_s):
    """Exact convolution of a linearly varying nonnegative event intensity.

    State counts exponentially weighted events; its derivative is rate-x/tau.
    Positive weights avoid cancellation for declining input near zero.
    """
    z=duration_s/tau_s;a=-np.expm1(-z)
    # 1-(1-exp(-z))/z; series avoids cancellation at very small intervals.
    b=(z/2-z*z/6+z**3/24-z**4/120) if z<1e-4 else 1-a/z
    return np.exp(-z)*state+tau_s*((a-b)*rate_start_hz+b*rate_end_hz)


class CholinergicPairRatePrior:
    def __init__(self,source_ids,*,time_ns,provenance):
        ids=np.asarray(source_ids)
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids<=0) or np.any(ids[1:]<=ids[:-1])
            or type(time_ns) is not int or time_ns<0 or not isinstance(provenance,str) or not provenance.strip()):
            raise ValueError('Sorted unique source identities, integer origin and explicit prior provenance required')
        self.source_ids=np.frombuffer(ids.astype(np.int64).tobytes(),dtype=np.int64)
        self.time_ns=time_ns;self.origin_ns=time_ns;self.fast=np.zeros(len(ids));self.slow=self.fast.copy()
        self.provenance=provenance
        h=hashlib.sha256((SCHEMA+provenance+repr((time_ns,RISE_S,DECAY_S,PEAK_NS,REVERSAL_MV))).encode());h.update(self.source_ids.tobytes());self.identity=h.hexdigest()

    def _rates(self,value):
        r=np.asarray(value)
        if r.shape!=self.fast.shape or r.dtype.kind not in 'fiu' or not np.isfinite(r).all() or np.any(r<0):
            raise ValueError('Finite nonnegative event-rate proxies in hertz for every named pair required')
        return r.astype(float,copy=False)

    def conductance_nS(self):return PEAK_NS*(self.slow-self.fast)/NORMALIZATION

    def preview(self,dt_ns,rate_start_hz,rate_end_hz):
        if type(dt_ns) is not int or dt_ns<=0:raise ValueError('Positive integer interval required')
        a,b=self._rates(rate_start_hz),self._rates(rate_end_hz);samples=[]
        with np.errstate(over='ignore',invalid='ignore'):
            for fraction in (GAMMA,1.):
                end=(1-fraction)*a+fraction*b;dt=dt_ns*1e-9*fraction
                x=linear_rate_filter(self.fast,a,end,dt,RISE_S)
                y=linear_rate_filter(self.slow,a,end,dt,DECAY_S)
                g=PEAK_NS*(y-x)/NORMALIZATION
                if not np.isfinite(g).all() or np.any(g<0):raise ValueError('Invalid predicted receptor conductance; no clipping')
                samples.append(g)
        candidate=dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns+dt_ns,fast=x,slow=y)
        self._validated(candidate)
        return samples,candidate

    def _validated(self,state):
        if (set(state)!={'schema','identity','time_ns','fast','slow'} or state['schema']!=SCHEMA or state['identity']!=self.identity
            or type(state['time_ns']) is not int or state['time_ns']<self.origin_ns):raise ValueError('Wrong receptor identity or clock')
        x,y=np.asarray(state['fast']),np.asarray(state['slow'])
        if (x.shape!=self.fast.shape or y.shape!=x.shape or x.dtype.kind not in 'fiu' or y.dtype.kind not in 'fiu'
            or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x<0) or np.any(y<x)):
            raise ValueError('Finite causal nonnegative pulse-filter states required')
        return x.astype(float,copy=True),y.astype(float,copy=True)

    def state_dict(self):return dict(schema=SCHEMA,identity=self.identity,time_ns=self.time_ns,fast=self.fast.copy(),slow=self.slow.copy())

    def load_state_dict(self,state):
        x,y=self._validated(state);self.fast=x;self.slow=y;self.time_ns=state['time_ns']

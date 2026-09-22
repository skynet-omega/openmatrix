"""Causal event-to-release proxy at the twelve existing WT9 axonal ports.

Port peaks, not soma/SIZ rate, drive release. Calcium, vesicle probability and
the correspondence between WT9 distance bands and canonical gamma1-5 remain
unidentified. Anatomical gamma-lobe fractions receive an area-weighted axonal
summary; no individual bouton or gamma subcompartment localization is claimed.
"""
import copy
import numpy as np


class AxonalRelease:
    def __init__(self, voltage, initial_q, initial_s, caps, tau, synaptic_tau,
                 weights, *, state=None):
        self.caps=np.asarray(caps,dtype=float).copy()
        self.tau=np.asarray(tau,dtype=float).copy()
        self.synaptic_tau=float(synaptic_tau)
        self.weights=np.asarray(weights,dtype=float).copy()
        n=len(self.caps);shape=(n,len(self.weights))
        if (np.asarray(voltage).shape!=shape or self.tau.shape!=(n,)
                or np.any(self.caps<=0) or np.any(self.tau<=0) or self.synaptic_tau<=0 or not np.isfinite(self.synaptic_tau)
                or np.any(self.weights<0) or not np.isclose(self.weights.sum(),1.,atol=1e-12)
                or not all(np.isfinite(x).all() for x in [voltage,self.caps,self.tau,self.weights])):
            raise ValueError('Invalid axonal release parameters')
        self.state=dict(q=np.broadcast_to(np.asarray(initial_q,dtype=float)[:,None],shape).copy(),
            s=np.broadcast_to(np.asarray(initial_s,dtype=float)[:,None],shape).copy(),
            last_voltage=np.asarray(voltage,dtype=float).copy(),previous_slope=np.zeros(shape),
            trough=np.asarray(voltage,dtype=float).copy(),counts=np.zeros(shape,dtype=np.int64),
            clipped=np.zeros(shape,dtype=np.int64),elapsed_ns=0)
        if state is not None:self.state=copy.deepcopy(state)
        self.validate()

    def validate(self):
        d=self.state;shape=(len(self.caps),len(self.weights))
        if set(d)!={'q','s','last_voltage','previous_slope','trough','counts','clipped','elapsed_ns'}:
            raise ValueError('Incomplete axonal release state')
        if type(d['elapsed_ns']) is not int or d['elapsed_ns']<0:raise ValueError('Invalid axonal clock')
        for k,v in d.items():
            if k=='elapsed_ns':continue
            dtype=np.int64 if k in ['counts','clipped'] else np.float64
            if not isinstance(v,np.ndarray) or v.shape!=shape or v.dtype!=dtype or not np.isfinite(v).all():
                raise ValueError('Invalid axonal field '+k)
            if k in ['q','s'] and np.any((v<0)|(v>1)):raise ValueError('Invalid normalized release')
            if k in ['counts','clipped'] and np.any(v<0):raise ValueError('Negative event ledger')

    def advance(self, voltage, dt_ns, suppression):
        if type(dt_ns) is not int or not 0<dt_ns<=25000:raise ValueError('Axonal sampling must resolve spatial steps')
        v=np.asarray(voltage);d=self.state;gain=np.asarray(suppression)
        if (v.shape!=d['q'].shape or gain.shape!=self.caps.shape or not np.isfinite(v).all()
                or not np.isfinite(gain).all() or np.any((gain<0)|(gain>1))):raise ValueError('Invalid release input')
        dt=dt_ns*1e-9;tq=self.tau[:,None];ts=self.synaptic_tau
        eq=np.exp(-dt/tq);es=np.exp(-dt/ts)
        equal=np.isclose(tq,ts,rtol=0.,atol=1e-15)
        factor=np.empty_like(tq)
        np.divide(tq*(eq-es),tq-ts,out=factor,where=~equal)
        factor[equal]=dt/ts*es
        # Events are detected at the right endpoint. Their release begins in
        # the following interval; no future voltage or retroactive impulse.
        s=d['s']*es+gain[:,None]*d['q']*factor
        slope=v-d['last_voltage'];peak=(d['previous_slope']>0)&(slope<=0)
        events=peak&(d['last_voltage']>-40.)&(d['last_voltage']-d['trough']>=20.)
        q=d['q']*eq+events/(self.caps[:,None]*tq)
        d['clipped']+=(q>1).astype(np.int64);d['q']=np.minimum(q,1.);d['s']=s
        d['counts']+=events.astype(np.int64)
        d['trough']=np.where(peak,v,np.minimum(d['trough'],v))
        d['last_voltage']=v.copy();d['previous_slope']=slope;d['elapsed_ns']+=dt_ns
        return events

    def summaries(self):return self.state['q']@self.weights,self.state['s']@self.weights
    def state_dict(self):return copy.deepcopy(self.state)


class AxonalSamplingBatch:
    """Sample after each unchanged spatial solver step; delegate parent state."""
    def __init__(self,base,publisher,gain):self.base=base;self.publisher=publisher;self.gain=gain
    def __getattr__(self,name):return getattr(self.base,name)
    def advance(self,dt_ns,ge_nS,gi_nS,*,inner_step_ns=25000,**kwargs):
        remaining=dt_ns
        while remaining:
            ns=min(remaining,inner_step_ns)
            self.base.advance(ns,ge_nS,gi_nS,inner_step_ns=ns,**kwargs)
            self.publisher.advance(self.base.host(self.base.delta[:,5:])+self.base.rest,ns,self.gain())
            remaining-=ns
        return self.base.host(self.base.q)

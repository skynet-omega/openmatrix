"""Batched WT9 spatial dynamics with native gates and explicit electrical ports.

Same Galerkin equations as kc_projected_active, with soma/dendrite/SIZ/axon
conductance distributions. GPU execution is optional for isolated comparisons;
the canonical CNS requires CUDA. Voltage is never reset to manufacture spikes.
"""
from pathlib import Path
import copy
import numpy as np
from session_io import sha256


def cell_parameters(artifact):
    with np.load(artifact,allow_pickle=False) as z:
        p={k:z[k].copy() for k in ['C_nF','G_nS','channel_G_nS','channel_b_nS','observation_basis']}
        P=z['basis'];B=z['port_weights'];cap=z['full_capacitance_nF'];names=z['port_names'].tolist()
        masks=[B[:,0]>0,B[:,1]>0,np.any(B[:,2:5]>0,axis=1),np.any(B[:,5:]>0,axis=1)]
        weights=np.stack([cap*m/np.sum(cap*m) for m in masks])
        p['shunt_G']=np.stack([P.T@(w[:,None]*P) for w in weights]);p['shunt_b']=weights@P
        p['rest_mV']=float(z['rest_mV']);p['port_names']=names
    p.update(artifact_sha256=sha256(artifact),domain_names=['soma_proxy','dendrites','SIZ_proxy','axon'])
    return p


class ProjectedKcBatch:
    def __init__(self,parameters,voltage_soma_mV,q,caps,tau,*,backend='cuda',state=None):
        self.parameters=copy.deepcopy(parameters);self.backend=backend
        if backend=='cuda':
            import cupy as xp
        elif backend=='numpy':xp=np
        else:raise ValueError('Unknown numerical backend')
        self.xp=xp;p=parameters;self.n=len(q);self.ports=len(p['port_names']);self.rest=p['rest_mV']
        self.C=xp.asarray(p['C_nF']);self.G=xp.asarray(p['G_nS'])
        self.chanG=xp.asarray(p['channel_G_nS'].reshape(-1,self.ports*self.ports))
        self.chanb=xp.asarray(p['channel_b_nS'].reshape(-1,self.ports))
        self.ena=xp.tile(xp.asarray([60.,60.,-80.])-self.rest,self.ports)
        self.shuntG=xp.asarray(p['shunt_G'].reshape(4,-1));self.shuntb=xp.asarray(p['shunt_b'])
        self.obs=xp.asarray(p['observation_basis']);self.caps=xp.asarray(caps);self.tau=xp.asarray(tau)
        # Only soma voltage is inherited. Hidden ports start at WT9 leak rest;
        # the soma coordinate enforces its observed midpoint voltage exactly.
        delta=np.zeros((self.n,self.ports));delta[:,0]=(voltage_soma_mV-self.rest)/p['observation_basis'][0,0]
        self.delta=xp.asarray(delta);self.gates=self.rates(self.delta+self.rest)[0]
        self.q=xp.asarray(q,dtype=xp.float64).copy();self.counts=xp.zeros(self.n,dtype=xp.int64)
        self.last_siz=self.observe()[:,1].copy();self.previous_slope=xp.zeros(self.n)
        self.trough=self.last_siz.copy();self.clipped=xp.zeros(self.n,dtype=xp.int64)
        self.elapsed_ns=0
        if state is not None:self.load_state(state)

    def rates(self,v):
        xp=self.xp
        steady=xp.stack((1/(1+xp.exp(-.1121*(v+29.13))),1/(1+xp.exp(.2*(v+47))),
                         1/(1+xp.exp(-.2717*(v+48.77))),1/(1+xp.exp(-.0502*(v+12.85)))),axis=-1)
        tau=xp.stack((.1270+3.434/(1+xp.exp((v+45.35)/5.98)),.36+xp.exp((v+20.65)/-10.47),
                      xp.ones_like(v),2.03+1.96/(1+xp.exp((v-30.83)/3.12))),axis=-1)/1000
        return steady,tau

    def observe(self):return self.rest+self.delta@self.obs.T

    def host(self,x):return self.xp.asnumpy(x) if self.backend=='cuda' else np.asarray(x).copy()

    def advance(self,dt_ns,ge_nS,gi_nS,*,current_pA=None,inner_step_ns=25000):
        xp=self.xp
        if type(dt_ns) is not int or dt_ns<=0 or type(inner_step_ns) is not int or not 0<inner_step_ns<=25000:
            raise ValueError('Invalid spatial integration clock')
        for x in [ge_nS,gi_nS]:
            if x.shape!=(self.n,4) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid domain conductance')
        ge=xp.asarray(ge_nS);gi=xp.asarray(gi_nS)
        conductance=(ge+gi)@self.shuntG
        drive=(-self.rest*ge+(-68.-self.rest)*gi)@self.shuntb
        current=xp.zeros((self.n,self.ports)) if current_pA is None else xp.asarray(current_pA)
        if current.shape!=(self.n,self.ports):raise ValueError('Invalid current ports')
        remaining=dt_ns
        while remaining:
            ns=min(remaining,inner_step_ns);dt=ns*1e-9
            steady,tau=self.rates(self.delta+self.rest)
            gates=steady+(self.gates-steady)*xp.exp(-dt/tau)
            m,h,p,n=(gates[:,:,j] for j in range(4));f=xp.stack((m*m*m*h,p,n**4),axis=-1).reshape(self.n,-1)
            lhs=(f@self.chanG+conductance).reshape(self.n,self.ports,self.ports)+self.G+self.C/dt
            rhs=self.delta@(self.C/dt).T+(f*self.ena)@self.chanb+drive+current
            self.delta=xp.linalg.solve(lhs,rhs[:,:,None])[:,:,0];self.gates=gates
            siz=self.observe()[:,1];slope=siz-self.last_siz
            event=(self.previous_slope>0)&(slope<=0)&(self.last_siz>-40.)&(self.last_siz-self.trough>=20.)
            self.q*=xp.exp(-dt/self.tau)
            value=self.q+event/(self.caps*self.tau)
            self.clipped+=(value>1).astype(xp.int64);self.q=xp.minimum(value,1.)
            self.counts+=event.astype(xp.int64)
            # Past-trough prominence is causal. Reset at every local maximum,
            # even if it fails the event criterion, then track the next trough.
            peak=(self.previous_slope>0)&(slope<=0)
            self.trough=xp.where(peak,siz,xp.minimum(self.trough,siz))
            self.previous_slope=slope;self.last_siz=siz;remaining-=ns
        self.elapsed_ns+=dt_ns
        return self.host(self.q)

    def state_dict(self):
        result={k:self.host(getattr(self,k)) for k in ['delta','gates','q','counts','last_siz','previous_slope','trough','clipped']}
        result.update(elapsed_ns=self.elapsed_ns,artifact_sha256=self.parameters['artifact_sha256'])
        return result

    def load_state(self,state):
        expected=self.state_dict()
        if set(state)!=set(expected) or state['artifact_sha256']!=expected['artifact_sha256']:raise ValueError('Wrong spatial state schema/source')
        for k,v in expected.items():
            if not isinstance(v,np.ndarray):continue
            x=np.asarray(state[k])
            if x.shape!=v.shape or x.dtype!=v.dtype or not np.isfinite(x).all():raise ValueError('Invalid spatial state '+k)
            if k in ['counts','clipped'] and np.any(x<0):raise ValueError('Negative event count')
            if k in ['q','gates'] and np.any((x<0)|(x>1)):raise ValueError('Invalid gate or output state')
        if type(state['elapsed_ns']) is not int or state['elapsed_ns']<0:raise ValueError('Invalid spatial clock')
        for k,v in expected.items():
            if isinstance(v,np.ndarray):setattr(self,k,self.xp.asarray(state[k]).copy())
        self.elapsed_ns=state['elapsed_ns']

"""Exact linear q/s flow with ordered effective clipped jumps, no neural fit."""
import math
import numpy as np
from numba import njit


def kernel(dt,tau_q,tau_s,xp=np):
    """Convolution exp(-t/tq) into s'= (q-s)/ts, stable at tq=ts."""
    tq=xp.asarray(tau_q);t=xp.asarray(dt);z=t*(1/tau_s-1/tq)
    den=1-tau_s/tq
    # Each exponent is nonpositive; no overflowing exp(z) for disparate taus.
    a=xp.exp(-t/tq)*(-xp.expm1(-xp.maximum(z,0)))
    b=xp.exp(-t/tau_s)*xp.expm1(xp.minimum(z,0))
    ordinary=xp.divide(xp.where(z>=0,a,b),xp.where(den==0,1.,den))
    series=t/tau_s*xp.exp(-t/tau_s)*(1+z/2+z*z/6+z*z*z/24)
    return xp.where(xp.abs(z)<1e-5,series,ordinary)


class Waveform:
    def __init__(self,q,s,tau_q,tau_s,duration):
        self.q=np.asarray(q,dtype=float).copy();self.s=np.asarray(s,dtype=float).copy()
        self.tau=np.broadcast_to(np.asarray(tau_q,dtype=float),self.q.shape).copy()
        self.ts=float(tau_s);self.duration=float(duration);self.times=[];self.rows=[];self.jumps=[];self.posts=[]
        if (self.q.ndim!=1 or self.s.shape!=self.q.shape or not np.isfinite([self.ts,self.duration]).all() or self.ts<=0 or self.duration<=0 or
            any(not np.isfinite(x).all() for x in (self.q,self.s,self.tau)) or np.any(self.tau<=0) or np.any((self.q<0)|(self.q>1)|(self.s<0)|(self.s>1))):
            raise ValueError('Invalid initial filter state/domain')
        self.device=None

    def add(self,times,rows,jumps,posts=None):
        t=np.asarray(times,dtype=float).ravel();r=np.asarray(rows);j=np.asarray(jumps,dtype=float).ravel()
        post=np.full(j.shape,np.nan) if posts is None else np.asarray(posts,dtype=float).ravel()
        if (r.dtype.kind not in 'iu' or r.ndim!=1 or r.shape!=t.shape or j.shape!=t.shape or not np.isfinite(t).all() or not np.isfinite(j).all()
            or post.shape!=t.shape or np.any((t<0)|(t>self.duration+1e-15)) or np.any((r<0)|(r>=len(self.q))) or np.any(j<0)
            or np.any(np.isinf(post))):
            raise ValueError('Invalid effective event jumps')
        keep=(j>0)|np.isfinite(post)
        self.times.extend(t[keep]);self.rows.extend(r[keep]);self.jumps.extend(j[keep]);self.posts.extend(post[keep]);self.device=None
        if hasattr(self,'packed'):del self.packed

    def at(self,time,xp=np):
        if not np.isscalar(time) or not np.isfinite(time) or time < -1e-15 or time>self.duration+1e-15:raise ValueError('Waveform queried outside interval')
        time=max(0.,min(float(time),self.duration))
        if xp is not np and any(np.isfinite(self.posts)):
            # Reference backend only. The protected native path uses the
            # dedicated SET-aware CUDA FilterPorts below.
            a,b=self.at(time,np);return xp.asarray(a),xp.asarray(b)
        if xp is np:q,s,tau=self.q,self.s,self.tau;t=np.asarray(self.times);rows=np.asarray(self.rows,dtype=np.int64);j=np.asarray(self.jumps)
        else:
            if self.device is None:self.device=tuple(xp.asarray(x) for x in (self.q,self.s,self.tau,np.asarray(self.times),np.asarray(self.rows,dtype=np.int64),np.asarray(self.jumps)))
            q,s,tau,t,rows,j=self.device
        qnow=q*xp.exp(-time/tau);snow=s*xp.exp(-time/self.ts)+q*kernel(time,tau,self.ts,xp)
        # Indexing each event separately has deterministic accumulation order.
        if len(self.times):
            if xp is np:
                mask=t<=time;u=time-t[mask];rr=rows[mask];jj=j[mask]
                np.add.at(qnow,rr,jj*np.exp(-u/tau[rr]));np.add.at(snow,rr,jj*kernel(u,tau[rr],self.ts))
            else:
                # CPU-created sparse per-cell event matrix avoids nondeterministic atomics.
                # This path is replaced by a cached per-cell representation below.
                return self._device_at(time,xp)
        if xp is np and any(np.isfinite(self.posts)):
            affected={int(self.rows[i]) for i,post in enumerate(self.posts) if np.isfinite(post)}
            for row in affected:
                qv=float(self.q[row]);sv=float(self.s[row]);prev=0.
                ordered=sorted((i for i,r in enumerate(self.rows) if r==row),key=lambda i:(self.times[i],i))
                for i in ordered:
                    te=float(self.times[i])
                    if te>time:break
                    dt=te-prev
                    sv=sv*math.exp(-dt/self.ts)+qv*float(kernel(dt,self.tau[row],self.ts))
                    qv=qv*math.exp(-dt/self.tau[row])
                    qv=float(self.posts[i]) if np.isfinite(self.posts[i]) else qv+float(self.jumps[i])
                    prev=te
                dt=time-prev
                qnow[row]=qv*math.exp(-dt/self.tau[row])
                snow[row]=sv*math.exp(-dt/self.ts)+qv*float(kernel(dt,self.tau[row],self.ts))
        return qnow,snow

    def _device_at(self,time,xp):
        if not hasattr(self,'packed'):
            n=len(self.q);count=np.bincount(np.asarray(self.rows,dtype=np.int64),minlength=n);width=int(count.max(initial=0))
            t=np.full((n,width),np.inf);j=np.zeros((n,width));used=np.zeros(n,dtype=int)
            for tt,rr,jj in zip(self.times,self.rows,self.jumps):t[rr,used[rr]]=tt;j[rr,used[rr]]=jj;used[rr]+=1
            self.packed=(xp.asarray(t),xp.asarray(j))
        q,s,tau,*_=self.device;t,j=self.packed
        elapsed=xp.maximum(0,time-t);weights=xp.where(t<=time,j,0.)
        return (q*xp.exp(-time/tau)+(weights*xp.exp(-elapsed/tau[:,None])).sum(axis=1),
                s*xp.exp(-time/self.ts)+q*kernel(time,tau,self.ts,xp)+(weights*kernel(elapsed,tau[:,None],self.ts,xp)).sum(axis=1))


@njit(cache=True,fastmath=False)
def lif_record(voltage,refractory_left,counts,q,vinf,rate,reset,threshold,caps,filter_tau,dt,refractory=.0022):
    """Same analytic held-input LIF as the parent, additionally retain effective jumps."""
    max_events=2+int(dt/refractory);times=np.full((len(q),max_events),np.nan);jumps=np.zeros_like(times);posts=np.full_like(times,np.nan);clipped=0
    for k in range(len(voltage)):
        t=0.;v=voltage[k];left=refractory_left[k];value=q[k]*math.exp(-dt/filter_tau[k]);n=0
        while t<dt:
            wait=min(left,dt-t);t+=wait;left-=wait
            if t>=dt:break
            if vinf[k]<=threshold[k]:v=vinf[k]+(v-vinf[k])*math.exp(-rate[k]*(dt-t));t=dt;break
            crossing=math.log1p((threshold[k]-v)/(vinf[k]-threshold[k]))/rate[k]
            if crossing>dt-t:v=vinf[k]+(v-vinf[k])*math.exp(-rate[k]*(dt-t));t=dt;break
            t+=max(0.,crossing);counts[k]+=1
            before=value*math.exp((dt-t)/filter_tau[k]);event_value=before+1./(caps[k]*filter_tau[k])
            if event_value>1.:clipped+=1;event_value=1.
            if n>=max_events:raise ValueError('Event accounting capacity exceeded')
            times[k,n]=t;jumps[k,n]=event_value-before;posts[k,n]=event_value;n+=1
            value=event_value*math.exp(-(dt-t)/filter_tau[k]);v=reset[k];left=refractory
        voltage[k]=v;refractory_left[k]=left;counts[k]+=0;q[k]=value
    return clipped,times,jumps,posts

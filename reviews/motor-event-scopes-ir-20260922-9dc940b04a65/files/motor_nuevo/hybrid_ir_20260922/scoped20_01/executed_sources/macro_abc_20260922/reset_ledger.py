"""ADD/SET ledger for the legacy affine-source adapter, independent of installation."""
import numpy as np

def make_ledger(original_wave, kernel):
 class Ledger(original_wave):
  def __init__(self,*args,**kw):super().__init__(*args,**kw);self.post_values=[]
  def add(self,times,rows,jumps,post_values=None):
   t=np.asarray(times,dtype=float).ravel();r=np.asarray(rows);j=np.asarray(jumps,dtype=float).ravel()
   posts=[None]*len(t) if post_values is None else list(post_values)
   if r.dtype.kind not in 'iu' or r.ndim!=1 or r.shape!=t.shape or j.shape!=t.shape or len(posts)!=len(t):raise ValueError('Invalid event layout')
   if not np.isfinite(t).all() or not np.isfinite(j).all() or np.any((t<0)|(t>self.duration+1e-15)) or np.any((r<0)|(r>=len(self.q))):raise ValueError('Invalid event metadata')
   normalized=[]
   for value,jump in zip(posts,j):
    if value is None:
     if jump<0:raise ValueError('Negative ADD is outside this legacy source contract')
     normalized.append(None)
    else:
     value=float(value)
     if not np.isfinite(value):raise ValueError('Nonfinite SET')
     normalized.append(value)
   # Validate everything before publishing; SET is an operation, not a positive delta.
   keep=[i for i in range(len(t)) if normalized[i] is not None or j[i]>0]
   self.times.extend(t[keep]);self.rows.extend(r[keep]);self.jumps.extend(j[keep]);self.post_values.extend(normalized[i] for i in keep)
   self.device=None
   if hasattr(self,'packed'):del self.packed
  def at(self,time,xp=np):
   if not any(v is not None for v in self.post_values):return super().at(time,xp)
   if not np.isscalar(time) or not np.isfinite(time) or time < -1e-15 or time>self.duration+1e-15:raise ValueError('Waveform queried outside interval')
   time=max(0.,min(float(time),self.duration))  # Same time-boundary convention as the parent.
   q=xp.asarray(self.q).copy();s=xp.asarray(self.s).copy();tau=xp.asarray(self.tau);previous=xp.zeros(len(q))
   order=np.lexsort((np.arange(len(self.times)),np.asarray(self.times),np.asarray(self.rows)))
   for i in order:
    at=float(self.times[i]);row=int(self.rows[i])
    if at>time:continue
    delta=at-previous[row]
    s[row]=s[row]*xp.exp(-delta/self.ts)+q[row]*kernel(delta,tau[row],self.ts,xp)
    q[row]=q[row]*xp.exp(-delta/tau[row])
    q[row]=q[row]+self.jumps[i] if self.post_values[i] is None else self.post_values[i]
    previous[row]=at
   delta=time-previous
   s=s*xp.exp(-delta/self.ts)+q*kernel(delta,tau,self.ts,xp)
   q=q*xp.exp(-delta/tau)
   return q,s
 return Ledger

"""Bounded generic affine block programs, with exact event ordering semantics.

For nonlinear/recurrent coefficients this adapter refuses inference: the caller
must provide a separately validated coupling method. No approximate averaging.
"""
from pathlib import Path
import numpy as np,cupy as cp

class AffinePorts:
 def __init__(self,matrix,times,jumps,initial):
  a=np.asarray(matrix,dtype=np.float64);t=np.asarray(times,dtype=np.float64)
  j=np.asarray(jumps,dtype=np.float64);y=np.asarray(initial,dtype=np.float64)
  if y.ndim!=2 or not 1<=y.shape[1]<=32:raise ValueError('Declared block dimension must be1..32')
  self.n,self.dim=y.shape;self.segments=len(t)-1
  if a.shape!=(self.n,self.segments,self.dim,self.dim) or j.shape!=(self.n,self.segments,self.dim):raise ValueError('Affine program shape mismatch')
  if self.segments<1 or t[0]!=0 or np.any(np.diff(t)<=0) or not all(np.isfinite(x).all() for x in (a,t,j,y)):raise ValueError('Nonfinite or unordered program')
  self.end=float(t[-1]);self.arrays=[cp.asarray(np.ascontiguousarray(x)) for x in (a,t,j,y)]
  self.kernel=cp.RawKernel((Path(__file__).parent/'affine_ports.cu').read_text(),'affine_prefix',options=('--fmad=false',))
 def prefix(self,until):
  if not np.isfinite(until) or not 0<=until<=self.end:raise ValueError('Prefix outside physical program')
  out=cp.empty((self.n,self.dim));status=cp.zeros(self.n,dtype=cp.int32)
  self.kernel((self.n,),(32,),(np.int32(self.n),np.int32(self.dim),np.int32(self.segments),*self.arrays,np.float64(until),out,status))
  if bool(cp.any(status)):raise RuntimeError('Affine work/nonfinite bound: '+str(status.get().tolist()))
  return out

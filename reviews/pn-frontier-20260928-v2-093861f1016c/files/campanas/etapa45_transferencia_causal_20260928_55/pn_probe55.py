"""Read-only witnesses of the ALPN common output actually used by FP32 CSR."""
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
def installer(brain,stimulus):
 import cupy as cp,fp32_operator
 from pilot52_owners import installer as parent
 ob,undo_parent=parent('parent')(brain,stimulus);old=fp32_operator.FastCSR
 with np.load(H/'reference/PN_donors.npz') as z:rows=cp.asarray(z['rows'],dtype=cp.int64)
 p=dict(rows=rows,first=cp.zeros(686,cp.float32),last=cp.zeros(686,cp.float32),lo=cp.zeros(686,cp.float32),hi=cp.zeros(686,cp.float32),counts=cp.zeros(686,cp.uint64),bad=cp.zeros(1,cp.uint64))
 observe=cp.RawKernel(r'''extern "C" __global__ void pn_output(const long long* rows,const float* release,float* first,float* last,float* lo,float* hi,unsigned long long* counts,unsigned long long* bad){int j=threadIdx.x+blockIdx.x*blockDim.x;if(j<686){float v=release[rows[j]];unsigned long long c=counts[j];if(c==0){first[j]=v;lo[j]=v;hi[j]=v;}else{lo[j]=fminf(lo[j],v);hi[j]=fmaxf(hi[j],v);}last[j]=v;counts[j]=c+1;if(!isfinite(v))atomicAdd(bad,1ULL);}}''','pn_output')
 class Audited(old):
  def __init__(self,b,**kw):
   super().__init__(b,**kw);original=self.kernel
   def kernel(grid,block,args):
    observe((6,),(128,),(rows,args[4],p['first'],p['last'],p['lo'],p['hi'],p['counts'],p['bad']))
    return original(grid,block,args)
   self.kernel=kernel
 fp32_operator.FastCSR=Audited;ob.pn_probe=p
 def undo():fp32_operator.FastCSR=old;undo_parent()
 return ob,undo

def reset(run):
 p=run.observer.pn_probe;p['counts'].fill(0);p['bad'].fill(0)

def sample(run):
 p=run.observer.pn_probe;c=p['counts'].get()
 if not np.all(c==c[0]) or c[0]==0 or int(p['bad'].get()[0]):raise ValueError('PN consumption witness failed')
 return {k:p[k].get() for k in ['first','last','lo','hi','counts']}

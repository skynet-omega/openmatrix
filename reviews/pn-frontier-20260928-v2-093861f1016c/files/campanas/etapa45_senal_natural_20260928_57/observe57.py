"""Read-only ORN generic-output witness, alongside the unchanged PN55 witness.
This is not the DM1 fine synaptic-filter input; those 74 filters remain in traces.
"""
import numpy as np
def installer(brain,stimulus):
 import cupy as cp,fp32_operator,pn_probe55
 ob,undo_parent=pn_probe55.installer(brain,stimulus);old=fp32_operator.FastCSR
 ids=np.asarray(stimulus.spec['ids'],np.int64);rows=np.searchsorted(brain.brain.node_ids,ids)
 if len(ids)!=694 or not np.array_equal(brain.brain.node_ids[rows],ids):raise ValueError('ORN57 identity')
 p=dict(rows=cp.asarray(rows,dtype=cp.int64),first=cp.zeros(694,cp.float32),last=cp.zeros(694,cp.float32),lo=cp.zeros(694,cp.float32),hi=cp.zeros(694,cp.float32),counts=cp.zeros(694,cp.uint64),bad=cp.zeros(1,cp.uint64))
 kernel=cp.RawKernel(r'''extern "C" __global__ void orn57(const long long* rows,const float* release,float* first,float* last,float* lo,float* hi,unsigned long long* counts,unsigned long long* bad){int j=threadIdx.x+blockIdx.x*blockDim.x;if(j<694){float v=release[rows[j]];unsigned long long c=counts[j];if(c==0){first[j]=v;lo[j]=v;hi[j]=v;}else{lo[j]=fminf(lo[j],v);hi[j]=fmaxf(hi[j],v);}last[j]=v;counts[j]=c+1;if(!isfinite(v))atomicAdd(bad,1ULL);}}''','orn57')
 class Observed(old):
  def __init__(self,b,**kw):
   super().__init__(b,**kw);original=self.kernel
   def call(grid,block,args):
    kernel((6,),(128,),(p['rows'],args[4],p['first'],p['last'],p['lo'],p['hi'],p['counts'],p['bad']))
    return original(grid,block,args)
   self.kernel=call
 fp32_operator.FastCSR=Observed;ob.orn_probe57=p
 def undo():fp32_operator.FastCSR=old;undo_parent()
 return ob,undo
def reset(run):
 import pn_probe55
 pn_probe55.reset(run);p=run.observer.orn_probe57;p['counts'].fill(0);p['bad'].fill(0)
def sample(run):
 p=run.observer.orn_probe57;c=p['counts'].get()
 if c[0]==0 or not np.all(c==c[0]) or int(p['bad'].get()[0]):raise ValueError('ORN57 every-RHS audit')
 return {k:p[k].get() for k in ['first','last','lo','hi','counts']}

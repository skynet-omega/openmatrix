"""Evaluator-only hold/pulse of declared ALPN output at the FP32 CSR boundary."""
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
CUDA=r'''
extern "C" __global__ void terminal56(const long long* rows,float* release,
 const float* fixed,const int* mode,float* first,float* last,float* lo,float* hi,
 unsigned long long* counts,unsigned long long* bad) {
 int j=threadIdx.x+blockIdx.x*blockDim.x;if(j>=686)return;
 long long row=rows[j];float v=release[row];unsigned long long c=counts[j];
 if(c==0){first[j]=v;lo[j]=v;hi[j]=v;}else{lo[j]=fminf(lo[j],v);hi[j]=fmaxf(hi[j],v);}
 last[j]=v;counts[j]=c+1;int m=mode[0];
 if(!isfinite(v)||m<0||m>2)atomicAdd(bad,1ULL);
 if(m==1)release[row]=fixed[j];
 else if(m==2)release[row]=v;
 if(!isfinite(release[row]) || (m==1 && release[row]!=fixed[j]) || (m!=1 && release[row]!=v))atomicAdd(bad,1ULL);
}
'''
def installer(spec):
 def install(brain,stimulus):
  import cupy as cp,fp32_operator,pn_probe55
  ob,undo_parent=pn_probe55.installer(brain,stimulus);old=fp32_operator.FastCSR
  with np.load(H/'reference/TERMINALS.npz',allow_pickle=False) as z:
   rows=z['rows'].copy();ids=z['ids'].copy();value=z[spec['donor']].copy()
  if len(rows)!=686 or value.dtype!=np.float32 or not np.array_equal(brain.brain.node_ids[rows],ids):raise ValueError('terminal56 support')
  holder={'mode':cp.zeros(1,cp.int32),'rows':cp.asarray(rows),'fixed':cp.asarray(value),'first':cp.zeros(686,cp.float32),'last':cp.zeros(686,cp.float32),'lo':cp.zeros(686,cp.float32),'hi':cp.zeros(686,cp.float32),'counts':cp.zeros(686,cp.uint64),'bad':cp.zeros(1,cp.uint64)}
  kernel=cp.RawKernel(CUDA,'terminal56',options=('--fmad=false',))
  class Held(old):
   def __init__(self,b,**kw):
    super().__init__(b,**kw);consume=self.kernel
    def call(grid,block,args):
     kernel((6,),(128,),(holder['rows'],args[4],holder['fixed'],holder['mode'],holder['first'],holder['last'],holder['lo'],holder['hi'],holder['counts'],holder['bad']))
     return consume(grid,block,args)
    self.kernel=call
  fp32_operator.FastCSR=Held;ob.holder56=holder
  def undo():fp32_operator.FastCSR=old;undo_parent()
  return ob,undo
 return install
def reset(run,spec,j):
 import cupy as cp,pn_probe55
 pn_probe55.reset(run);p=run.observer.holder56
 mode=2 if spec['mode']=='identity' else int(spec['mode']=='hold' or (spec['mode']=='pulse' and j<=1))
 p['mode'].set(np.array([mode],np.int32));p['counts'].fill(0);p['bad'].fill(0);cp.cuda.get_current_stream().synchronize()
def sample(run):
 p=run.observer.holder56;c=p['counts'].get()
 if c[0]==0 or not np.all(c==c[0]) or int(p['bad'].get()[0]):raise ValueError('terminal56 every-RHS audit')
 return {**{k:p[k].get() for k in ['first','last','lo','hi','counts']},'mode':p['mode'].get(),'errors':p['bad'].get()}

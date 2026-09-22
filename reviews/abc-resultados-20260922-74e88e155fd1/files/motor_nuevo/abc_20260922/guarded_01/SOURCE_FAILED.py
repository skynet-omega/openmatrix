"""Instrumentation-only verification replay prompted by external source review.
No equation, step, candidate or scientific tolerance changes. Persistent flags are
checked before operations capable of masking NaN. C requires nonnegative release.
"""
from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from recurrence import Engine,write,digest
import numpy as np
import cupy as cp
H=Path(__file__).resolve().parent
CHECK='''extern "C" __global__ void inspect(int n,const double* x,unsigned int* flags,bool nonnegative){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){if(!isfinite(x[i]))atomicOr(flags,1U);if(nonnegative&&x[i]<0.)atomicOr(flags,2U);}}'''
class Guarded(Engine):
 def __init__(self,*args):
  super().__init__(*args);self.flags=cp.zeros(1,dtype=cp.uint32);self.inspect=cp.RawKernel(CHECK,'inspect');self.inspect.compile();self.last_s=cp.empty(self.n)
 def check(self,x,nonnegative=False):self.inspect(((x.size+255)//256,),(256,),(np.int32(x.size),x,self.flags,np.bool_(nonnegative)))
 def coefficient(self,s,quantum=None):
  self.check(s,quantum is not None);cp.copyto(self.last_s,s)
  super().coefficient(s,quantum);self.check(self.target);self.check(self.rate)
 def small(self,name,m,*args):
  # Inspect all float buffers before and after arithmetic, including residual inputs.
  arrays=[a for a in args if isinstance(a,cp.ndarray) and a.dtype==cp.float64]
  if name!='residual':
   # Only initialized input buffers: outputs are checked after execution.
   for a in arrays[:-1]:self.check(a)
  else:
   for a in arrays:self.check(a)
  super().small(name,m,*args)
  if name=='finish':self.check(args[0])
  elif arrays:self.check(arrays[-1])
 def reset(self,spec):
  super().reset(spec);self.flags.fill(0);cp.cuda.get_current_stream().synchronize()

def main():
 out=Path(sys.argv[2]);out.mkdir(exist_ok=False);c=json.loads((H/'contract.json').read_text())
 z=np.load(sys.argv[1],allow_pickle=False);d={k:z[k] for k in z.files};z=np.load(H/'initial.npz',allow_pickle=False);initial=np.r_[z['q'],z['s']]
 if digest(sys.argv[1])!=c['fixture_sha256']:raise ValueError('Wrong fixture')
 # Fixed finite-input magnitude checks make overflow inside the bounded coefficient
 # sum impossible at this fixture's weights, caps, gain and strictly positive tau.
 for k in ['weights','caps','tau','gain','theta','drive','photo']:
  if not np.isfinite(d[k]).all() or np.max(abs(d[k]),initial=0)>1e12:raise ValueError('Unsafe coefficient input bound')
 if min(d['tau'])<1e-12 or float(d['scale'])<0:raise ValueError('Outside supported coefficient domain')
 e=Guarded(d,initial);rows=[];start=time.perf_counter()
 specs=[{'id':'R'+str(i),'method':'rk4','h_s':h} for i,h in enumerate(c['reference_steps_s'])]+c['candidates']
 for condition in c['conditions']:
  for spec in specs:
   name=condition+'_'+spec['id'];r,state=e.run(spec,condition,.005,.0005);flag=int(e.flags.get()[0])
   if flag:raise ValueError(f'Invalid intermediate state {name}, flags={flag}')
   original=np.load(H/'runs_01'/name/'states.npz',allow_pickle=False)['state'];error=float(np.max(abs(state-original)))
   # Instrumentation may change atomic scheduling in C, not the operator definition.
   if error>1e-12:raise ValueError('Instrumentation changed trajectory beyond roundoff')
   row={'run':name,'persistent_flags':flag,'max_trajectory_change':error,'wall_s':r['wall_with_sampling_s']}
   values={'final_state':state[-1],'flags':e.flags.get()}
   if spec['method']=='incremental_rk4':
    aa=e.aa.copy();bb=e.bb.copy();previous=e.previous.copy();last=e.last_s.copy()
    e.raw_sums(previous);cp.cuda.get_current_stream().synchronize()
    drift=max(float(cp.max(cp.abs(aa-e.aa)/(1+cp.abs(e.aa))).get()),float(cp.max(cp.abs(bb-e.bb)/(1+cp.abs(e.bb))).get()))
    delta=float(cp.max(cp.abs(last-previous)).get())
    if drift>1e-10 or delta>spec['quantum']+1e-15:raise ValueError('Incremental cache invariant failed')
    row.update(cache_sum_relative_drift=drift,last_source_quantization_error=delta)
    values.update(cached_a=aa.get(),cached_b=bb.get(),previous=previous.get(),last_input=last.get())
   np.savez_compressed(out/(name+'.npz'),**values);rows.append(row);print(json.dumps(row),flush=True)
 # Deliberately demonstrate the legacy fmax NaN masking, then require sticky detection.
 y=cp.zeros(2*e.n);z=y.copy();z[0]=cp.nan;f=y.copy();maximum=y.copy();e.flags.fill(0)
 e.small('residual',2*e.n,y,z,f,f,np.float64(.000125),maximum)
 masked=float(maximum[0].get())==0.;detected=int(e.flags.get()[0])&1
 if not masked or not detected:raise ValueError('NaN masking corruption test failed')
 e.flags.fill(0);s=cp.ones(e.n);s[0]=-.01;e.check(s,True)
 if not int(e.flags.get()[0])&2:raise ValueError('Negative communicated release not rejected')
 result={'scope':'instrumentation replay, not new candidates','equations_steps_and_tolerances_unchanged':True,'rows':rows,'legacy_nan_masking_reproduced':masked,'persistent_nan_detection':bool(detected),'negative_release_detection':True,'elapsed_s':time.perf_counter()-start,'source_sha256':digest(__file__),'original_contract_sha256':digest(H/'contract.json')}
 write(out/'RESULT.json',result)
if __name__=='__main__':main()

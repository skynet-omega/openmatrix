"""Three engineering hypotheses on the full base graph, not the full heterogeneous fly."""
from pathlib import Path
import hashlib,json,time,sys,platform,resource
import numpy as np
import cupy as cp
from scipy.sparse import csr_matrix
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

class Engine:
 def __init__(self,data,initial):
  self.data=data;self.n=n=len(data['caps']);self.initial=initial
  self.d={k:cp.asarray(data[k]) for k in ('indptr','indices','weights','caps','visual','tau','gain','theta','drive','photo')}
  self.scale=np.float64(data['scale']);self.connected=np.bool_(data['connected'])
  self.y=cp.asarray(initial);self.z=cp.empty_like(self.y);self.k=[cp.empty_like(self.y) for _ in range(4)]
  self.target=cp.empty(n);self.rate=cp.empty(n);self.res=cp.zeros(2*n)
  self.aa=cp.empty(n);self.bb=cp.empty(n);self.previous=cp.empty(n);self.counts=cp.zeros(2,dtype=cp.uint64)
  opts=('--std=c++11','--fmad=false','--prec-div=true','--prec-sqrt=true')
  src=(HERE/'recurrence.cu').read_text();names=('rhs','stage','finish','trapezoid','residual','sums','scatter','cached_coefficient')
  self.kernels={name:cp.RawKernel(src,name,options=opts) for name in names}
  self.coeff=cp.RawKernel((HERE.parent/'connections.cu').read_text(),'coefficient',options=opts)
  for kernel in list(self.kernels.values())+[self.coeff]:kernel.compile()
  self.csc=None;self.evaluations=0
 def small(self,name,m,*args):self.kernels[name](((m+255)//256,),(256,),(np.int32(m),*args))
 def setup_csc(self):
  if self.csc is None:
   d=self.data;n=self.n
   c=csr_matrix((d['weights'],d['indices'],d['indptr']),shape=(n,n)).tocsc()
   if c.nnz!=len(d['weights']):raise ValueError('CSC changed stored edge count')
   self.csc=[cp.asarray(c.indptr.astype(np.int32)),cp.asarray(c.indices.astype(np.int32)),cp.asarray(c.data)]
 def raw_sums(self,s):
  d=self.d;n=self.n
  self.kernels['sums'](((n*32+255)//256,),(256,),(np.int32(n),d['indptr'],d['indices'],d['weights'],s,d['caps'],d['visual'],self.scale,self.connected,self.aa,self.bb))
 def coefficient(self,s,quantum=None):
  d=self.d;n=self.n;self.evaluations+=1
  if quantum is None:
   self.coeff(((n*32+255)//256,),(256,),(np.int32(n),d['indptr'],d['indices'],d['weights'],s,d['caps'],d['visual'],d['tau'],d['gain'],d['theta'],d['drive'],d['photo'],self.scale,self.connected,self.target,self.rate))
  else:
   self.small('scatter',n,*self.csc,s,self.previous,d['caps'],d['visual'],self.scale,self.connected,np.float64(quantum),self.aa,self.bb,self.counts)
   self.small('cached_coefficient',n,self.aa,self.bb,d['visual'],d['tau'],d['gain'],d['theta'],d['drive'],d['photo'],self.target,self.rate)
 def rhs(self,y,out,quantum=None):
  self.coefficient(y[self.n:],quantum);self.small('rhs',self.n,y,self.d['visual'],self.target,self.rate,out)
 def step(self,spec):
  n=self.n;h=np.float64(spec['h_s']);y=self.y;z=self.z;k=self.k;quantum=spec.get('quantum')
  self.rhs(y,k[0],quantum)
  if spec['method']=='trapezoid':
   cp.copyto(z,y)
   for _ in range(spec['iterations']):
    self.coefficient(z[n:]);self.small('trapezoid',n,y,k[0],self.d['visual'],self.target,self.rate,h,z)
   self.rhs(z,k[1]);self.small('residual',2*n,y,z,k[0],k[1],h,self.res);cp.copyto(y,z)
  else:
   for j,fraction in ((1,.5),(2,.5),(3,1.)):
    self.small('stage',2*n,y,k[j-1],h*fraction,z);self.rhs(z,k[j],quantum)
   self.small('finish',2*n,y,*k,h)
 def reset(self,spec):
  cp.copyto(self.y,cp.asarray(self.initial));self.res.fill(0);self.counts.fill(0);self.d['photo'].fill(0);self.evaluations=0
  if spec['method']=='incremental_rk4':
   self.setup_csc();self.raw_sums(self.y[self.n:]);cp.copyto(self.previous,self.y[self.n:])
  cp.cuda.get_current_stream().synchronize()
 def run(self,spec,condition,duration,sample,out=None):
  self.reset(spec)
  count=round(sample/spec['h_s'])
  if abs(count*spec['h_s']-sample)>1e-12:raise ValueError('Sample must split steps exactly')
  stream=cp.cuda.Stream(non_blocking=True)
  with stream:
   stream.begin_capture()
   for _ in range(count):self.step(spec)
   graph=stream.end_capture()
  stream.synchronize();per_sample=self.evaluations;self.evaluations=0
  snapshots=[self.initial.copy()];times=[0.];walls=[];active=False
  started=time.perf_counter()
  for block in range(round(duration/sample)):
   t=block*sample;new_active=condition=='photo_pulse' and .001-1e-12<=t<.003-1e-12
   if new_active!=active:
    # Discontinuities lie exactly on integration boundaries; no stale graph input.
    self.d['photo'][:]=self.d['visual']*(.2 if new_active else 0.)
    cp.cuda.get_current_stream().synchronize();active=new_active
   tick=time.perf_counter();graph.launch(stream=stream);stream.synchronize();walls.append(time.perf_counter()-tick)
   state=cp.asnumpy(self.y)
   if not np.isfinite(state).all():raise FloatingPointError('Nonfinite recurrent state')
   snapshots.append(state);times.append((block+1)*sample)
   if time.perf_counter()-started>300:raise TimeoutError('Per-run wall budget exhausted')
  wall=time.perf_counter()-started
  states=np.asarray(snapshots);counts=cp.asnumpy(self.counts).tolist()
  result={'spec':spec,'condition':condition,'scope':'base recurrent q/s only','duration_s':duration,'samples_s':times,'wall_with_sampling_s':wall,'stepping_wall_s':sum(walls),'block_wall_s':walls,'coefficient_evaluations':per_sample*(len(times)-1),'max_implicit_residual':float(cp.max(self.res).get()),'updated_columns':counts[0],'updated_edges':counts[1],'state_min':float(states.min()),'state_max':float(states.max()),'finite':True}
  if out:
   out=Path(out);out.mkdir(exist_ok=False);np.savez_compressed(out/'states.npz',time_s=times,state=states);write(out/'timing.json',result)
  return result,states

def main():
 fixture=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
 contract=json.loads((HERE/'contract.json').read_text())
 if digest(fixture)!=contract['fixture_sha256']:raise ValueError('Graph fixture hash mismatch')
 p=json.loads((HERE/'initial_provenance.json').read_text())
 if digest(HERE/'initial.npz')!=p['initial_sha256']:raise ValueError('Initial state hash mismatch')
 z=np.load(fixture,allow_pickle=False);data={k:z[k] for k in z.files};z=np.load(HERE/'initial.npz',allow_pickle=False);initial=np.concatenate([z['q'],z['s']])
 if len(data['caps'])!=contract['neurons'] or len(data['weights'])!=contract['stored_edges']:raise ValueError('Wrong scope')
 write(out/'freeze.json',{'contract_sha256':digest(HERE/'contract.json'),'files':{str(p.relative_to(HERE.parent)):digest(p) for p in [HERE/'recurrence.py',HERE/'recurrence.cu',HERE/'contract.json',HERE/'initial.npz',HERE/'initial_provenance.json',HERE.parent/'connections.cu']},'python':sys.version,'numpy':np.__version__,'cupy':cp.__version__,'platform':platform.platform(),'cuda_device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),'scope':contract['scope']})
 engine=Engine(data,initial);started=time.perf_counter();rows=[]
 specs=[{'id':'R'+str(i),'method':'rk4','h_s':h} for i,h in enumerate(contract['reference_steps_s'])]+contract['candidates']
 for condition in contract['conditions']:
  for spec in specs:
   name=condition+'_'+spec['id'];tick=time.perf_counter()
   try:
    result,_=engine.run(spec,condition,contract['duration_s'],contract['sample_s'],out/name);rows.append({'run':name,**result});print(json.dumps({'run':name,'wall_s':result['wall_with_sampling_s'],'residual':result['max_implicit_residual'],'updated_edges':result['updated_edges']}),flush=True)
   except Exception as err:
    write(out/(name+'_failure.json'),{'type':type(err).__name__,'message':str(err),'elapsed_s':time.perf_counter()-tick});print(name,type(err).__name__,flush=True)
   if time.perf_counter()-started>contract['budget']['gpu_wall_seconds']:raise TimeoutError('Campaign wall budget exhausted')
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>contract['budget']['host_memory_GiB']*2**30:raise MemoryError('Host budget exceeded')
   if cp.get_default_memory_pool().total_bytes()>contract['budget']['gpu_memory_GiB']*2**30:raise MemoryError('GPU pool budget exceeded')
 write(out/'runs.json',rows)
if __name__=='__main__':main()

"""Declared 17-coordinate adapter to the generic independent-block runtime.

The epoch inputs are held by the existing coupling method. There are no reads
of another block's evolving state. Original geometry and FP64 residual retained.
"""
from pathlib import Path
import sys,time,types
import numpy as np
import cupy as cp
HERE=Path(__file__).resolve().parent
sys.path.append('/home/daroch/AXIOMA_ASTRA/motor_nuevo/native_hybrid_20260922')
from native_cell import Cell,WARP

def source():
    code=WARP.replace('extern "C" __global__ void warp_midpoint','__device__ __noinline__ void warp_midpoint')
    # Same arithmetic; compile bounded row loops without dynamic local arrays.
    for needle in ('for(int j=0;j<17;j++)','for(int j=16;j>=0;j--)','for(int k=j+1;k<17;k++)'):
        code=code.replace(needle,'\n#pragma unroll\n'+needle)
    physical=(HERE/'physical_events.cu').read_text()
    physical=physical.replace('extern "C" __global__ void','__device__ void')
    physical=physical.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;', 'int i=blockIdx.x;')
    physical=physical.replace('int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=n*axons)return;', 'int j=blockIdx.x*axons+threadIdx.x;')
    return code+'\n'+physical+'\n'+(HERE/'block_runtime.hpp').read_text()+'\n'+(HERE/'device_cell.cu').read_text()

class DeviceCell(Cell):
 def __init__(self,b,events,compressed=False,*,validate_publisher=True):
  if compressed:raise ValueError('No uncertified geometry compression in this prototype')
  if b.ports!=17 or b.backend!='cuda':raise ValueError('Wrong model for declared adapter')
  self.b=b;self.events=events;self.handle=None;self.stream=cp.cuda.Stream(non_blocking=True);self.pool=cp.cuda.MemoryPool();self.width=8
  self.report={'calls':0,'accepted':0,'rejected':0,'build_s':0.,'native_wall_s':0.,'event_capacity_per_cell_per_epoch':8,'compressed':False,'scheduler':'device_independent_block','count_unit':'cell_trials'}
  p=b._motor_axonal_callback
  if validate_publisher:
   from axon_gpu import Publisher
   if type(p) is not Publisher:raise ValueError('Publisher callback independence not admitted')
  self.axons=p.fields['q'].shape[1];self.ts=p.wrapper.publisher.synaptic_tau
  if self.axons!=12:raise ValueError('Unknown axonal map')
  start=time.perf_counter();cp.cuda.get_current_stream().synchronize()
  with cp.cuda.using_allocator(self.pool.malloc),self.stream:
   self.state={k:cp.array(getattr(b,k),copy=True) for k in ('delta','gates','q','counts','last_siz','previous_slope','trough','clipped')}
   self.ax={k:cp.array(v,copy=True) for k,v in p.fields.items()};self.gain=p.gain.copy()
   self.ge=cp.zeros((b.n,4));self.gi=cp.zeros_like(self.ge);self.current=cp.zeros_like(b.delta)
   self.et=cp.zeros((b.n,self.width));self.ej=cp.zeros_like(self.et);self.ep=cp.zeros_like(self.et);self.ec=cp.zeros(b.n,dtype=cp.int32);self.flag=cp.zeros(b.n,dtype=cp.int32)
   self.observation=cp.ascontiguousarray(b.obs[1]);self.coords=cp.arange(5,17,dtype=cp.int32)
   self.vscratch=[cp.empty_like(b.delta) for _ in range(3)];self.gscratch=[cp.empty_like(b.gates) for _ in range(3)]
   self.errors=cp.zeros((b.n,6));self.counts=cp.zeros((b.n,3),dtype=cp.int64);self.maximum_error=cp.zeros(b.n);self.status=cp.zeros(b.n,dtype=cp.int32)
   self.clock=cp.zeros((b.n,2),dtype=cp.int64)
   self.kernel=cp.RawKernel(source(),'cell_epoch',options=('--fmad=false',))
   self.kernel.compile()
  self.report.update(build_s=time.perf_counter()-start,device_bytes=self.pool.total_bytes(),kernel=self.kernel.attributes)

 def advance(self,b,ns,ge,gi,*,current_pA=None,inner_step_ns=25000):
  if getattr(self,'publication_failed',False):raise RuntimeError('Native publication owner invalidated; rebuild session')
  if type(ns)is not int or ns<=0 or type(inner_step_ns)is not int or not 0<inner_step_ns<=25000:raise ValueError('Invalid physical epoch')
  for x in (ge,gi):
   if x.shape!=(b.n,4) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid held conductance')
  p=b._motor_axonal_callback
  if p.elapsed!=0 or self.events.active is None:raise ValueError('Physical publisher ownership mismatch')
  origin=b.elapsed_ns;cp.cuda.get_current_stream().synchronize();start=time.perf_counter()
  with self.stream:
   self.load_epoch(p);self.status.fill(0);self.ge.set(np.ascontiguousarray(ge));self.gi.set(np.ascontiguousarray(gi))
   if current_pA is None:self.current.fill(0)
   else:
    x=cp.asarray(current_pA)
    if x.shape!=self.current.shape or not bool(cp.isfinite(x).all()):raise ValueError('Invalid current')
    cp.copyto(self.current,x)
   s=self.state;a=self.ax
   args=(np.int32(b.n),np.int64(ns),np.int64(inner_step_ns),np.float64(b.rest),s['delta'],s['gates'],self.ge,self.gi,self.current,b.C,b.G,b.chanG,b.chanb,b.shuntG,b.shuntb,b.ena,*self.vscratch,*self.gscratch,self.errors,self.clock,self.observation,b.caps,b.tau,s['q'],s['last_siz'],s['previous_slope'],s['trough'],s['counts'],s['clipped'],self.ec,self.et,self.ej,self.ep,self.flag,self.coords,np.float64(self.ts),self.gain,a['q'],a['s'],a['last_voltage'],a['previous_slope'],a['trough'],a['counts'],a['clipped'],self.counts,self.maximum_error,self.status)
   self.kernel((b.n,),(32,),args)
   status=self.status.get();counts=self.counts.get();maximum_error=self.maximum_error.get()
  self.stream.synchronize();self.report['native_wall_s']+=time.perf_counter()-start
  if np.any(status):raise RuntimeError('Device epoch rejected: '+str(np.unique(status,return_counts=True)))
  if not np.isfinite(maximum_error).all():raise RuntimeError('Invalid device estimator')
  # Publication failures invalidate this owner; they are not retryable epochs.
  try:
   # Whole epoch commit. A failed block never exposes other completed blocks.
   with self.stream:
    for k,v in s.items():setattr(b,k,v.copy())
    for k,v in a.items():cp.copyto(p.fields[k],v)
    count=self.ec.get();et=self.et.get();ej=self.ej.get();ep=self.ep.get()
   self.stream.synchronize();rr,cc=np.where(np.arange(self.width)[None,:]<count[:,None])
   if len(rr):
    order=np.argsort(et[rr,cc],kind='stable');rr,cc=rr[order],cc[order]
    self.events.active.add(et[rr,cc]+(origin-self.events.start_elapsed)*1e-9,self.events.gamma[rr],ej[rr,cc],posts=ep[rr,cc])
   b.elapsed_ns+=ns;p.elapsed=ns
  except BaseException:
   self.publication_failed=True;b._native_publication_invalid=True
   raise
  stats=getattr(b,'_motor_adaptive_stats',None)
  if stats is None:stats=b._motor_adaptive_stats={'accepted':0,'rejected':0,'min_step_ns':25000,'max_estimated_error':0.}
  accepted=int(counts[:,0].sum());rejected=int(counts[:,1].sum());minimum=int(counts[:,2].min());error=float(maximum_error.max())
  stats['accepted']+=accepted;stats['rejected']+=rejected;stats['min_step_ns']=min(stats['min_step_ns'],minimum);stats['max_estimated_error']=max(stats['max_estimated_error'],error)
  self.report['calls']+=1;self.report['accepted']+=accepted;self.report['rejected']+=rejected;self.report['last_accepted_quantiles']=np.quantile(counts[:,0],[0,.5,.9,.99,1]).tolist()
  return b.host(b.q)

def install(brain,events):
 b=brain._spatial_batch
 while hasattr(b,'base'):b=b.base
 old=b.advance;holder=types.SimpleNamespace(core=None,report={'initialized':False})
 def advance(self,*args,**kw):
  if holder.core is None:holder.core=DeviceCell(self,events);holder.report=holder.core.report
  return holder.core.advance(self,*args,**kw)
 b.advance=types.MethodType(advance,b)
 def restore():b.advance=old
 return holder,restore

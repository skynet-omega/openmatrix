"""Native trial/commit transactions for the conserved spatial-cell adapter.

The generic C++ controller sees opaque trial and physical-commit graphs.
This adapter declares the existing 17-coordinate membrane model and its SIZ/
axonal observations. It does not replace it with a rate neuron or fit a spike.
"""
from pathlib import Path
import ctypes as ct,time,types
import numpy as np
import cupy as cp
from kc_fused_warp import WARP
from kc_adaptive import V_ATOL,G_ATOL
HERE=Path(__file__).resolve().parent

def build_step(groups=None):
 code=WARP.replace('int count,double dt,double rest','int count,const long long*clock,int split,double rest')
 code=code.replace('int n=blockIdx.x,i=threadIdx.x;', 'double dt=(split==0?clock[1]:(split==1?clock[1]/2:clock[1]-clock[1]/2))*1e-9; int n=blockIdx.x,i=threadIdx.x;')
 if groups is not None:
  from basis_compile import compile_warp
  code=compile_warp(code,groups)
 return cp.RawKernel(code,'warp_midpoint',options=('--fmad=false',))

class Cell:
 def __init__(self,b,events,compressed=False):
  if b.ports!=17 or b.backend!='cuda':raise ValueError('Adapter requires the conserved FP64 17-coordinate cell')
  self.b=b;self.events=events;self.handle=None;self.stream=cp.cuda.Stream(non_blocking=True);self.pool=cp.cuda.MemoryPool();self.width=8
  self.report={'calls':0,'accepted':0,'rejected':0,'build_s':0.,'native_wall_s':0.,'event_capacity_per_cell_per_epoch':8,'compressed':False}
  self.lib=ct.CDLL(str(HERE/'libcell_control.so'));l=self.lib;P=ct.c_void_p
  l.cell_create.argtypes=[P]*6;l.cell_create.restype=P;l.cell_error.restype=ct.c_char_p;l.cell_destroy.argtypes=[P]
  l.cell_advance.argtypes=[P,ct.c_longlong,ct.c_longlong,ct.POINTER(ct.c_longlong),ct.POINTER(ct.c_double)];l.cell_advance.restype=ct.c_int
  p=b._motor_axonal_callback;self.axons=p.fields['q'].shape[1];self.ts=p.wrapper.publisher.synaptic_tau
  if self.axons!=12:raise ValueError('Unknown axonal observation map')
  start=time.perf_counter();cp.cuda.get_current_stream().synchronize()
  with cp.cuda.using_allocator(self.pool.malloc),self.stream:
   self.clock=cp.asarray([0,25000],dtype=cp.int64);self.flag=cp.zeros(1,dtype=cp.int32)
   self.state={k:cp.array(getattr(b,k),copy=True) for k in ('delta','gates','q','counts','last_siz','previous_slope','trough','clipped')}
   self.ax={k:cp.array(v,copy=True) for k,v in p.fields.items()};self.gain=p.gain.copy()
   self.ge=cp.zeros((b.n,4));self.gi=cp.zeros_like(self.ge);self.current=cp.zeros_like(b.delta)
   self.et=cp.zeros((b.n,self.width));self.ej=cp.zeros_like(self.et);self.ec=cp.zeros(b.n,dtype=cp.int32)
   self.observation=cp.ascontiguousarray(b.obs[1]);self.coords=cp.arange(5,17,dtype=cp.int32)
   groups=None;self.chanG=b.chanG;self.chanb=b.chanb
   if compressed:
    from basis_compile import compile_basis
    cg,cb,groups,basis_report=compile_basis(cp.asnumpy(b.chanG).reshape(51,17,17),cp.asnumpy(b.chanb).reshape(51,17))
    self.chanG=cp.asarray(cg);self.chanb=cp.asarray(cb);self.report.update(compressed=True,basis=basis_report)
   self.step_kernel=build_step(groups);mod=cp.RawModule(code=(HERE/'physical_events.cu').read_text(),options=('--fmad=false',));self.somatic=mod.get_function('commit_event');self.axonal=mod.get_function('commit_axon')
   self.trial();self.stream.synchronize();self.commit();self.stream.synchronize();self.load_epoch(p)
   self.trial();self.stream.synchronize()
   self.stream.begin_capture()
   try:self.trial()
   finally:self.trial_graph=self.stream.end_capture()
   self.stream.begin_capture()
   try:self.commit()
   finally:self.commit_graph=self.stream.end_capture()
  self.stream.synchronize()
  self.handle=l.cell_create(self.trial_graph.graphExec,self.commit_graph.graphExec,self.stream.ptr,self.clock.data.ptr,self.error.data.ptr,self.flag.data.ptr)
  if self.handle is None:raise RuntimeError(l.cell_error().decode())
  self.report.update(build_s=time.perf_counter()-start,device_bytes=self.pool.total_bytes())
 def step(self,v,g,split):
  b=self.b;out=cp.empty_like(v);gates=cp.empty_like(g);errors=cp.zeros((b.n,2))
  self.step_kernel((b.n,),(32,),(np.int32(b.n),self.clock,np.int32(split),np.float64(b.rest),v,g,self.ge,self.gi,self.current,b.C,b.G,self.chanG,self.chanb,b.shuntG,b.shuntb,b.ena,out,gates,errors))
  return out,gates,errors
 def trial(self):
  v,g=self.state['delta'],self.state['gates']
  vf,gf,ef=self.step(v,g,0);self.va,self.ga,ea=self.step(v,g,1);self.vb,self.gb,eb=self.step(self.va,self.ga,2)
  h=self.clock[1].astype(cp.float64);h1=(self.clock[1]//2).astype(cp.float64);h2=h-h1
  cubes=(h1/h)**3+(h2/h)**3;factor=cubes/(1-cubes)
  valid=cp.isfinite(ef).all()&cp.isfinite(ea).all()&cp.isfinite(eb).all()
  norm=cp.maximum(cp.max(cp.abs(self.vb-vf))*factor/V_ATOL,cp.max(cp.abs(self.gb-gf))*factor/G_ATOL)
  self.error=cp.where(valid,norm,cp.inf)
 def commit(self):
  b=self.b;s=self.state;a=self.ax;self.flag.fill(0)
  for split,v in ((1,self.va),(2,self.vb)):
   self.somatic(((b.n+255)//256,),(256,),(np.int32(b.n),np.int32(b.ports),np.int32(self.width),np.int32(split),self.clock,v,np.float64(b.rest),self.observation,np.float64(-40.),np.float64(20.),b.caps,b.tau,s['q'],s['last_siz'],s['previous_slope'],s['trough'],s['counts'],s['clipped'],self.ec,self.et,self.ej,self.flag))
   self.axonal(((b.n*self.axons+255)//256,),(256,),(np.int32(b.n),np.int32(b.ports),np.int32(self.axons),np.int32(split),self.clock,self.coords,v,np.float64(b.rest),np.float64(self.ts),b.caps,b.tau,self.gain,np.float64(-40.),np.float64(20.),a['q'],a['s'],a['last_voltage'],a['previous_slope'],a['trough'],a['counts'],a['clipped'],self.flag))
  cp.copyto(s['delta'],self.vb);cp.copyto(s['gates'],self.gb)
 def load_epoch(self,p):
  for k,v in self.state.items():cp.copyto(v,getattr(self.b,k))
  for k,v in self.ax.items():cp.copyto(v,p.fields[k])
  cp.copyto(self.gain,p.gain);self.ec.fill(0);self.et.fill(0);self.ej.fill(0);self.flag.fill(0)
 def advance(self,b,ns,ge,gi,*,current_pA=None,inner_step_ns=25000):
  if type(ns) is not int or ns<=0 or type(inner_step_ns) is not int or not 0<inner_step_ns<=25000:raise ValueError('Invalid native physical clock')
  for x in (ge,gi):
   if x.shape!=(b.n,4) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('Invalid receptor conductances')
  p=b._motor_axonal_callback
  if p.elapsed!=0 or self.events.active is None:raise ValueError('Physical publisher ownership mismatch')
  origin=b.elapsed_ns;cp.cuda.get_current_stream().synchronize()
  with self.stream:
   self.load_epoch(p);self.ge.set(np.ascontiguousarray(ge));self.gi.set(np.ascontiguousarray(gi))
   if current_pA is None:self.current.fill(0)
   else:
    current=cp.asarray(current_pA)
    if current.shape!=self.current.shape or not bool(cp.isfinite(current).all()):raise ValueError('Invalid current')
    cp.copyto(self.current,current)
  counts=(ct.c_longlong*3)();error=ct.c_double();start=time.perf_counter()
  status=self.lib.cell_advance(self.handle,ns,inner_step_ns,counts,ct.byref(error))
  self.report['native_wall_s']+=time.perf_counter()-start
  if status:raise RuntimeError(self.lib.cell_error().decode())
  with self.stream:
   for k,v in self.state.items():setattr(b,k,v.copy())
   for k,v in self.ax.items():cp.copyto(p.fields[k],v)
   count=self.ec.get();et=self.et.get();ej=self.ej.get()
  self.stream.synchronize()
  rr,cc=np.where(np.arange(self.width)[None,:]<count[:,None])
  if len(rr):self.events.active.add(et[rr,cc]+(origin-self.events.start_elapsed)*1e-9,self.events.gamma[rr],ej[rr,cc])
  b.elapsed_ns+=ns;p.elapsed=ns
  stats=getattr(b,'_motor_adaptive_stats',None)
  if stats is None:stats=b._motor_adaptive_stats={'accepted':0,'rejected':0,'min_step_ns':25000,'max_estimated_error':0.}
  stats['accepted']+=counts[0];stats['rejected']+=counts[1];stats['min_step_ns']=min(stats['min_step_ns'],counts[2]);stats['max_estimated_error']=max(stats['max_estimated_error'],error.value)
  self.report['calls']+=1;self.report['accepted']+=counts[0];self.report['rejected']+=counts[1]
  return b.host(b.q)
 def close(self):
  if self.handle is not None:self.lib.cell_destroy(self.handle);self.handle=None

def install(brain,events,compressed=False):
 b=brain._spatial_batch
 while hasattr(b,'base'):b=b.base
 old=b.advance;holder=types.SimpleNamespace(core=None,report={'initialized':False})
 def advance(self,*args,**kw):
  if holder.core is None:holder.core=Cell(self,events,compressed);holder.report=holder.core.report
  return holder.core.advance(self,*args,**kw)
 b.advance=types.MethodType(advance,b)
 def restore():
  b.advance=old
  if holder.core:holder.core.close()
 return holder,restore

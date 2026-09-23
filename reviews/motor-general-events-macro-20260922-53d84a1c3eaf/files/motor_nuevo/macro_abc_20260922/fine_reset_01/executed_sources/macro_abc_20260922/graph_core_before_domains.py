"""Generic native adaptive target/rate execution with optional physical waveform ports.
The operator and boundary are supplied by a model adapter; this module has no anatomical names.
"""
from pathlib import Path
import ctypes as ct,time
import numpy as np,cupy as cp
KERNEL=r'''
extern "C" __global__ void check_values(const double* y,const double* a,const double* b,int n,int* flag){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&(!isfinite(y[i])||!isfinite(a[i])||!isfinite(b[i])))atomicOr(flag,1);}
extern "C" __global__ void norm(const double*a,const double*b,int n,int normn,double atol,double rtol,double*err,int*flag,int*domain){__shared__ double z[256];int i=blockIdx.x*blockDim.x+threadIdx.x;double e=0;if(i<n){if(!isfinite(a[i])||!isfinite(b[i]))atomicOr(flag,1);if(b[i]<0||b[i]>1)atomicOr(domain,1);if(i<normn)e=fabs(a[i]-b[i])/(3*(atol+rtol*fmax(fabs(a[i]),fabs(b[i]))));if(!isfinite(e)){atomicOr(flag,1);e=0;}}z[threadIdx.x]=e;__syncthreads();for(int k=128;k;k/=2){if(threadIdx.x<k)z[threadIdx.x]=fmax(z[threadIdx.x],z[threadIdx.x+k]);__syncthreads();}if(threadIdx.x==0)atomicMax((unsigned long long*)err,__double_as_longlong(z[0]));}
extern "C" __global__ void status(const double*e,const int*f,const int*d,double*s){s[0]=e[0];s[1]=f[0];s[2]=d[0];}
'''
class NativeGraph:
 def __init__(self,initial,coefficient,*,rtol,atol,norm_size,project=None,freeze=None,native_library=None):
  start=time.perf_counter();self.stream=cp.cuda.Stream(non_blocking=True);self.pool=cp.cuda.MemoryPool();self.n=len(initial);self.project=project;self.handle=None
  self.lib=ct.CDLL(str(native_library or Path(__file__).resolve().parent/'libgraph_control.so'));l=self.lib
  l.engine_create.argtypes=[ct.c_void_p]*6+[ct.c_long];l.engine_create.restype=ct.c_void_p;l.engine_error.restype=ct.c_char_p;l.engine_destroy.argtypes=[ct.c_void_p]
  l.engine_advance.argtypes=[ct.c_void_p,ct.c_long,ct.POINTER(ct.c_long),ct.c_long,ct.c_long,ct.c_double,ct.POINTER(ct.c_long),ct.POINTER(ct.c_double)];l.engine_advance.restype=ct.c_int
  self.coefficient=coefficient;self.norm_size=norm_size;self.rtol=rtol;self.atol=atol
  cp.cuda.get_current_stream().synchronize()
  with cp.cuda.using_allocator(self.pool.malloc),self.stream:
   self.x=cp.asarray(initial);self.backup=cp.empty_like(self.x);self.clock=cp.zeros(2);self.err=cp.zeros(1);self.flag=cp.zeros(1,dtype=cp.int32);self.domain=cp.zeros(1,dtype=cp.int32);self.status=cp.zeros(3)
   mod=cp.RawModule(code=KERNEL);self.check=mod.get_function('check_values');self.norm=mod.get_function('norm');self.summary=mod.get_function('status');self.grid=((self.n+255)//256,)
   if freeze:freeze(self)
   self.clock[:]=cp.asarray([0.,1e-6]);self.trial();self.stream.synchronize();self.trial();self.stream.synchronize()
   self.stream.begin_capture()
   try:self.trial()
   finally:self.graph=self.stream.end_capture()
  self.stream.synchronize();self.handle=l.engine_create(self.graph.graphExec,self.stream.ptr,self.clock.data.ptr,self.status.data.ptr,self.x.data.ptr,self.fine.data.ptr,self.n)
  if self.handle is None:raise RuntimeError(l.engine_error().decode())
  self.build_s=time.perf_counter()-start;self.calls=0;self.steps=0;self.rejected=0
 def coeff(self,y,frac):
  z=self.project(y,self.clock,frac) if self.project else y
  a,b=self.coefficient(z);self.check(self.grid,(256,),(z,a,b,np.int32(self.n),self.flag))
  return z,a,b
 def midpoint(self,y,frac,start):
  z,a,b=self.coeff(y,start);middle=z+(-cp.expm1(-.5*frac*self.clock[1]*b))*(a-z)
  _,a,b=self.coeff(middle,start+.5*frac);out=z+(-cp.expm1(-frac*self.clock[1]*b))*(a-z)
  return self.project(out,self.clock,start+frac) if self.project else out
 def trial(self):
  self.err.fill(0);self.flag.fill(0);self.domain.fill(0)
  self.full=self.midpoint(self.x,1.,0.);self.fine=self.midpoint(self.midpoint(self.x,.5,0.),.5,.5)
  self.norm(self.grid,(256,),(self.full,self.fine,np.int32(self.n),np.int32(self.norm_size),np.float64(self.atol),np.float64(self.rtol),self.err,self.flag,self.domain));self.summary((1,),(1,),(self.err,self.flag,self.domain,self.status))
 def advance(self,ns,next_ns,min_ns,max_ns,budget=30,boundaries=None):
  with self.stream:cp.copyto(self.backup,self.x)
  nxt=ct.c_long(next_ns);counts=(ct.c_long*3)();mx=ct.c_double()
  if boundaries is None:
   code=self.lib.engine_advance(self.handle,ns,ct.byref(nxt),min_ns,max_ns,budget,counts,ct.byref(mx))
  else:
   times=np.ascontiguousarray(np.unique(boundaries),dtype=np.float64)
   if times.ndim!=1:raise ValueError('One-dimensional event schedule required')
   fn=self.lib.engine_advance_events
   fn.argtypes=[ct.c_void_p,ct.c_long,ct.POINTER(ct.c_long),ct.c_long,ct.c_long,ct.c_double,ct.POINTER(ct.c_double),ct.c_long,ct.POINTER(ct.c_long),ct.POINTER(ct.c_double)];fn.restype=ct.c_int
   code=fn(self.handle,ns,ct.byref(nxt),min_ns,max_ns,budget,times.ctypes.data_as(ct.POINTER(ct.c_double)),len(times),counts,ct.byref(mx))
  if code!=0:
   message=self.lib.engine_error().decode()
   # Preserve the attempted endpoint before rollback. Diagnostics run only on
   # failure; they never clamp state or change numerical acceptance.
   self.last_failure={'message':message,'scope':'attempted fine endpoint before rollback'}
   try:
    attempted=self.fine.get(stream=self.stream)
    bad=np.flatnonzero(~np.isfinite(attempted)|(attempted<0)|(attempted>1))
    self.last_failure.update(out_of_domain_count=int(len(bad)),sample_indices=bad[:16].tolist(),
     sample_values=[float(attempted[i]) if np.isfinite(attempted[i]) else str(attempted[i]) for i in bad[:16]],
     attempted_clock_s=self.clock.get(stream=self.stream).tolist())
   except BaseException as exc:
    self.last_failure['diagnostic_error']=repr(exc)
   with self.stream:cp.copyto(self.x,self.backup)
   self.stream.synchronize()
   exc=RuntimeError(message);exc.numerical_diagnostic=self.last_failure
   raise exc
  self.calls+=1;self.steps+=counts[0];self.rejected+=counts[1]
  return nxt.value,list(counts),mx.value
 def close(self):
  if self.handle is not None:self.lib.engine_destroy(self.handle);self.handle=None
 def __del__(self):
  if getattr(self,'handle',None) is not None:self.close()

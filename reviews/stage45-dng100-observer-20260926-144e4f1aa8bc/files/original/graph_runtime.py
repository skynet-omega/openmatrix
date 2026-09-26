"""Model-independent resident RK3(2) for a captured CUDA right-hand side.

Python allocates and captures once. C++/CUDA owns every trial, decision and
commit inside an epoch. A model may provide exact time-dependent projection;
it must not publish side effects from speculative RHS calls.
"""
from pathlib import Path
import ctypes as ct
import time
import numpy as np
import cupy as cp

KERNEL = r'''
extern "C" __global__ void endpoint_clocks(const double*c,double*left,double*right) {
 if(threadIdx.x==0 && blockIdx.x==0) {
  left[0]=nextafter(c[2],c[0]);left[1]=0.;
  right[0]=c[2];right[1]=0.;
 }
}
extern "C" __global__ void stage(int n,const double*y,const double*k1,
 const double*k2,const double*k3,const double*c,double a,double b,double d,double*out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=y[i]+c[1]*(a*k1[i]+b*k2[i]+d*k3[i]);
}
extern "C" __global__ void estimate(int n,int normn,const double*y,const double*next,
 const double*k1,const double*k2,const double*k3,const double*k4,const double*c,
 const double*atol,const double*rtol,const double*lower,const double*upper,double*err) {
 __shared__ double block[256]; int i=blockIdx.x*blockDim.x+threadIdx.x; double e=0.;
 if(i<n) {
  double v=next[i],scale=atol[i]+rtol[i]*fmax(fabs(y[i]),fabs(v));
  if(i<normn)e=fabs(c[1]*((-5./72.)*k1[i]+(1./12.)*k2[i]+(1./9.)*k3[i]-(1./8.)*k4[i]))/scale;
  // A finite rejection sentinel lets the common controller reduce the step.
  // No invalid candidate is committed, including outside the norm prefix.
  if(!isfinite(scale)||scale<=0.||!isfinite(e)||!isfinite(v)||v<lower[i]||v>upper[i]||
     !isfinite(k1[i])||!isfinite(k2[i])||!isfinite(k3[i])||!isfinite(k4[i]))e=1e300;
 }
 block[threadIdx.x]=e;__syncthreads();
 for(int k=128;k;k/=2){if(threadIdx.x<k)block[threadIdx.x]=fmax(block[threadIdx.x],block[threadIdx.x+k]);__syncthreads();}
 if(threadIdx.x==0)atomicMax((unsigned long long*)err,__double_as_longlong(block[0]));
}
'''


def need(ok, message):
    if not ok:
        raise ValueError(message)


class GraphRK23:
    def __init__(self, initial, rhs, *, rtol, atol, norm_size=None,
                 project=None, freeze=None, state_bounds=(-np.inf, np.inf),
                 event_capacity=4096, library=None):
        started=time.perf_counter()
        initial=np.asarray(initial,dtype=np.float64)
        need(initial.ndim==1 and len(initial)>0 and np.isfinite(initial).all(),'invalid initial state')
        self.n=len(initial);self.norm_size=self.n if norm_size is None else norm_size
        need(0<self.norm_size<=self.n,'invalid norm size')
        self.rhs=rhs;self.project=project;self.handle=None
        self.stream=cp.cuda.Stream(non_blocking=True);self.pool=cp.cuda.MemoryPool()
        self.grid=((self.n+255)//256,)
        self.lower_host=np.broadcast_to(np.asarray(state_bounds[0],dtype=np.float64),initial.shape).copy()
        self.upper_host=np.broadcast_to(np.asarray(state_bounds[1],dtype=np.float64),initial.shape).copy()
        aa=np.broadcast_to(np.asarray(atol,dtype=np.float64),initial.shape).copy()
        rr=np.broadcast_to(np.asarray(rtol,dtype=np.float64),initial.shape).copy()
        need(not np.isnan(self.lower_host).any() and not np.isnan(self.upper_host).any()
             and np.all(self.lower_host<=initial) and np.all(initial<=self.upper_host),'invalid state domain')
        need(np.isfinite(aa).all() and np.isfinite(rr).all() and np.all(aa>0) and np.all(rr>=0),'invalid tolerances')
        dll=ct.CDLL(str(library or Path(__file__).with_name('libresident_controller.so')))
        ptr=ct.c_void_p;lng=ct.c_long
        # Versioned ABI: clock has start, step, and the canonical endpoint.
        # An older library must fail during binding, before any device access.
        dll.resident_create_endpoint_v1.argtypes=[ptr]*6+[lng,lng];dll.resident_create_endpoint_v1.restype=ptr
        dll.resident_advance.argtypes=[ptr,lng,ct.POINTER(lng),lng,lng,ct.POINTER(ct.c_double),lng,lng,ct.POINTER(lng),ct.POINTER(ct.c_double)]
        dll.resident_advance.restype=ct.c_int;dll.resident_destroy.argtypes=[ptr]
        dll.resident_error.restype=ct.c_char_p;self.lib=dll
        cp.cuda.get_current_stream().synchronize()
        with cp.cuda.using_allocator(self.pool.malloc),self.stream:
            self.x=cp.asarray(initial);self.backup=cp.empty_like(self.x)
            self.clock=cp.asarray([0.,1e-6,1e-6]);self.left=cp.zeros(2);self.right=cp.zeros(2)
            self.status=cp.zeros(3);self.atol=cp.asarray(aa);self.rtol=cp.asarray(rr)
            self.lower=cp.asarray(self.lower_host);self.upper=cp.asarray(self.upper_host)
            self.stage_buffer=[cp.empty_like(self.x) for _ in range(3)]
            self.module=cp.RawModule(code=KERNEL,options=('--std=c++17','--fmad=false'))
            self.stage_kernel=self.module.get_function('stage')
            self.estimate=self.module.get_function('estimate')
            self.endpoint_kernel=self.module.get_function('endpoint_clocks')
            if freeze:freeze(self)
            self.trial();self.stream.synchronize()
            self.trial();self.stream.synchronize()
            raw=0
            cp.cuda.runtime.streamBeginCapture(self.stream.ptr)
            try:self.trial()
            finally:raw=cp.cuda.runtime.streamEndCapture(self.stream.ptr)
            try:
                self.handle=dll.resident_create_endpoint_v1(raw,self.stream.ptr,self.clock.data.ptr,
                    self.status.data.ptr,self.x.data.ptr,self.fine.data.ptr,self.n,event_capacity)
            finally:cp.cuda.runtime.graphDestroy(raw)
        self.stream.synchronize()
        need(self.handle is not None,'resident setup: '+dll.resident_error().decode())
        self.build_s=time.perf_counter()-started
        self.calls=0;self.steps=0;self.rejected=0;self.resident_wall_s=0.;self.last_counts=None

    def projected(self,y,clock,fraction):
        return self.project(y,clock,fraction) if self.project else y

    def field(self,y,clock,fraction):
        z=self.projected(y,clock,fraction)
        value=self.rhs(z,clock,fraction)
        need(value.shape==self.x.shape and value.dtype==cp.float64,'RHS contract')
        return z,value

    def stage(self,y,k1,k2,k3,a,b,c,out):
        self.stage_kernel(self.grid,(256,),(np.int32(self.n),y,k1,k2,k3,self.clock,
                          np.float64(a),np.float64(b),np.float64(c),out))
        return out

    def trial(self):
        self.status.fill(0)
        z,k1=self.field(self.x,self.clock,0.)
        u=self.stage(z,k1,k1,k1,.5,0.,0.,self.stage_buffer[0])
        _,k2=self.field(u,self.clock,.5)
        u=self.stage(z,k2,k2,k2,.75,0.,0.,self.stage_buffer[1])
        _,k3=self.field(u,self.clock,.75)
        high=self.stage(z,k1,k2,k3,2./9.,1./3.,4./9.,self.stage_buffer[2])
        # A mandatory discontinuity belongs to the following interval. The
        # endpoint derivative uses its left limit; committed state is right-sided.
        # Use the scheduler's exact endpoint for both sides, without re-adding h.
        self.endpoint_kernel((1,),(1,),(self.clock,self.left,self.right))
        self.fine=self.projected(high,self.right,0.)
        _,k4=self.field(high,self.left,0.)
        self.estimate(self.grid,(256,),(np.int32(self.n),np.int32(self.norm_size),z,self.fine,
            k1,k2,k3,k4,self.clock,self.atol,self.rtol,self.lower,self.upper,self.status))

    def advance(self,ns,next_ns,min_ns,max_ns,budget=30,boundaries=None):
        need(type(ns) is int and ns>0 and type(next_ns) is int and next_ns>0,'integer epoch/step required')
        times=np.ascontiguousarray(np.unique(boundaries) if boundaries is not None else [],dtype=np.float64)
        need(times.ndim==1 and np.isfinite(times).all(),'invalid boundary schedule')
        with self.stream:cp.copyto(self.backup,self.x)
        nxt=ct.c_long(next_ns);counts=(ct.c_long*3)();error=ct.c_double()
        started=time.perf_counter()
        code=self.lib.resident_advance(self.handle,ns,ct.byref(nxt),min_ns,max_ns,
            times.ctypes.data_as(ct.POINTER(ct.c_double)),len(times),10000,counts,ct.byref(error))
        self.resident_wall_s+=time.perf_counter()-started
        if code:
            message=self.lib.resident_error().decode()
            self.last_failure={'message':message,'clock':self.clock.get(stream=self.stream).tolist(),
                               'status':self.status.get(stream=self.stream).tolist()}
            with self.stream:cp.copyto(self.x,self.backup)
            self.stream.synchronize()
            exc=RuntimeError(message);exc.numerical_diagnostic=self.last_failure;raise exc
        self.calls+=1;self.steps+=counts[0];self.rejected+=counts[1];self.last_counts=list(counts)
        return nxt.value,list(counts),error.value

    def close(self):
        if self.handle is not None:self.lib.resident_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()

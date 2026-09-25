"""Resident adaptive exponential midpoint with a shared initial evaluation.

For y' = rate(y,t,side) * (target(y,t,side) - y), a full midpoint step
is compared with two half steps. Their identical initial coefficient is
evaluated once and copied before the model may reuse its output buffers.
The resulting five evaluations preserve the six-evaluation reference method.

Model callbacks receive (state, device_time[1], side), LEFT=-1 / RIGHT=1.
They must be pure during a trial. Python captures the model once; the CUDA
scheduler owns adaptive trials and commits. No anatomy or behavior is here.
"""
from pathlib import Path
import ctypes as ct
import threading
import time
import numpy as np
import cupy as cp

LEFT, RIGHT = -1, 1
KERNEL = r'''
extern "C" __global__ void stage_times(const double*c,double*t,int*flags) {
 if(blockIdx.x || threadIdx.x)return;
 t[0]=c[0]; t[1]=c[0]+.25*c[1]; t[2]=c[0]+.5*c[1];
 t[3]=c[0]+.75*c[1]; t[4]=c[2];
 for(int i=0;i<4;i++)if(!(t[i]<t[i+1]))flags[0]=1;
}
extern "C" __global__ void values(int n,const double*y,const double*a,
 const double*r,int*flags) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n && (!isfinite(y[i])||!isfinite(a[i])||!isfinite(r[i])))atomicOr(flags,1);
}
extern "C" __global__ void flow(int n,const double*y,const double*a,
 const double*r,const double*c,double fraction,double*out) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)out[i]=y[i]+(-expm1(-fraction*c[1]*r[i]))*(a[i]-y[i]);
}
extern "C" __global__ void norm(int n,int normn,const double*a,const double*b,
 const double*atol,const double*rtol,const double*lower,const double*upper,
 double*status,int*flags) {
 __shared__ double block[256]; int i=blockIdx.x*blockDim.x+threadIdx.x;double e=0.;
 if(i<n) {
  if(!isfinite(a[i])||!isfinite(b[i]))atomicOr(flags,1);
  if(b[i]<lower[i]||b[i]>upper[i])atomicOr(flags+1,1);
  if(i<normn)e=fabs(a[i]-b[i])/(3.*(atol[i]+rtol[i]*fmax(fabs(a[i]),fabs(b[i]))));
  if(!isfinite(e)){atomicOr(flags,1);e=0.;}
 }
 block[threadIdx.x]=e;__syncthreads();
 for(int k=128;k;k/=2){if(threadIdx.x<k)block[threadIdx.x]=fmax(block[threadIdx.x],block[threadIdx.x+k]);__syncthreads();}
 if(threadIdx.x==0)atomicMax((unsigned long long*)status,__double_as_longlong(block[0]));
}
extern "C" __global__ void summarize(double*s,const int*f) {
 if(blockIdx.x==0 && threadIdx.x==0){s[1]=f[0];s[2]=f[1];}
}
'''


def need(ok, message):
    if not ok: raise ValueError(message)


class GraphMidpoint:
    evaluations_per_trial = 5

    def __init__(self, initial, coefficient, *, rtol, atol, norm_size=None,
                 project=None, freeze=None, state_bounds=(-np.inf, np.inf),
                 event_capacity=4096, library=None):
        started=time.perf_counter()
        initial=np.asarray(initial,dtype=np.float64)
        need(initial.ndim==1 and len(initial)>0 and np.isfinite(initial).all(),'invalid initial state')
        self.n=len(initial);self.norm_size=self.n if norm_size is None else norm_size
        need(0<self.norm_size<=self.n,'invalid norm size')
        self.coefficient=coefficient;self.project=project;self.handle=None
        self._gate=threading.Lock()
        self.stream=cp.cuda.Stream(non_blocking=True);self.pool=cp.cuda.MemoryPool()
        self.grid=((self.n+255)//256,)
        lower=np.broadcast_to(np.asarray(state_bounds[0],float),initial.shape).copy()
        upper=np.broadcast_to(np.asarray(state_bounds[1],float),initial.shape).copy()
        aa=np.broadcast_to(np.asarray(atol,float),initial.shape).copy()
        rr=np.broadcast_to(np.asarray(rtol,float),initial.shape).copy()
        need(not np.isnan(lower).any() and not np.isnan(upper).any()
             and np.all(lower<=initial) and np.all(initial<=upper),'invalid state domain')
        need(np.isfinite(aa).all() and np.isfinite(rr).all() and np.all(aa>0) and np.all(rr>=0),'invalid tolerances')
        dll=ct.CDLL(str(library or Path(__file__).with_name('libresident_controller.so')))
        dll.resident_abi_version.restype=ct.c_int
        need(dll.resident_abi_version()==2,'resident clock ABI 2 required')
        ptr=ct.c_void_p;lng=ct.c_long
        dll.resident_create_v2.argtypes=[ptr]*6+[lng,lng];dll.resident_create_v2.restype=ptr
        dll.resident_advance.argtypes=[ptr,lng,ct.POINTER(lng),lng,lng,ct.POINTER(ct.c_double),lng,lng,ct.POINTER(lng),ct.POINTER(ct.c_double)]
        dll.resident_advance.restype=ct.c_int;dll.resident_destroy.argtypes=[ptr]
        dll.resident_error.restype=ct.c_char_p;self.lib=dll
        cp.cuda.get_current_stream().synchronize()
        with cp.cuda.using_allocator(self.pool.malloc),self.stream:
            self.x=cp.asarray(initial);self.backup=cp.empty_like(self.x)
            self.clock=cp.asarray([0.,1e-6,1e-6]);self.times=cp.empty(5)
            self.time_views=[self.times[i:i+1] for i in range(5)]
            self.status=cp.zeros(3);self.flags=cp.zeros(2,dtype=cp.int32)
            self.atol=cp.asarray(aa);self.rtol=cp.asarray(rr)
            self.lower=cp.asarray(lower);self.upper=cp.asarray(upper)
            self.start_target=cp.empty_like(self.x);self.start_rate=cp.empty_like(self.x)
            self.buffers=[cp.empty_like(self.x) for _ in range(6)]
            module=cp.RawModule(code=KERNEL,options=('--std=c++17','--fmad=false'))
            self.make_times=module.get_function('stage_times')
            self.check_values=module.get_function('values');self.flow_kernel=module.get_function('flow')
            self.norm_kernel=module.get_function('norm');self.summarize=module.get_function('summarize')
            if freeze:freeze(self)
            self.trial();self.stream.synchronize()
            self.trial();self.stream.synchronize()
            raw=0
            cp.cuda.runtime.streamBeginCapture(self.stream.ptr)
            try:self.trial()
            finally:raw=cp.cuda.runtime.streamEndCapture(self.stream.ptr)
            try:
                self.handle=dll.resident_create_v2(raw,self.stream.ptr,self.clock.data.ptr,
                    self.status.data.ptr,self.x.data.ptr,self.fine.data.ptr,self.n,event_capacity)
            finally:cp.cuda.runtime.graphDestroy(raw)
        self.stream.synchronize()
        need(self.handle is not None,'resident setup: '+dll.resident_error().decode())
        self.build_s=time.perf_counter()-started
        self.calls=0;self.steps=0;self.rejected=0;self.resident_wall_s=0.;self.last_counts=None

    def projected(self,y,slot,side=RIGHT):
        z=self.project(y,self.time_views[slot],side) if self.project else y
        need(isinstance(z,cp.ndarray) and z.shape==self.x.shape and
             z.dtype==cp.float64 and z.flags.c_contiguous,'projection must return a contiguous FP64 state')
        return z

    def coefficients(self,y,slot,already_projected=False):
        z=y if already_projected else self.projected(y,slot)
        a,r=self.coefficient(z,self.time_views[slot],RIGHT)
        need(isinstance(a,cp.ndarray) and isinstance(r,cp.ndarray) and
             a.shape==r.shape==self.x.shape and a.dtype==r.dtype==cp.float64 and
             a.flags.c_contiguous and r.flags.c_contiguous,'coefficients must be contiguous FP64 arrays')
        self.check_values(self.grid,(256,),(np.int32(self.n),z,a,r,self.flags))
        return z,a,r

    def flow(self,y,a,r,fraction,buffer):
        out=self.buffers[buffer]
        self.flow_kernel(self.grid,(256,),(np.int32(self.n),y,a,r,self.clock,np.float64(fraction),out))
        return out

    def trial(self):
        self.status.fill(0);self.flags.fill(0)
        self.make_times((1,),(1,),(self.clock,self.times,self.flags))
        z,a,r=self.coefficients(self.x,0)
        # Coefficient outputs may alias model workspaces. Own the shared values.
        cp.copyto(self.start_target,a);cp.copyto(self.start_rate,r)
        a0,r0=self.start_target,self.start_rate
        middle=self.flow(z,a0,r0,.5,0)
        _,a,r=self.coefficients(middle,2)
        full=self.projected(self.flow(z,a,r,1.,1),4)
        quarter=self.flow(z,a0,r0,.25,2)
        _,a,r=self.coefficients(quarter,1)
        half=self.projected(self.flow(z,a,r,.5,3),2)
        _,a,r=self.coefficients(half,2,already_projected=True)
        middle=self.flow(half,a,r,.25,4)
        _,a,r=self.coefficients(middle,3)
        self.fine=self.projected(self.flow(half,a,r,.5,5),4)
        self.norm_kernel(self.grid,(256,),(np.int32(self.n),np.int32(self.norm_size),full,self.fine,
            self.atol,self.rtol,self.lower,self.upper,self.status,self.flags))
        self.summarize((1,),(1,),(self.status,self.flags))

    def advance(self,ns,next_ns,min_ns,max_ns,*,boundaries=None,max_attempts=10000):
        if not self._gate.acquire(blocking=False):raise RuntimeError('engine already in use')
        try:
            need(self.handle is not None,'engine is closed')
            return self._advance(ns,next_ns,min_ns,max_ns,boundaries,max_attempts)
        finally:self._gate.release()

    def _advance(self,ns,next_ns,min_ns,max_ns,boundaries,max_attempts):
        need(type(ns) is int and ns>0 and type(next_ns) is int and next_ns>0,'integer epoch/step required')
        need(type(max_attempts) is int and 0<max_attempts<=10000,'invalid attempt budget')
        times=np.asarray(boundaries if boundaries is not None else [],dtype=np.float64)
        need(times.ndim==1 and np.isfinite(times).all(),'invalid boundary schedule')
        times=np.ascontiguousarray(np.unique(times))
        with self.stream:cp.copyto(self.backup,self.x)
        nxt=ct.c_long(next_ns);counts=(ct.c_long*3)();error=ct.c_double()
        started=time.perf_counter()
        code=self.lib.resident_advance(self.handle,ns,ct.byref(nxt),min_ns,max_ns,
            times.ctypes.data_as(ct.POINTER(ct.c_double)),len(times),max_attempts,counts,ct.byref(error))
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
        if not self._gate.acquire(blocking=False):raise RuntimeError('engine already in use')
        try:
            if self.handle is not None:self.lib.resident_destroy(self.handle);self.handle=None
        finally:self._gate.release()

    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()

"""ARKStep adapter: raw F, global JVP and full constant mass stay on one CUDA stream.
SUNDIALS owns Newton/Krylov/adaptation. Python callbacks are explicitly counted.
"""
import ctypes as ct,time
from pathlib import Path
import numpy as np
import cupy as cp
from model import Model,require
from coupled import Coupled
from runtime import PROFILES
H=Path(__file__).resolve().parent
CB=ct.CFUNCTYPE(ct.c_int,ct.c_int,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_double)

def library():
    lib=ct.CDLL(str(H/'libopenmatrix_ark.so'))
    lib.om_create.argtypes=[ct.c_long,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_double,ct.c_void_p,CB];lib.om_create.restype=ct.c_void_p
    lib.om_advance.argtypes=[ct.c_void_p,ct.c_double,ct.POINTER(ct.c_double)];lib.om_advance.restype=ct.c_int
    lib.om_stats.argtypes=[ct.c_void_p,ct.POINTER(ct.c_long)];lib.om_destroy.argtypes=[ct.c_void_p];lib.om_error.restype=ct.c_char_p
    return lib

class Implicit:
    def __init__(self,spec,profile='precise',time0=0.,state=None):
        started=time.perf_counter();require(profile in PROFILES,'profile')
        self.model=Model(spec,prepare_mass_solver=False);self.profile=profile;require(np.isfinite(time0),"initial time");self.t=time0;self.handle=None
        # Sufficient nonsingularity admission: strict row diagonal dominance with positive diagonal.
        # This accepts arbitrary sparse topology; it is not a claim to support every invertible M.
        mass=self.model.mass.tocsr();diag=mass.diagonal();off=np.asarray(abs(mass).sum(axis=1)).ravel()-abs(diag)
        require((diag>off).all(),'first GPU mass backend requires positive strictly row-diagonally-dominant M')
        initial=np.asarray(self.model.initial if state is None else state,dtype=np.float64)
        require(initial.shape==(self.model.n,) and np.isfinite(initial).all(),"initial state shape/finiteness")
        require(np.array_equal(initial[self.model.clamp_mask],self.model.clamp_values[self.model.clamp_mask]),"initial clamp state")
        self.stream=cp.cuda.Stream(non_blocking=True);self.operator=Coupled(self.model,self.stream)
        self.exception=None;self.calls=[0]*5;self.deadline=float('inf');self.unowned=[]
        with self.stream:
            self.x=cp.asarray(initial,dtype=cp.float64)
            self.mdiag=cp.asarray(diag);self.atol=cp.asarray(PROFILES[profile][1]*self.model.scale)
            self.ratol=cp.asarray(abs(mass)@(PROFILES[profile][1]*self.model.scale))
            self.precondition=cp.ElementwiseKernel('float64 r,float64 md,float64 jd,float64 gamma','float64 z','z=r/(md-gamma*jd);','om_precondition')
            self.callback=CB(self._call)
            self.operator.evaluate(time0,self.x)
            require(int(self.operator.flag.get(stream=self.stream)[0])==0,'invalid initial implicit state')
            self.lib=library();self.handle=self.lib.om_create(self.model.n,self.x.data.ptr,self.atol.data.ptr,self.ratol.data.ptr,PROFILES[profile][0],time0,self.stream.ptr,self.callback)
            require(self.handle is not None,'SUNDIALS initialization: '+self.lib.om_error().decode())
        self.setup_s=time.perf_counter()-started
    def view(self,ptr):
        require(ptr is not None,'null device vector')
        mem=cp.cuda.UnownedMemory(ptr,self.model.n*8,self)
        return cp.ndarray((self.model.n,),dtype=cp.float64,memptr=cp.cuda.MemoryPointer(mem,0))
    def _call(self,kind,t,yp,vp,op,gamma):
        try:
            require(time.perf_counter()<self.deadline,'implicit pilot wall budget exhausted')
            self.calls[kind]+=1
            with self.stream:
                out=self.view(op);x=self.view(yp) if yp else None;v=self.view(vp) if vp else None
                self.operator.flag.fill(0)
                if kind in (0,1):
                    f,j=self.operator.evaluate(t,x,v if kind==1 else None);cp.copyto(out,f if kind==0 else j)
                elif kind==2:self.operator.mass(v,out)
                elif kind==3:
                    self.operator.evaluate(t,x);self.precondition(v,self.mdiag,self.operator.diag,gamma,out)
                elif kind==4:cp.divide(v,self.mdiag,out=out)
                else:raise ValueError('callback kind')
                flag=int(self.operator.flag.get(stream=self.stream)[0])
                require(flag==0 and bool(cp.isfinite(out).all()),'nonfinite implicit operator/preconditioner')
            return 0
        except Exception as e:self.exception=f'{type(e).__name__}: {e}';return -1
    def advance(self,end,wall_limit_s=180):
        require(np.isfinite(end) and end>=self.t,'time order')
        if end==self.t:return
        self.deadline=time.perf_counter()+wall_limit_s;actual=ct.c_double(self.t)
        code=self.lib.om_advance(self.handle,end,ct.byref(actual));self.t=actual.value
        require(code==0,'implicit failure: '+str(self.exception or self.lib.om_error().decode()))
        require(abs(self.t-end)<1e-13,'implicit integration did not reach boundary')
        self.deadline=float('inf')
    def read(self):
        with self.stream:return self.x.get(stream=self.stream)
    def stats(self):
        out=(ct.c_long*5)();self.lib.om_stats(self.handle,out)
        return dict(zip(['steps','rhs_evaluations','nonlinear_iterations','linear_iterations','error_test_failures'],list(out)))|{'callbacks':self.calls.copy()}
    def close(self):
        if self.handle is not None:self.stream.synchronize();self.lib.om_destroy(self.handle);self.handle=None
    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()

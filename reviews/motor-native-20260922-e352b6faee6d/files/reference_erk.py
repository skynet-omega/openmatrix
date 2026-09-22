"""Existing SUNDIALS explicit RK8 reference. Smooth ODE and identity mass only."""
import ctypes as ct
from pathlib import Path
import time,json,sys,hashlib
import numpy as np
import cupy as cp
from model import Model,require
from native_ops import NativeOps
from extensions import fixture
class Reference:
    def __init__(self,spec,rtol=1e-12,atol_scale=1e-14):
        self.model=Model(spec);m=self.model;require(m.diagonal_mass and np.array_equal(m.mass_diag,np.ones(m.n)),'RK8 reference requires identity mass')
        self.stream=cp.cuda.Stream(non_blocking=True);self.op=NativeOps(m,self.stream);self.lib=ct.CDLL(str(Path(__file__).resolve().parent/'libreference_erk.so'));self.handle=None;self.t=0.
        self.lib.om_erk_create.argtypes=[ct.c_long,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_double,ct.c_void_p,ct.c_void_p];self.lib.om_erk_create.restype=ct.c_void_p
        self.lib.om_advance.argtypes=[ct.c_void_p,ct.c_double,ct.POINTER(ct.c_double)];self.lib.om_advance.restype=ct.c_int
        self.lib.om_destroy.argtypes=[ct.c_void_p];self.lib.om_error.restype=ct.c_char_p;self.lib.om_erk_stats.argtypes=[ct.c_void_p,ct.POINTER(ct.c_long)]
        with self.stream:
            self.x=cp.asarray(m.initial);self.atol=cp.asarray(atol_scale*m.scale)
            self.handle=self.lib.om_erk_create(m.n,self.x.data.ptr,self.atol.data.ptr,rtol,0,self.stream.ptr,self.op.handle)
        require(self.handle is not None,self.lib.om_error().decode())
    def advance(self,end,budget=230):
        self.op.limit(budget);actual=ct.c_double(self.t);code=self.lib.om_advance(self.handle,end,ct.byref(actual));self.t=actual.value
        require(code==0,self.op.error() or self.lib.om_error().decode())
    def read(self):
        with self.stream:return self.x.get(stream=self.stream)
    def close(self):
        if self.handle is not None:self.stream.synchronize();self.lib.om_destroy(self.handle);self.handle=None
        self.op.close()
    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();spec=fixture((90000,74700,2000),seed=0);spec['connections'][0]['offsets']=list(range(1,257));spec['connections'][0]['weight']=.2/256
    e=Reference(spec);m=e.model;setup=time.perf_counter()-started;times=np.linspace(0,10,41);probe=np.unique(np.linspace(0,m.n-1,4096,dtype=np.int64));ys=[e.read()[probe]];start=time.perf_counter();failure=None
    try:
        for t in times[1:]:e.advance(float(t),max(.01,235-(time.perf_counter()-started)));ys.append(e.read()[probe]);print(json.dumps({'time':e.t,'wall_s':time.perf_counter()-start}),flush=True)
    except Exception as exc:failure=repr(exc)
    final=e.read();advance=time.perf_counter()-start;counts=(ct.c_long*3)();e.lib.om_erk_stats(e.handle,counts);stats=list(counts);e.close()
    trace=np.asarray(ys).T;completed=failure is None and trace.shape==(4096,41) and e.t==10
    np.savez_compressed(out/'trajectory.npz',times=times[:len(ys)],probes=probe,trace=trace,final=final,scale=m.scale)
    r={'role':'independent existing ERK8 reference','rtol':1e-12,'atol_scale':1e-14,'model_identity':m.identity,'descriptor_hash':m.descriptor_hash,'states':m.n,'connections':m.connection.nnz,'simulated_s':e.t,'target_s':10,'completed':completed,'failure':failure,'setup_s':setup,'advance_scan_s':advance,'measured_window_s':time.perf_counter()-started,'steps_rhs_failures':stats,'scope':'shared descriptor/native RHS, different established integrator; check against independent RK4 reference, not biological validation'}
    (out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()

"""Own immutable GPU model storage and a compiled native per-operation graph dispatcher."""
import ctypes as ct
from pathlib import Path
import numpy as np
import cupy as cp
from model import require
from native_codegen import source
H=Path(__file__).resolve().parent

def attach(lib):
    lib.om_native_create.argtypes=[ct.c_char_p,ct.POINTER(ct.c_long),ct.POINTER(ct.c_ulonglong),ct.POINTER(ct.c_int),ct.c_void_p];lib.om_native_create.restype=ct.c_void_p
    lib.om_native_error.argtypes=[ct.c_void_p];lib.om_native_error.restype=ct.c_char_p
    lib.om_native_call.argtypes=[ct.c_void_p,ct.c_int,ct.c_double,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_double];lib.om_native_call.restype=ct.c_int
    lib.om_native_limit.argtypes=[ct.c_void_p,ct.c_double];lib.om_native_stats.argtypes=[ct.c_void_p,ct.POINTER(ct.c_long)];lib.om_native_destroy.argtypes=[ct.c_void_p]
    lib.om_create_native.argtypes=[ct.c_long,ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_double,ct.c_double,ct.c_void_p,ct.c_void_p];lib.om_create_native.restype=ct.c_void_p
    return lib

class NativeOps:
    def __init__(self,model,stream):
        self.model=model;self.stream=stream;self.handle=None;self.lib=attach(ct.CDLL(str(H/'libopenmatrix_ark.so')))
        require(model.n<2**31 and model.nin<2**31,'native index range')
        c=model.connection;mass=model.mass.tocsr()
        with stream:
            self.buffers=[cp.asarray(model.parameters),cp.asarray(model.clamp_mask),cp.asarray(model.port_bias),cp.zeros(model.nin),cp.asarray(c.data),cp.asarray(c.indices,dtype=cp.int32),cp.asarray(c.indptr,dtype=cp.int64),cp.empty(model.nout),cp.empty(model.nout),cp.empty(model.nin),cp.empty(model.nin),cp.asarray(mass.diagonal()),cp.asarray(mass.data),cp.asarray(mass.indices,dtype=cp.int32),cp.asarray(mass.indptr,dtype=cp.int64),cp.zeros(model.n)]
            self.scratch=cp.empty(model.n)
        stream.synchronize()
        sizes=(ct.c_long*3)(model.n,model.nin,len(model.pops));ptrs=(ct.c_ulonglong*16)(*[a.data.ptr for a in self.buffers]);counts=(ct.c_int*len(model.pops))(*[p['n'] for p in model.pops])
        self.generated=source(model);self.handle=self.lib.om_native_create(self.generated.encode(),sizes,ptrs,counts,stream.ptr)
        require(self.handle is not None,'native compile/capture failed: '+self.lib.om_native_error(self.handle).decode())
    def call(self,kind,t,x,v,out,gamma=0.):return self.lib.om_native_call(self.handle,kind,t,0 if x is None else x.data.ptr,0 if v is None else v.data.ptr,out.data.ptr,gamma)
    def error(self):return self.lib.om_native_error(self.handle).decode()
    def limit(self,seconds):self.lib.om_native_limit(self.handle,seconds)
    def stats(self):
        out=(ct.c_long*6)();self.lib.om_native_stats(self.handle,out)
        return {'native_callbacks':list(out)[:5],'diagonal_fallback_callbacks':out[5],'Python_callbacks_during_integration':0}
    def close(self):
        if self.handle is not None:self.stream.synchronize();self.lib.om_native_destroy(self.handle);self.handle=None
    def __del__(self):
        if getattr(self,'handle',None) is not None:self.close()

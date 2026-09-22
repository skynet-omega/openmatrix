"""Minimal FP64 cuBLAS calls on the current CUDA stream, including capture.

Uses the documented native API because this CuPy version rejects its BLAS
wrapper during capture. Every LU is still checked by info and full residual.
"""
from pathlib import Path
import ctypes as ct
import numpy as np
import cupy as cp

LIB=ct.CDLL(str(Path(cp.__file__).parent.parent/'nvidia/cublas/lib/libcublas.so.12'))
P=ct.c_void_p;I=ct.c_int
LIB.cublasCreate_v2.argtypes=[ct.POINTER(P)];LIB.cublasSetStream_v2.argtypes=[P,P]
LIB.cublasDgemm_v2.argtypes=[P,I,I,I,I,I,P,P,I,P,I,P,P,I]
LIB.cublasDgetrfBatched.argtypes=[P,I,P,I,P,P,I]
LIB.cublasDgetrsBatched.argtypes=[P,I,I,I,P,I,P,P,I,P,I]
HANDLE=P()
def check(code):
 if code:raise RuntimeError('Native cuBLAS status '+str(code))
check(LIB.cublasCreate_v2(ct.byref(HANDLE)))
ONE=ct.c_double(1.);ZERO=ct.c_double(0.)
def stream():check(LIB.cublasSetStream_v2(HANDLE,cp.cuda.get_current_stream().ptr))
def gemm(a,b):
 if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0] or a.dtype!=cp.float64 or b.dtype!=cp.float64:raise ValueError('FP64 matrix product contract')
 a=cp.ascontiguousarray(a);b=cp.ascontiguousarray(b);m,k=a.shape;n=b.shape[1];out=cp.empty((m,n))
 stream();check(LIB.cublasDgemm_v2(HANDLE,0,0,n,m,k,ct.byref(ONE),b.data.ptr,n,a.data.ptr,k,ct.byref(ZERO),out.data.ptr,n))
 return out
def solve(a,rhs):
 if a.ndim!=3 or rhs.ndim!=2 or a.shape[:2]!=rhs.shape or a.shape[1]!=a.shape[2]:raise ValueError('Batched square system contract')
 batch,n=rhs.shape
 aa=cp.ascontiguousarray(a.transpose(0,2,1));bb=rhs.copy()
 ap=cp.arange(batch,dtype=cp.uint64)*(n*n*8)+aa.data.ptr;bp=cp.arange(batch,dtype=cp.uint64)*(n*8)+bb.data.ptr
 pivot=cp.empty((batch,n),dtype=cp.int32);info=cp.empty(batch,dtype=cp.int32);hostinfo=ct.c_int()
 stream();check(LIB.cublasDgetrfBatched(HANDLE,n,ap.data.ptr,n,pivot.data.ptr,info.data.ptr,batch))
 check(LIB.cublasDgetrsBatched(HANDLE,0,n,1,ap.data.ptr,n,pivot.data.ptr,bp.data.ptr,n,ct.byref(hostinfo),batch))
 if hostinfo.value:raise RuntimeError('Invalid native triangular solve '+str(hostinfo.value))
 return bb,info

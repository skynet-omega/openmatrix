"""Batched FP64 affine membrane operators with unchanged midpoint equations.

Matrix basis contraction uses BLAS, followed by batched pivoted solves. No
fixed anatomy, coordinate count, skipped coefficient or relaxed residual.
The adapter supplies kinetics and conductance basis arrays.
"""
import cupy as cp
from native_dense import gemm,solve as dense_solve

def step(b,v,g,ge,gi,dt,current,*,check=True):
 C2=b.C*(2./dt)
 conductance=gemm(ge+gi,b.shuntG)
 drive=gemm(-b.rest*ge+(-68.-b.rest)*gi,b.shuntb)+current
 def electrical(gates):
  m,h,p,n=(gates[:,:,j] for j in range(4))
  f=cp.stack((m*m*m*h,p,n*n*n*n),axis=-1).reshape(b.n,-1)
  return (gemm(f,b.chanG)+conductance).reshape(b.n,b.ports,b.ports)+b.G,gemm(f*b.ena,b.chanb)+drive
 def solve(K,rhs):
  A=K+C2;x,info=dense_solve(A,rhs)
  residual=(A*x[:,None,:]).sum(axis=2)-rhs
  scale=cp.max(cp.sum(cp.abs(A),axis=2),axis=1)*cp.max(cp.abs(x),axis=1)+cp.max(cp.abs(rhs),axis=1)
  error=cp.max(cp.abs(residual),axis=1)/cp.maximum(scale,1e-300)
  return x,cp.where(cp.isfinite(error)&(error<=1e-12)&cp.isfinite(x).all(axis=1)&(info==0),error,cp.inf)
 s,t=b.rates(v+b.rest);gh=s+(g-s)*cp.exp(-.5*dt/t);K,rhs=electrical(gh)
 pred,e1=solve(K,gemm(v,C2.T)+rhs)
 sm,tm=b.rates(pred+b.rest);gm=sm+(g-sm)*cp.exp(-.5*dt/tm);K,rhs=electrical(gm)
 out,e2=solve(K,gemm(v,C2.T)-(K*v[:,None,:]).sum(axis=2)+2*rhs)
 gates=sm+(g-sm)*cp.exp(-dt/tm)
 valid=cp.isfinite(gates).all(axis=(1,2))&((gates>=0)&(gates<=1)).all(axis=(1,2))
 errors=cp.stack((cp.where(valid,e1,cp.inf),cp.where(valid,e2,cp.inf)),axis=1)
 if check:
  if not bool(cp.isfinite(errors).all()):raise FloatingPointError('Batched membrane solve failed')
  return out,gates
 return out,gates,errors

def install(brain):
 import kc_trial_graph
 b=brain._spatial_batch
 while hasattr(b,'base'):b=b.base
 if getattr(b,'_motor_trial_graphs',None):raise ValueError('Install membrane executor before first advance')
 old=kc_trial_graph.step;kc_trial_graph.step=step
 return lambda:setattr(kc_trial_graph,'step',old)

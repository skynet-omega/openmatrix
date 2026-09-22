"""Integrate the B linear solver into the unchanged original PN step.

This does not implement another complete PN step: original Newton, chemistry,
commit, timestep and coefficients are used. Full residual and fallback retained.
"""
import numpy as np
from scipy.sparse import csr_matrix,diags
from cyclic_plan import CyclicPlan
from cyclic_tree import CyclicTree

def install(p):
 backend=p.backend;original=backend.solve;extra=backend.M-diags(backend.C);plan=CyclicPlan(backend.G,backend.M,np.unique(extra.nonzero()[0]));cache={};statistics={'calls':0,'fallbacks':0}
 def solve(rhs,*,shift,rtol,atol,maxiter,diagonal_update,**kw):
  backend.assert_model();nodes,values=diagonal_update;nodes=np.asarray(nodes);values=np.asarray(values)
  if nodes.ndim!=1 or nodes.dtype.kind not in 'iu' or values.shape!=nodes.shape or len(np.unique(nodes))!=len(nodes) or np.any(nodes<0) or np.any(nodes>=len(rhs)) or not np.isfinite(values).all():raise ValueError('Invalid diagonal update')
  if not np.isfinite(shift) or shift<=0 or not np.isfinite([rtol,atol]).all() or min(rtol,atol)<0 or rtol+atol<=0 or maxiter<1:raise ValueError('Invalid solver controls')
  if shift not in cache:cache.clear();cache[shift]=CyclicTree(plan,shift)
  gpu=cache[shift];diagonal=np.zeros_like(rhs);diagonal[nodes]=values;passive,positions=backend._operator(shift);data=passive.data.copy();data[positions[nodes]]+=values;A=csr_matrix((data,passive.indices,passive.indptr),shape=passive.shape,copy=False)
  limit=max(atol,rtol*float(np.linalg.norm(rhs)));statistics['calls']+=1
  try:
   x=gpu.solve(rhs,diagonal)
   for iteration in range(1,min(maxiter,3)+1):
    r=rhs-A@x;norm=float(np.linalg.norm(r))
    if np.isfinite(norm) and norm<=limit:return x,dict(info=0,iterations=iteration,fine_residual_passed=True,residual_l2_pA=norm,threshold_pA=limit,linear_solver='GPU independent-set contraction with deterministic gather',graph_fallback=False)
    if iteration<min(maxiter,3):x+=gpu.solve(r,diagonal)
  except ArithmeticError:pass
  statistics['fallbacks']+=1
  return original(rhs,shift=shift,rtol=rtol,atol=atol,maxiter=maxiter,diagonal_update=diagonal_update,**kw)
 return solve,original,statistics

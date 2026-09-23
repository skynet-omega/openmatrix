"""Algebraic elimination of degree<=2 nodes, retaining all full-mass unknowns.

A numerical factorization, not a neuronal reduction. Non-symmetric roundoff
entries retain their separate directions. Caller checks the original residual.
"""
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit

class EliminationPlan:
 def __init__(self,G,M,protected):
  self.G=csr_matrix(G);self.M=csr_matrix(M);n=G.shape[0]
  pattern=self.G.copy();pattern.data=np.ones_like(pattern.data)
  p=self.M.copy();p.data=np.ones_like(p.data);pattern=(pattern+p).tocsr();pattern.setdiag(0);pattern.eliminate_zeros()
  coo=pattern.tocoo();pairs=sorted({(min(int(i),int(j)),max(int(i),int(j))) for i,j in zip(coo.row,coo.col)})
  adj=[set() for _ in range(n)];edge={key:k for k,key in enumerate(pairs)}
  for i,j in pairs:adj[i].add(j);adj[j].add(i)
  blocked=set(map(int,protected));queue=[i for i in range(n) if i not in blocked and len(adj[i])<=2];removed=np.zeros(n,dtype=bool);steps=[]
  k=0
  while k<len(queue):
   i=queue[k];k+=1
   if removed[i] or i in blocked or len(adj[i])>2:continue
   neighbors=sorted(adj[i]);removed[i]=True;js=neighbors+[-1]*(2-len(neighbors))
   es=[edge[tuple(sorted((i,j)))] for j in neighbors]+[-1]*(2-len(neighbors));cross=-1
   for j in neighbors:adj[j].remove(i)
   if len(neighbors)==2:
    a,b=neighbors
    if b not in adj[a]:
     edge[(a,b)]=len(pairs);pairs.append((a,b));adj[a].add(b);adj[b].add(a)
    cross=edge[(a,b)]
   steps.append([i,*js,*es,cross]);adj[i].clear()
   for j in neighbors:
    if j not in blocked and len(adj[j])<=2:queue.append(j)
  self.steps=np.asarray(steps,dtype=np.int64).reshape(-1,6);self.core=np.flatnonzero(~removed);self.pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
  if len(self.core)>256:raise ValueError(f'Unbounded retained junction system:{len(self.core)}')
  i,j=self.pairs.T
  self.base=np.stack([np.asarray(self.G[i,j]).ravel(),np.asarray(self.G[j,i]).ravel()],axis=1) if len(i) else np.zeros((0,2))
  self.mass=np.stack([np.asarray(self.M[i,j]).ravel(),np.asarray(self.M[j,i]).ravel()],axis=1) if len(i) else np.zeros((0,2))
  self.dg=self.G.diagonal();self.dm=self.M.diagonal()
  ci={int(row):i for i,row in enumerate(self.core)}
  self.core_edges=np.asarray([[edge[(i,j)],ci[i],ci[j]] for i in self.core for j in sorted(adj[i]) if i<j],dtype=np.int64).reshape(-1,3)

 def solve(self,rhs,shift,diagonal):
  diag=self.dg+shift*self.dm+diagonal;values=self.base+shift*self.mass;b=np.array(rhs,dtype=float,copy=True)
  ok=eliminate(self.steps,self.pairs,diag,values,b)
  if not ok:raise ArithmeticError('Nonpositive/nonfinite elimination pivot; use general solver')
  A=np.diag(diag[self.core]);fill_core(self.core_edges,values,A)
  x=np.zeros_like(b)
  if len(self.core):x[self.core]=np.linalg.solve(A,b[self.core])
  back(self.steps,self.pairs,diag,values,b,x)
  return x

@njit(cache=True)
def eliminate(steps,pairs,d,values,b):
 for s in steps:
  i,j,k,e,f,c=s;pivot=d[i]
  if not np.isfinite(pivot) or pivot<=0:return False
  if j<0:continue
  ij=0 if pairs[e,0]==i else 1;out1=values[e,ij];in1=values[e,1-ij]
  d[j]-=in1*out1/pivot;b[j]-=in1*b[i]/pivot
  if k<0:continue
  ik=0 if pairs[f,0]==i else 1;out2=values[f,ik];in2=values[f,1-ik]
  d[k]-=in2*out2/pivot;b[k]-=in2*b[i]/pivot
  # j<k because symbolic neighbors were sorted.
  values[c,0]-=in1*out2/pivot;values[c,1]-=in2*out1/pivot
 return True

@njit(cache=True)
def fill_core(edges,values,A):
 for e,i,j in edges:A[i,j]=values[e,0];A[j,i]=values[e,1]

@njit(cache=True)
def back(steps,pairs,d,values,b,x):
 for index in range(len(steps)-1,-1,-1):
  i,j,k,e,f,_=steps[index];v=b[i]
  if j>=0:v-=values[e,0 if pairs[e,0]==i else 1]*x[j]
  if k>=0:v-=values[f,0 if pairs[f,0]==i else 1]*x[k]
  x[i]=v/d[i]


from scipy.sparse import diags
from collections import OrderedDict
from pn_mass_krylov_backend import FullMassKrylovBackend
from pn_mass_backend import immutable

POLICY='PN_full_mass_graph_elimination_true_residual_Krylov_fallback_v1'

class FullMassGraphBackend(FullMassKrylovBackend):
 @classmethod
 def adopt(cls,parent):
  if type(parent) is not FullMassKrylovBackend:raise ValueError('Exact Krylov full-mass parent required')
  parent.assert_model();obj=cls.__new__(cls);obj.__dict__.update(parent.__dict__)
  extra=obj.M-diags(obj.C);extra.eliminate_zeros();protected=np.unique(extra.nonzero()[0])
  obj._graph_plan=EliminationPlan(obj.G,obj.M,protected)
  obj._graph_buffers={}
  for key in ('steps','core','pairs','base','mass','dg','dm','core_edges'):
   a=immutable(getattr(obj._graph_plan,key));setattr(obj._graph_plan,key,a);obj._graph_buffers[key]=a
  obj.graph_solver_policy=POLICY
  obj._graph_operators=OrderedDict()
  return obj

 def _operator(self,shift):
  key=float(shift)
  if key not in self._graph_operators:
   A=(self.G+shift*self.M).tocsr();A.sort_indices()
   positions=np.flatnonzero(A.indices==np.repeat(np.arange(A.shape[0]),np.diff(A.indptr)))
   if len(positions)!=A.shape[0]:
    A.setdiag(A.diagonal());A.sort_indices()
    positions=np.flatnonzero(A.indices==np.repeat(np.arange(A.shape[0]),np.diff(A.indptr)))
   for field in ('data','indices','indptr'):setattr(A,field,immutable(getattr(A,field)))
   positions=immutable(positions)
   self._graph_operators[key]=(A,positions,(A.data,A.indices,A.indptr))
   if len(self._graph_operators)>4:self._graph_operators.popitem(last=False)
  A,positions,buffers=self._graph_operators[key]
  if any(getattr(A,k) is not v or v.flags.writeable for k,v in zip(('data','indices','indptr'),buffers)):
   raise ValueError('Cached passive operator changed')
  return A,positions

 def solve(self,rhs,*,shift,rtol,atol,maxiter,diagonal_update,**unused):
  self.assert_model()
  if self.graph_solver_policy!=POLICY or any(getattr(self._graph_plan,k) is not v or v.flags.writeable for k,v in self._graph_buffers.items()):
   raise ValueError('Graph numerical plan changed')
  rhs=np.asarray(rhs,dtype=float);nodes,values=diagonal_update;nodes=np.asarray(nodes);values=np.asarray(values,dtype=float)
  if (not np.isscalar(shift) or not np.isfinite(shift) or shift<=0 or rhs.shape!=self.C.shape or not np.isfinite(rhs).all()
      or not np.isscalar(rtol) or not np.isscalar(atol) or not np.isfinite([rtol,atol]).all() or rtol<0 or atol<0
      or rtol+atol<=0 or type(maxiter) is not int or maxiter<1):raise ValueError('Finite positive-tolerance electrical system required')
  if (nodes.dtype.kind not in 'iu' or nodes.shape!=values.shape or nodes.ndim!=1 or not np.isfinite(values).all()
      or np.any(nodes<0) or np.any(nodes>=len(rhs)) or len(np.unique(nodes))!=len(nodes)):
   raise ValueError('Explicit unique signed diagonal correction required')
  diagonal=np.zeros_like(rhs);diagonal[nodes]=values
  passive,positions=self._operator(shift);data=passive.data.copy();data[positions[nodes]]+=values
  A=csr_matrix((data,passive.indices,passive.indptr),shape=passive.shape,copy=False)
  limit=max(float(atol),float(rtol)*float(np.linalg.norm(rhs)));reason=None;norm=float('inf');iterations=0
  try:
   x=self._graph_plan.solve(rhs,shift,diagonal);iterations=1;residual=rhs-A@x;norm=float(np.linalg.norm(residual))
   while norm>limit and iterations<min(maxiter,3):
    x+=self._graph_plan.solve(residual,shift,diagonal);iterations+=1;residual=rhs-A@x;norm=float(np.linalg.norm(residual))
   if not np.isfinite(x).all() or not np.isfinite(norm) or norm>limit:reason='Original unpreconditioned residual failed'
  except (ArithmeticError,np.linalg.LinAlgError) as error:reason=str(error)
  if reason is not None:
   x,report=super().solve(rhs,shift=shift,rtol=rtol,atol=atol,maxiter=maxiter,diagonal_update=(nodes,values))
   report.update(graph_policy=POLICY,graph_fallback=True,graph_failure=reason,graph_true_residual_l2_pA=norm)
   return x,report
  return x,dict(info=0,iterations=iterations,fine_residual_passed=True,residual_l2_pA=norm,threshold_pA=limit,
   graph_policy=POLICY,graph_fallback=False,direct_fallback=False,junction_core=len(self._graph_plan.core),
   linear_solver='Degree<=2 elimination plus small junction solve and full residual',
   residual_scope='Original full generalized equation; not residual in omitted fine volume')

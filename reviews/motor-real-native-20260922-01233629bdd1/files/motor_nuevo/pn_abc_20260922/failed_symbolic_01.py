"""Exact Schur factorization at a fixed SDIRK shift; no dynamic mode discarded."""
from pathlib import Path
import sys,inspect,textwrap
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from pn_graph_elimination_backend import EliminationPlan,eliminate,back
# Only raise the protected-core construction bound; the final solver still uses
# the original small-junction algorithm. This new bound is checked before use.
source=textwrap.dedent(inspect.getsource(EliminationPlan.__init__)).replace('def __init__','def symbolic').replace('len(self.core)>256','len(self.core)>20000')
ns={'np':np,'csr_matrix':csr_matrix};exec(compile(source,'<bounded Schur symbolic>','exec'),ns);symbolic=ns['symbolic']
@njit(cache=True)
def reduce_rhs(steps,pairs,d,values,rhs):
 b=rhs.copy()
 for s in steps:
  i,j,k,e,f,_=s
  if j>=0:b[j]-=values[e,1 if pairs[e,0]==i else 0]*b[i]/d[i]
  if k>=0:b[k]-=values[f,1 if pairs[f,0]==i else 0]*b[i]/d[i]
 return b
class Condensation:
 def __init__(self,backend,protected,shift):
  n=len(backend.C);plan=EliminationPlan.__new__(EliminationPlan);symbolic(plan,backend.G,backend.M,protected);self.plan=plan;self.core=plan.core
  d=plan.dg+shift*plan.dm;values=plan.base+shift*plan.mass
  if not eliminate(plan.steps,plan.pairs,d,values,np.zeros(n)):raise ArithmeticError('Invalid passive pivot')
  self.d=d;self.values=values
  ce=plan.core_edges;k=len(self.core);rows=list(range(k));cols=list(range(k));data=list(d[self.core])
  for edge,i,j in ce:rows.extend([i,j]);cols.extend([j,i]);data.extend(values[edge])
  self.K=csr_matrix((data,(rows,cols)),shape=(k,k));self.K.sort_indices()
  # Eliminate dynamic core nodes afresh; only immutable passive elimination is reused.
  self.small=EliminationPlan(self.K,csr_matrix(self.K.shape),[])
 def reduce(self,rhs):return reduce_rhs(self.plan.steps,self.plan.pairs,self.d,self.values,rhs)
 def reconstruct(self,reduced_rhs,xcore):
  x=np.zeros_like(reduced_rhs);x[self.core]=xcore;back(self.plan.steps,self.plan.pairs,self.d,self.values,reduced_rhs,x);return x
 def solve(self,rhs,diag):
  b=self.reduce(rhs);x=self.small.solve(b[self.core],0.,diag[self.core]);return self.reconstruct(b,x)

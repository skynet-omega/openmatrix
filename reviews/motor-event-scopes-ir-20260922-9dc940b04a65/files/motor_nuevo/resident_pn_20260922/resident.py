"""Resident electrical PN. Explicit small CPU boundary for release chemistry."""
from pathlib import Path
import sys
import numpy as np
import cupy as cp
from scipy.sparse import diags
H=Path(__file__).resolve().parent;P=H.parent/'pn_abc_20260922';sys.path[:0]=[str(P),str(P/'vendor')]
from portable import PortablePN,read
from cyclic_plan import CyclicPlan
from cyclic_tree import CyclicTree
module=cp.RawModule(code=(H/'gpu_kernels.cu').read_text(),options=('--fmad=false',))
channel_kernel=module.get_function('channels');ca_kernel=module.get_function('calcium');diff_kernel=module.get_function('difference');csr_kernel=module.get_function('csr')

def channels(v,base,q,gbar,E):
 n=len(v);x=cp.empty_like(base);ionic=cp.empty((n,3));jac=cp.empty(n);ge=cp.empty(n);flag=cp.zeros(1,dtype=cp.int32)
 channel_kernel(((n+127)//128,),(128,),(np.int32(n),v,base,np.float64(q),gbar,E,x,ionic,jac,ge,flag))
 error=float(cp.max(ge))
 if int(flag.get()[0]):raise FloatingPointError('Invalid resident Na/K kinetics')
 return x,ionic,jac,error

def channel_conductance(x,gbar):
 m,h,p,n=x.T;return gbar*cp.column_stack((m**3*h,p,n**4))

class Backend:
 def __init__(self,parent):
  self.parent=parent;self.cp=cp;self.C=cp.asarray(parent.C);self.levels=[{'C':self.C}]
  def csr(A):return (cp.asarray(A.indptr,dtype=cp.int32),cp.asarray(A.indices,dtype=cp.int32),cp.asarray(A.data))
  self.G=csr(parent.G);self.M=csr(parent.M);self.gsum=cp.asarray(parent._G_sum);self.csr=csr;extra=parent.M-diags(parent.C);self.plan=CyclicPlan(parent.G,parent.M,np.unique(extra.nonzero()[0]));self.shift=None;self.tree=None;self.count=0;self.fallbacks=0
 def assert_model(self):self.parent.assert_model()
 def action(self,v):
  out=cp.empty_like(v);diff_kernel(((len(v)+127)//128,),(128,),(np.int32(len(v)),*self.G,self.gsum,v,out));return out
 def product(self,A,v,minimum):
  out=cp.empty_like(v);csr_kernel(((len(v)+127)//128,),(128,),(np.int32(len(v)),*A,v,np.int32(minimum),out));return out
 def mass_action(self,v):return self.product(self.M,v,16)
 def solve(self,rhs,*,shift,rtol,atol,maxiter,diagonal_update,**kw):
  nodes,values=diagonal_update
  if self.shift!=shift:
   cp.cuda.get_current_stream().synchronize();self.tree=CyclicTree(self.plan,shift);self.A=self.csr((self.parent.G+shift*self.parent.M).tocsr());self.shift=shift
  tree=self.tree;diagonal=cp.zeros_like(rhs);diagonal[nodes]=values;limit=max(atol,rtol*float(cp.linalg.norm(rhs)));self.count+=1
  def solve_vector(r):
   cp.copyto(tree.rhs,r);cp.copyto(tree.diagonal,diagonal);tree.graph.launch(stream=cp.cuda.get_current_stream());return tree.x.copy()
  x=solve_vector(rhs)
  for iteration in range(1,min(maxiter,3)+1):
   r=rhs-(self.product(self.A,x,1000000000)+diagonal*x);norm=float(cp.linalg.norm(r));flag=int(tree.flag.get()[0])
   if not flag and np.isfinite(norm) and norm<=limit:return x,dict(info=0,iterations=iteration,fine_residual_passed=True,residual_l2_pA=norm,threshold_pA=limit,graph_fallback=False,linear_solver='Resident cyclic GPU')
   if iteration<min(maxiter,3):x+=solve_vector(r)
  # Preserve the original recovery policy rather than accepting a bad solve.
  self.fallbacks+=1;x,r=self.parent.solve(cp.asnumpy(rhs),shift=shift,rtol=rtol,atol=atol,maxiter=maxiter,diagonal_update=(cp.asnumpy(nodes),cp.asnumpy(values)),**kw)
  r['resident_fallback']=True;return cp.asarray(x),r

class Calcium:
 def __init__(self,parent):
  self.parent=parent;self.nodes=parent.nodes;self.gates=cp.asarray(parent.gates);self.parameters=tuple(cp.asarray(a,dtype=cp.float64) for a in (parent.half,parent.slope,parent.tau,parent.power,parent.gbar))
  if len(parent.power)>4:raise ValueError('Kernel supports at most4 Ca gates')
 @property
 def time_ns(self):return self.parent.time_ns
 def assert_state(self):self.parent.assert_state()
 def stage(self,v,base,q):
  n=len(v);k=len(self.parent.power);x=cp.empty_like(base);cur=cp.empty(n);jac=cp.empty(n);frozen=cp.empty(n);ge=cp.empty(n);flag=cp.zeros(1,dtype=cp.int32)
  ca_kernel(((n+127)//128,),(128,),(np.int32(n),np.int32(k),v,base,np.float64(q),*self.parameters,np.float64(self.parent.reversal),np.int32(self.parent.enabled),x,cur,jac,frozen,ge,flag))
  error=float(cp.max(ge))
  if int(flag.get()[0]):raise FloatingPointError('Invalid resident calcium kinetics')
  return dict(gates=x,current=cur,jacobian=jac,frozen_conductance=frozen,gate_error=error)
 def final_proposal(self,dt,first,second):
  a={'current':cp.asnumpy(first['current'])};b={'current':cp.asnumpy(second['current']),'gates':cp.asnumpy(second['gates'])};self.pending=second['gates'];return self.parent.final_proposal(dt,a,b)
 def commit(self,p):self.parent.commit(p);self.gates=self.pending

class ResidentPN:
 def __init__(self,folder):
  self.host=PortablePN(folder);p=self.host;self.cp=cp;self.backend=Backend(p.backend);self.active_nodes=p.active_nodes;self._nodes_gpu=cp.asarray(p.active_nodes)
  self.gbar_nS=cp.asarray(p.gbar_nS);self.reversal_mV=cp.asarray(p.reversal_mV);self.leak_reversal_mV=p.leak_reversal_mV;self.calcium_port=Calcium(p.calcium_port);self.restore(p.state())
 def _assert_model(self):self.host._assert_model()
 def restore(self,state):
  self.host.restore(state);self.voltage=cp.asarray(state['voltage']);self.gates=cp.asarray(state['gates']);self.ionic_charge_pC=cp.asarray(state['charge']);self.time_ns=state['time_ns'];self.calcium_port.gates=cp.asarray(self.host.calcium_port.gates)
 def state(self):return dict(voltage=cp.asnumpy(self.voltage),gates=cp.asnumpy(self.gates),charge=cp.asnumpy(self.ionic_charge_pC),time_ns=self.time_ns,calcium=self.host.calcium_port.state_dict())
 def advance(self,dt,current,**options):
  from resident_step import advance_resident
  return advance_resident(self,dt,current,_local_channel=self.calcium_port,**options)

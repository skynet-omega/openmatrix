from pathlib import Path
import numpy as np
import cupy as cp
class CyclicTree:
 def __init__(self,plan,shift):
  self.plan=plan;self.level_count=len(plan.rounds);self.maximum_parallel=max(c for _,c in plan.rounds);self.order=cp.arange(len(plan.steps),dtype=cp.int64)
  self.steps=cp.asarray(plan.steps);self.pairs=cp.asarray(plan.pairs);self.core=cp.asarray(plan.core);self.edges=cp.asarray(plan.core_edges)
  self.base_d=cp.asarray(plan.dg+shift*plan.dm);self.base_v=cp.asarray(plan.base+shift*plan.mass)
  self.diagonal=cp.zeros(len(plan.dg));self.rhs=cp.zeros(len(plan.dg));self.d=cp.zeros_like(self.base_d);self.v=cp.zeros_like(self.base_v);self.b=cp.zeros_like(self.rhs);self.x=cp.zeros_like(self.rhs)
  self.flag=cp.zeros(1,dtype=cp.int32);n=len(plan.core);self.matrix=cp.zeros((n,n));self.y=cp.zeros(n);self.msg=cp.zeros((len(plan.steps),6))
  self.gathers=[tuple(cp.asarray(x) for x in row) for row in plan.gathers]
  path=Path(__file__).resolve().parents[2]/'pn_abc_20260922';code=(path/'parallel_tree.cu').read_text()+(path/'cyclic_tree.cu').read_text()+(path/'parallel_core.cu').read_text();module=cp.RawModule(code=code,options=('--fmad=false',));forward=module.get_function('messages');gather=module.get_function('gather');back=module.get_function('back_level');core=module.get_function('core_parallel')
  if n>64:raise ValueError('Parallel core implementation bound64')
  self.stream=cp.cuda.Stream(non_blocking=True)
  def launch():
   cp.add(self.base_d,self.diagonal,out=self.d);cp.copyto(self.v,self.base_v);cp.copyto(self.b,self.rhs);self.x.fill(0);self.flag.fill(0)
   for (start,count),(targets,offsets,slots) in zip(plan.rounds,self.gathers):
    forward(((count+127)//128,),(128,),(self.steps,self.pairs,np.int32(start),np.int32(count),self.d,self.v,self.b,self.msg,self.flag))
    gather(((len(targets)+127)//128,),(128,),(targets,offsets,slots,np.int32(len(targets)),np.int32(len(plan.dg)),self.msg,self.d,self.v,self.b))
   core((1,),(64,),(self.core,self.edges,np.int32(n),np.int32(len(plan.core_edges)),self.d,self.v,self.b,self.x,self.flag),shared_mem=(n*n+n)*8)
   for start,count in plan.rounds[::-1]:back(((count+127)//128,),(128,),(self.steps,self.pairs,self.order,np.int32(start),np.int32(count),self.d,self.v,self.b,self.x))
  # The arrays above were initialized on the caller stream.
  # The private nonblocking stream must wait before reading them.
  cp.cuda.get_current_stream().synchronize()
  with self.stream:
   launch();self.stream.synchronize();self.stream.begin_capture();launch();self.graph=self.stream.end_capture()
 def solve(self,rhs,diagonal):
  with self.stream:
   self.rhs.set(rhs,stream=self.stream);self.diagonal.set(diagonal,stream=self.stream);self.graph.launch(self.stream)
   result=self.x.get(stream=self.stream);flag=int(self.flag.get(stream=self.stream)[0])
  if flag or not np.isfinite(result).all():raise ArithmeticError('Invalid cyclic pivot/state')
  return result

from pathlib import Path
import numpy as np
import cupy as cp

class ParallelTree:
 def __init__(self,plan,shift):
  self.plan=plan;last_node=np.zeros(len(plan.dg),dtype=np.int64);last_edge=np.zeros(len(plan.pairs),dtype=np.int64);levels=[]
  for i,j,k,e,f,c in plan.steps:
   nodes=[x for x in (i,j,k) if x>=0];edges=[x for x in (e,f,c) if x>=0];level=1+max([last_node[x] for x in nodes]+[last_edge[x] for x in edges]);levels.append(level)
   for x in nodes:last_node[x]=level
   for x in edges:last_edge[x]=level
  levels=np.asarray(levels);order=np.argsort(levels,kind='stable').astype(np.int64);counts=np.bincount(levels)[1:];offsets=np.r_[0,np.cumsum(counts[:-1])]
  self.level_count=len(counts);self.maximum_parallel=int(max(counts));self.order=cp.asarray(order)
  self.steps=cp.asarray(plan.steps);self.pairs=cp.asarray(plan.pairs);self.core=cp.asarray(plan.core);self.edges=cp.asarray(plan.core_edges)
  self.base_d=cp.asarray(plan.dg+shift*plan.dm);self.base_v=cp.asarray(plan.base+shift*plan.mass)
  self.diagonal=cp.zeros(len(plan.dg));self.rhs=cp.zeros(len(plan.dg));self.d=cp.zeros_like(self.base_d);self.v=cp.zeros_like(self.base_v);self.b=cp.zeros_like(self.rhs);self.x=cp.zeros_like(self.rhs)
  self.flag=cp.zeros(1,dtype=cp.int32);n=len(plan.core);self.matrix=cp.zeros((n,n));self.y=cp.zeros(n)
  if n>256:raise ValueError('Bounded core exceeded')
  module=cp.RawModule(code=(Path(__file__).parent/'parallel_tree.cu').read_text(),options=('--fmad=false',));forward=module.get_function('eliminate_level');back=module.get_function('back_level');core=module.get_function('core_solve')
  self.stream=cp.cuda.Stream(non_blocking=True)
  def launch():
   cp.add(self.base_d,self.diagonal,out=self.d);cp.copyto(self.v,self.base_v);cp.copyto(self.b,self.rhs);self.x.fill(0);self.flag.fill(0)
   for count,offset in zip(counts,offsets):
    forward(((int(count)+127)//128,),(128,),(self.steps,self.pairs,self.order,np.int32(offset),np.int32(count),self.d,self.v,self.b,self.flag))
   core((1,),(1,),(self.core,self.edges,np.int32(n),np.int32(len(plan.core_edges)),self.d,self.v,self.b,self.x,self.matrix,self.y,self.flag))
   for count,offset in zip(counts[::-1],offsets[::-1]):
    back(((int(count)+127)//128,),(128,),(self.steps,self.pairs,self.order,np.int32(offset),np.int32(count),self.d,self.v,self.b,self.x))
  with self.stream:
   launch();self.stream.synchronize();self.stream.begin_capture();launch();self.graph=self.stream.end_capture()
 def solve(self,rhs,diagonal):
  with self.stream:
   self.rhs.set(rhs,stream=self.stream);self.diagonal.set(diagonal,stream=self.stream);self.graph.launch(self.stream)
   result=self.x.get(stream=self.stream);flag=int(self.flag.get(stream=self.stream)[0])
  if flag or not np.isfinite(result).all():raise ArithmeticError('Invalid parallel pivot/state')
  return result

"""Two speculative Newton corrections captured on GPU, with original fallback.

A failed trial never commits state. Original full residual, gates, signed
Jacobian, linear error and nonlinear decrease decide whether speculation is used.
"""
import numpy as np
import cupy as cp
from resident import module,channel_kernel,ca_kernel,diff_kernel,csr_kernel
from cyclic_tree import CyclicTree
scatter=module.get_function('scatter_current');syn_current=module.get_function('synapse_current')

def call_scatter(nodes,values,stride,out):
 if len(nodes):scatter(((len(nodes)+127)//128,),(128,),(np.int32(len(nodes)),nodes,values,np.int32(stride),out))

class StageGraph:
 def __init__(self,p,syn,shift,q):
  self.p=p;self.shift=shift;self.q=q;self.n=len(p.voltage);self.active=p._nodes_gpu;self.cn=cp.asarray(p.calcium_port.nodes);self.sn=cp.asarray(syn[0]) if syn is not None else cp.empty(0,dtype=cp.int64)
  self.vbase=cp.empty_like(p.voltage);self.guess=cp.empty_like(p.voltage);self.reference=cp.empty_like(p.voltage);self.current=cp.empty_like(p.voltage);self.xbase=cp.empty_like(p.gates);self.cbase=cp.empty_like(p.calcium_port.gates)
  self.sg=cp.zeros(len(self.sn));self.se=cp.zeros(len(self.sn));self.v=cp.zeros_like(p.voltage);self.diagonal=cp.zeros_like(p.voltage);self.signed_work=cp.zeros_like(p.voltage);self.flag=cp.zeros(1,dtype=cp.int32)
  self.linear_residual=cp.zeros_like(p.voltage);self.linear_product=cp.zeros_like(p.voltage);self.linear_diag=cp.zeros_like(p.voltage)
  self.tree=CyclicTree(p.backend.plan,shift);self.A=p.backend.csr((p.backend.parent.G+shift*p.backend.parent.M).tocsr());self.keep=[]
  def arrays():
   z={k:cp.empty(self.n) for k in ('r','delta','Gv','Mv')}
   z.update(vc=cp.empty(len(self.active)),cc=cp.empty(len(self.cn)),x=cp.empty_like(p.gates),ionic=cp.empty((len(self.active),3)),jac=cp.empty(len(self.active)),ge=cp.empty(len(self.active)),cg=cp.empty_like(self.cbase),ci=cp.empty(len(self.cn)),cj=cp.empty(len(self.cn)),cf=cp.empty(len(self.cn)),ce=cp.empty(len(self.cn)))
   return z
  self.evals=[arrays() for _ in range(4)]
  # Warm numerical data avoids meaningless invalid pivots during capture warmup.
  self.vbase[:]=p.voltage;self.guess[:]=p.voltage;self.reference[:]=p.voltage;self.current.fill(0);self.xbase[:]=p.gates;self.cbase[:]=p.calcium_port.gates
  self.launch();cp.cuda.get_current_stream().synchronize()
  self.keep=[];stream=cp.cuda.Stream(non_blocking=True)
  with stream:
   stream.begin_capture();self.launch();self.graph=stream.end_capture()
  stream.synchronize()
 def retain(self,x):self.keep.append(x);return x
 def evaluate(self,w,z):
  p=self.p;n=self.n;q=self.q;ca=p.calcium_port;npn=len(self.active);nca=len(self.cn)
  cp.take(w,self.active,out=z['vc']);cp.add(z['vc'],p.leak_reversal_mV,out=z['vc']);cp.take(w,self.cn,out=z['cc']);cp.add(z['cc'],p.leak_reversal_mV,out=z['cc'])
  channel_kernel(((npn+127)//128,),(128,),(np.int32(npn),z['vc'],self.xbase,np.float64(q),p.gbar_nS,p.reversal_mV,z['x'],z['ionic'],z['jac'],z['ge'],self.flag))
  ca_kernel(((nca+127)//128,),(128,),(np.int32(nca),np.int32(len(ca.parent.power)),z['cc'],self.cbase,np.float64(q),*ca.parameters,np.float64(ca.parent.reversal),np.int32(ca.parent.enabled),z['cg'],z['ci'],z['cj'],z['cf'],z['ce'],self.flag))
  cp.subtract(w,self.vbase,out=z['delta']);cp.multiply(z['delta'],self.shift,out=z['delta'])
  diff_kernel(((n+127)//128,),(128,),(np.int32(n),*p.backend.G,p.backend.gsum,w,z['Gv']));csr_kernel(((n+127)//128,),(128,),(np.int32(n),*p.backend.M,z['delta'],np.int32(16),z['Mv']))
  cp.add(z['Gv'],z['Mv'],out=z['r']);cp.subtract(z['r'],self.current,out=z['r']);call_scatter(self.active,z['ionic'],3,z['r']);call_scatter(self.cn,z['ci'],1,z['r'])
  if len(self.sn):syn_current(((len(self.sn)+127)//128,),(128,),(np.int32(len(self.sn)),self.sn,self.sg,self.se,np.float64(p.leak_reversal_mV),w,z['r']))
  return self.retain(cp.linalg.norm(z['r']))
 def launch(self):
  self.flag.fill(0);cp.copyto(self.v,self.guess);refnorm=self.evaluate(self.reference,self.evals[0]);norms=[self.evaluate(self.v,self.evals[1])];linears=[];signed=[]
  for iteration in range(2):
   z=self.evals[iteration+1];self.diagonal.fill(0);call_scatter(self.active,z['jac'],1,self.diagonal);call_scatter(self.cn,z['cj'],1,self.diagonal);call_scatter(self.sn,self.sg,1,self.diagonal)
   cp.multiply(self.p.backend.C,self.shift,out=self.signed_work);cp.add(self.signed_work,self.diagonal,out=self.signed_work);signed.append(self.retain(cp.min(self.signed_work)))
   cp.negative(z['r'],out=self.tree.rhs);cp.copyto(self.tree.diagonal,self.diagonal);self.tree.graph.launch(stream=cp.cuda.get_current_stream());cp.bitwise_or(self.flag,self.tree.flag,out=self.flag)
   csr_kernel(((self.n+127)//128,),(128,),(np.int32(self.n),*self.A,self.tree.x,np.int32(1000000000),self.linear_product));cp.multiply(self.diagonal,self.tree.x,out=self.linear_diag);cp.add(self.linear_product,self.linear_diag,out=self.linear_residual);cp.add(self.linear_residual,z['r'],out=self.linear_residual);linears.append(self.retain(cp.linalg.norm(self.linear_residual)))
   cp.add(self.v,self.tree.x,out=self.v);norms.append(self.evaluate(self.v,self.evals[iteration+2]))
  ge=self.retain(cp.max(self.evals[3]['ge']));ce=self.retain(cp.max(self.evals[3]['ce']));flag=self.retain(self.flag[0].astype(cp.float64));self.control=cp.stack([refnorm,*norms,ge,ce,*linears,*signed,flag])
 def run(self,vbase,xbase,guess,syn,ca_base,reference,current,rtol,atol,gate_atol):
  for target,value in [(self.vbase,vbase),(self.xbase,xbase),(self.guess,guess),(self.reference,reference),(self.current,current),(self.cbase,ca_base)]:cp.copyto(target,value)
  if syn is not None:cp.copyto(self.sg,syn[1]);cp.copyto(self.se,cp.broadcast_to(syn[2],self.se.shape))
  self.graph.launch(stream=cp.cuda.get_current_stream());control=self.control.get();ref,n0,n1,n2,ge,ce,l0,l1,s0,s1,flag=control;threshold=max(atol,rtol*ref)
  passed=(np.isfinite(control).all() and flag==0 and min(s0,s1)>=0 and n2<=threshold and max(ge,ce)<=gate_atol and max(l0,l1)<=.25*threshold and (n1<n0 or n1<=threshold) and (n2<n1 or n2<=threshold))
  if not passed:return None,{'control':control.tolist(),'threshold':threshold}
  z=self.evals[3];data=(self.v.copy(),z['x'].copy(),z['ionic'].copy(),dict(gates=z['cg'].copy(),current=z['ci'].copy(),jacobian=z['cj'].copy(),frozen_conductance=z['cf'].copy(),gate_error=max(ge,ce)))
  report=dict(accepted=True,threshold_pA=threshold,total_iterations=2,history=[{'iteration':0,'residual_l2_pA':n0},{'iteration':1,'residual_l2_pA':n1},{'iteration':2,'residual_l2_pA':n2,'gate_residual':max(ge,ce)}],speculative_graph=True)
  return data,report

def graph_stage(p,fallback,vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess,*,shift,q,current,rtol,atol,gate_atol):
 key=(shift,b'' if synapse is None else cp.asnumpy(synapse[0]).tobytes(),p.calcium_port.parent.enabled)
 if getattr(p,'_stage_graph_key',None)!=key:p._stage_graph=StageGraph(p,synapse,shift,q);p._stage_graph_key=key
 data,report=p._stage_graph.run(vbase,xbase,guess,synapse,ca_base,guess if threshold_guess is None else threshold_guess,current,rtol,atol,gate_atol)
 stats=getattr(p,'graph_statistics',{'accepted':0,'fallback':0});p.graph_statistics=stats
 if data is not None:stats['accepted']+=1;return data,report
 stats['fallback']+=1
 data,original=fallback(vbase,xbase,guess,synapse,ca_base,ca_update,threshold_guess);original['speculative_graph_fallback']=report;return data,original

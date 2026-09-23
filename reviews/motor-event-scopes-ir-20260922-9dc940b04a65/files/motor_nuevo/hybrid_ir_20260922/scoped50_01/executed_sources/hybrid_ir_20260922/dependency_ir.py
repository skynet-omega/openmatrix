"""Executable event-dependency IR: state jumps, live reads and replaced outputs.

This IR certifies direct RHS continuity, not temporal accuracy or mass solvability.
No neuron names or species-specific policies in this module.
"""
from dataclasses import dataclass
import hashlib,json,numpy as np
@dataclass(frozen=True)
class Read:
 name:str
 inputs:np.ndarray
 outputs:np.ndarray
 pointwise:bool=True

class EventProgram:
 def __init__(self,n,reads,zero_derivative_rows=()):
  if type(n)is not int or n<=0:raise ValueError('Positive state dimension required')
  self.n=n;self.reads=[]
  def indices(x):
   a=np.asarray(x)
   if a.size==0:return np.array([],dtype=np.int64)
   if a.ndim!=1 or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=n):raise ValueError('Invalid dependency index')
   return a.astype(np.int64,copy=True)
  for r in reads:
   i,o=indices(r.inputs),indices(r.outputs)
   if r.pointwise and len(i)!=len(o):raise ValueError('Pointwise mapping has unequal sizes')
   self.reads.append(Read(r.name,i,o,r.pointwise))
  self.zero=indices(zero_derivative_rows)
 def classify(self,jump_rows):
  # A direct state read affects an output unless the full derivative is declared zero.
  rows=np.asarray(jump_rows)
  if rows.size and (rows.ndim!=1 or rows.dtype.kind not in 'iu' or np.any(rows<0) or np.any(rows>=self.n)):raise ValueError('Invalid jump index')
  jump=np.zeros(self.n,dtype=bool);jump[rows.astype(np.int64)]=True
  affected=np.zeros(self.n,dtype=bool);reasons=[]
  for r in self.reads:
   mask=jump[r.inputs]
   out=r.outputs[mask] if r.pointwise else (r.outputs if mask.any() else np.array([],dtype=np.int64))
   live=np.setdiff1d(out,self.zero);affected[live]=True
   if len(live):reasons.append({'reader':r.name,'live_outputs':live.tolist()})
  return {'continuous_free_rhs':not bool(affected.any()),'affected_rows':np.flatnonzero(affected).tolist(),'reasons':reasons,'zero_rows':len(self.zero),'meaning':'Direct jump dependence only; controller must still estimate temporal error.'}
 def identity(self):
  d={'n':self.n,'zero':self.zero.tolist(),'reads':[{'name':r.name,'inputs':r.inputs.tolist(),'outputs':r.outputs.tolist(),'pointwise':r.pointwise} for r in self.reads]}
  return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()

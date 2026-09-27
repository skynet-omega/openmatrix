"""UNIT REPAIR CANDIDATE: not used in any recorded51 CNS run; requires a new qualification.
Original frozen pilot_owners.py remains byte-identical.
"""
"""Explicit external air port and normalized generic-law experimental owner."""
from pathlib import Path
import sys,types
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'aporte_motor'))
from air_interface import AirInterface

def need(x,m):
 if not x:raise ValueError(m)

class AirOwner:
 def __init__(self,run,setting,prefix):
  import cupy as cp
  import mujoco
  self.cp=cp;self.mj=mujoco;self.run=run;self.setting=setting;self.prefix=prefix
  self.api=AirInterface();h=run.obj.core.hybrid
  need(np.array_equal(h.brain.node_ids[self.api.rows],self.api.ids),'JO IDs')
  self.device=cp.zeros(h.brain.n_neurons,cp.float64);self.last=np.zeros(len(self.api.ids));self.seen=cp.zeros(len(self.api.ids),cp.float64);self.rows=cp.asarray(self.api.rows)
  self.original=h.coefficients_gpu;self.origin_rotation=self.rotation();self.field=self.origin_rotation@np.array([0.,100.*setting['air'],0.])
  self.original_consume=run.stimulus.consume;self.origin=run.stimulus.k
  self.kinematics=[];self.calls=cp.zeros(1,cp.uint64)
  # A consumer-side witness from the sum passed into the historical operator.
  self.witness=cp.RawKernel(r'''extern "C" __global__ void witness(int n,const long long* rows,const double* before,const double* after,const double* extra,double* seen,unsigned long long* calls){int j=threadIdx.x+blockIdx.x*blockDim.x;if(j<n){long long r=rows[j];seen[j]=after[r]-before[r];if(j==0)atomicAdd(calls,1ULL);}}''','witness')
  def coefficient(owner,state,drive,light):
   used=drive+self.device
   self.witness(((len(self.api.ids)+127)//128,),(128,),(np.int32(len(self.api.ids)),self.rows,drive,used,self.device,self.seen,self.calls))
   return self.original(state,used,light)
  h.coefficients_gpu=types.MethodType(coefficient,h)
  def consume(owner,k):
   j=k-self.origin;need(k==owner.k+1,'Input clock')
   owner.current=owner.spec['baseline']+(owner.spec['delta']['profile'] if setting['odor'] and j>prefix else 0.)
   owner.device_rates.set(np.ascontiguousarray(owner.current));owner.errors.fill(0);owner.calls.fill(0);owner.k=k
   self.last=np.zeros(len(self.api.ids))
   rot=self.rotation();velocity=self.velocity_mm_s()
   if j>prefix:self.last=self.api.encode(air_velocity_world_mm_s=self.field,body_velocity_world_mm_s=velocity,body_to_world=rot)
   full=np.zeros(h.brain.n_neurons);full[self.api.rows]=self.last;self.device.set(full);self.calls.fill(0)
   self.kinematics.append(np.r_[self.field,velocity,rot.ravel()]);cp.cuda.get_current_stream().synchronize();return owner.current.copy()
  run.stimulus.consume=types.MethodType(consume,run.stimulus)
 def velocity_mm_s(self):
  # This body uses centimetres internally; all air-interface inputs are mm/s.
  return 10.0*np.asarray(self.run.obj.body.data.qvel[:3],dtype=np.float64).copy()
 def rotation(self):
  mat=np.empty(9);self.mj.mju_quat2Mat(mat,np.asarray(self.run.obj.body.data.qpos[3:7]));return mat.reshape(3,3)
 def check(self):
  need(int(self.calls.get()[0])>0,'Air input never consumed')
  # JO external native drive is exactly zero in this preparation. If that changes,
  # subtraction may round and this exact check must expose it, not silently relax.
  need(np.array_equal(self.seen.get(),self.last),'Actual JO additive drive differs')
 def close(self):
  self.run.obj.core.hybrid.coefficients_gpu=self.original;self.run.stimulus.consume=self.original_consume

def installer(mode):
 def install(brain,stimulus):
  import cupy as cp
  import observer,fp32_operator
  ob,restore_ob=observer.install(brain,stimulus);old=fp32_operator.FastCSR
  ob.candidate=None
  if mode=='parent':return ob,restore_ob
  class Candidate(old):
   def __init__(self,b,**kw):
    super().__init__(b,**kw);n=b.brain.n_neurons;self.q0=cp.asarray(b.state[:n].copy());self.scales=cp.full(n,-1.,cp.float64);self.ge0=cp.zeros(n,cp.float64);self.low=cp.full(n,np.inf);self.high=cp.full(n,-np.inf)
    kernel=cp.RawKernel((H/'coefficient51.cu').read_text(),'coefficient_fast',options=('--std=c++17','--fmad=false','--prec-div=true','--prec-sqrt=true'))
    def call(grid,block,args):kernel(grid,block,(*args,ob.rows_gpu,ob.csr,np.int32(1 if mode=='conductance' else 2),self.q0,self.scales,self.ge0,self.low,self.high))
    self.kernel=call;ob.candidate=self
  fp32_operator.FastCSR=Candidate
  def undo():fp32_operator.FastCSR=old;restore_ob()
  return ob,undo
 return install

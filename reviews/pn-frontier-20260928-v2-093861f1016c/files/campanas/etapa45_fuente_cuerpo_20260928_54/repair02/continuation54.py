"""Spatial prescribed ORN input and fixed neural motor reader. Engineering probe."""
from pathlib import Path
import sys,types,math
import numpy as np
H=Path(__file__).resolve().parent;C48=H.parents[1]/'etapa45_composicion_20260927_48'
sys.path.insert(0,str(C48))
from protocol import decode

def need(x,m):
 if not x:raise ValueError(m)

def source_geometry(antennae,qpos):
 a=np.asarray(antennae,float);q=np.asarray(qpos,float)
 b=float(np.linalg.norm(a[0,:2]-a[1,:2]));need(b>0,'separation')
 left=(a[0,:2]-a[1,:2])/b;forward=np.array([left[1],-left[0]])
 w,x,y,z=q[3:7];heading=np.array([1-2*(y*y+z*z),2*(w*z+x*y)])
 if forward@heading<0:forward=-forward
 mid=a[:,:2].mean(0)
 return dict(sources_mm=np.array([mid+.5*b*forward+sign*2*b*left for sign in (1,-1)]),sigma_mm=2*b,separation_mm=b)

def nominal_spatial(spec,concentration):
 c=np.asarray(concentration,np.float64);need(c.shape==(2,) and np.isfinite(c).all() and np.all((c>=0)&(c<=1)),'concentration0..1')
 side=(np.asarray(spec['sides'])=='R').astype(int)
 need(np.isin(spec['sides'],['L','R']).all(),'anatomical side')
 return spec['baseline']+spec['delta']['profile']*c[side]

class SpatialOwner:
 def __init__(self,run,arm,prefix,uniform=False):
  self.run=run;self.arm=arm;self.prefix=prefix;self.uniform=uniform;self.origin=run.stimulus.k
  need(arm in ['L','R','none'],'source condition')
  self.boundary=run.obj.core.world.boundary.base
  need(type(self.boundary).__name__=='AntennalBoundary','single legacy zero wrapper')
  self.initial_qpos=run.obj.body.data.qpos.copy()
  a=self.boundary.sample(run.obj.body.data,[0.,0.],1.)['antennae_mm'];self.geometry=source_geometry(a,self.initial_qpos)
  self.source=self.geometry['sources_mm'][0 if arm!='R' else 1].copy();self.sigma=self.geometry['sigma_mm']
  self.original=run.stimulus.consume;self.samples=[];self.used=[];self.start_qpos=[]
  def consume(owner,k):
   self.original(k) # preserves AirOwner and its 1ms body-relative sampling.
   j=k-self.origin;geom=self.boundary.sample(run.obj.body.data,self.source,self.sigma)
   c=np.ones(2) if uniform else (geom['concentration'] if arm!='none' and j>prefix else np.zeros(2))
   owner.current=nominal_spatial(owner.spec,c);need(np.all(owner.current<=owner.caps),'no hidden cap')
   owner.device_rates.set(np.ascontiguousarray(owner.current));owner.errors.fill(0);owner.calls.fill(0)
   self.samples.append(geom['concentration'].copy());self.used.append(c.copy());self.start_qpos.append(run.obj.body.data.qpos.copy())
   owner.cp.cuda.get_current_stream().synchronize();return owner.current.copy()
  run.stimulus.consume=types.MethodType(consume,run.stimulus)
 def close(self):self.run.stimulus.consume=self.original
 def state(self):
  return dict(schema='spatial_ORN54_v1',source_side=self.arm,prefix_ms=self.prefix,uniform_qualification=self.uniform,origin_k=self.origin,source_mm=self.source.tolist(),sigma_mm=self.sigma,geometry={k:np.asarray(v).tolist() for k,v in self.geometry.items()},consumed_ms=len(self.used),last_consumed_concentration=self.used[-1].tolist() if self.used else [0.,0.],mapping='nominal_Hz=baseline+profile_increment*c[anatomical_side]; uncalibrated linear peripheral proxy',legacy_world_sensors_zero=True,source_bearing_not_sent_to_motor=True)

def install_motor(run,apply_yaw):
 m=run.motor;obj=run.obj;previous=obj.body.advance
 def advance(owner,torque_native,nsteps=1):
  need(nsteps==1 and obj.command_mode=='neural','motor schedule')
  if owner.body_calls%owner.substeps==0:
   owner.trial_step+=1;owner.raw_forward,owner.forward,owner.raw_yaw=decode(obj.last_dn,obj.dn_baseline)
  owner.body_calls+=1;requested=obj.requested.copy();obj.command_mode='device'
  delta=obj.last_dn-obj.dn_baseline
  steering=float(np.tanh(250.*(delta[2]-delta[3]))) if apply_yaw else 0.
  obj.requested=np.array([owner.forward,steering])
  try:return owner.original_advance(torque_native,nsteps)
  finally:obj.command_mode='neural';obj.requested=requested
 obj.body.advance=types.MethodType(advance,m)
 return lambda:setattr(obj.body,'advance',previous)

class Intervals:
 def __init__(self,run,apply_yaw):
  self.previous=run.auditor;self.apply_yaw=apply_yaw
 def before(self,obj,k):return self.previous.before(obj,k)
 def after(self,obj,row,k,used):
  a=self.previous;end=a.origin_ns+k*1_000_000
  need(all(int(row[n])==end for n in ['CNS_time_ns','PN_time_ns','body_time_ns']),'clock')
  need(np.array_equal(row['sensores_usados'],used) and np.array_equal(used,np.zeros(3)),'legacy port double input')
  need(np.array_equal(row['sensores_pendientes'],np.zeros(3)),'legacy pending')
  need(np.array_equal(row['DN_q_usada'],a.previous_dn) and np.array_equal(row['DN_baseline'],a.baseline),'motor latency/baseline')
  raw,fwd,yaw=decode(a.previous_dn,a.baseline)
  need(row['command_forward_mm_s']==fwd and row['command_yaw_rate_rad_s']==(yaw if self.apply_yaw else 0.),'applied neural command')
  need(row['forward_unclipped_mm_s']==raw and row['neural_yaw_raw_rad_s']==yaw,'raw reader')
  need(a.motor.trial_step==k and a.motor.body_calls==40*k,'physical count')
  a.pending=np.zeros(3);a.previous_dn=row['DN_q_actual'].copy()

def step(run,spatial):
 import cupy as cp
 k=run.stimulus.k+1;used=run.auditor.before(run.obj,k);run.stimulus.consume(k);run.observer.current_ms=k
 run.obj.step();cp.cuda.runtime.deviceSynchronize()
 row=run.d.captura(run.obj,run.ports,('cola_OFF' if spatial.uniform else 'spatial54'),k,used,run.yaw0,cp);run.stimulus.observed()
 row.update(forward_unclipped_mm_s=run.motor.raw_forward,neural_yaw_raw_rad_s=run.motor.raw_yaw,
  spatial_concentration_used=spatial.used[-1].copy(),spatial_geometric_concentration=spatial.samples[-1].copy(),spatial_sample_qpos=spatial.start_qpos[-1].copy())
 run.auditor.after(run.obj,row,k,used);return row

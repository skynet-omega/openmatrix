"""Adapter from the conserved complete CNS operator to the native runtime.

Specialized membrane/event solvers retain ownership. Captured sensor and PN
buffers are refreshed at every declared exchange boundary, never held forever.
"""
import time
from pathlib import Path
import numpy as np
import cupy as cp
from gpu_coefficient_buffers import CoefficientBuffers
from graph_core import NativeGraph
from event_ports import FilterPorts

class OrganismAdapter:
 def __init__(self,brain,events,event_boundaries=False):
  self.brain=brain;self.events=events;self.core=None;self.report={'epochs':0,'accepted':0,'rejected':0,'build_s':0.,'device_bytes':0,'event_capacity_per_cell_per_epoch':8}
  self.host_read=brain._online_source.general_transmission
  self.event_boundaries=event_boundaries;self.report['mandatory_event_boundaries']=event_boundaries
 def build(self,drive,light):
  b=self.brain;source=b._online_source;old_reader=source.general_transmission
  old_buffers=getattr(b,'_coefficient_buffers',None);old_statistics=dict(b.statistics)
  self.boundary=b._boundary_cuda['light'];self.held=b._orn_pn_cuda['held_rate']
  def freeze(core):
   self.drive=cp.asarray(drive);self.light=cp.asarray(light);self.pn=cp.asarray(self.host_read())
   self.ports=FilterPorts(self.events.rows,b.transmission_start+self.events.rows)
   self.ports.update(self.events.active)
   source.general_transmission=lambda:self.pn
   self.buffers=CoefficientBuffers(len(b.state));b._coefficient_buffers=self.buffers
  def coefficient(z):
   a,r=b.coefficients_gpu(z,self.drive,self.light)
   a[self.ports.qr]=z[self.ports.qr];a[self.ports.sr]=z[self.ports.sr]
   r[self.ports.qr]=0.;r[self.ports.sr]=0.
   return a,r
  try:
   library=Path(__file__).resolve().parents[2]/'motor_nuevo/native_hybrid_20260922/libgraph_control_v2.so' if self.event_boundaries else None
   self.core=NativeGraph(b.state,coefficient,rtol=b.parameters['rtol'],atol=b.parameters['atol'],norm_size=b._norm_size(),project=lambda z,c,f:self.ports.project(z,c,f),freeze=freeze,native_library=library)
  finally:
   source.general_transmission=old_reader
   if old_buffers is None:del b._coefficient_buffers
   else:b._coefficient_buffers=old_buffers
   b.statistics.clear();b.statistics.update(old_statistics)
  self.report.update(build_s=self.core.build_s,device_bytes=self.core.pool.total_bytes())
 def step(self,b,ns,drive,light):
  if self.events.active is None:raise RuntimeError('Missing physical event owner')
  drive,light=b._validated_inputs(ns,drive,light)
  if b._general_buffer_active or not b._edge_buffer_active:raise RuntimeError('Invalid operator ownership')
  if self.core is None:self.build(drive,light)
  g=self.core
  cp.cuda.get_current_stream().synchronize()
  with g.stream:
   for dst,src in ((g.x,b.state),(self.drive,drive),(self.light,light),(self.pn,self.host_read()),(self.boundary,b.held_boundary_light),(self.held,b.held_afferent_rate_hz)):
    if dst.shape!=np.shape(src):raise ValueError('Captured boundary layout changed')
    dst.set(np.ascontiguousarray(src))
   self.ports.update(self.events.active)
  p=b.parameters
  boundaries=np.asarray(self.events.active.times,dtype=np.float64) if self.event_boundaries else None
  nxt,counts,error=g.advance(ns,b.next_step_ns,p['minimum_step_ns'],p['maximum_step_ns'],boundaries=boundaries)
  b.state=g.x.get(stream=g.stream);b.time_ns+=ns;b.next_step_ns=nxt
  b.statistics['accepted']+=counts[0];b.statistics['rejected']+=counts[1];b.statistics['evaluations']+=6*(counts[0]+counts[1])
  b.statistics['minimum_accepted_ns']=min(b.statistics['minimum_accepted_ns'],counts[2])
  b.statistics['maximum_accepted_error']=max(b.statistics['maximum_accepted_error'],error)
  self.events.report['global_trials']+=counts[0]+counts[1]
  self.report['epochs']+=1;self.report['accepted']+=counts[0];self.report['rejected']+=counts[1]
  b.publish_rates()
 def close(self):
  if self.core:self.core.close()

def install(brain,events,event_boundaries=False):
 adapter=OrganismAdapter(brain,events,event_boundaries);old=events.step;events.step=adapter.step
 def restore():events.step=old;adapter.close()
 return adapter,restore

"""Legacy model adapter declares SET for its already-saturated LIF events.

Other producers retain their existing ADD events. This is not a guarantee that
all future source types export a numerically self-consistent history.
"""
import numpy as np
from reset_ports import ResetFilterPorts
from lif_event_metadata import lif_record_with_post

def prepare():
 import event_coupling,organism_adapter
 original_wave=event_coupling.Waveform;original_ports=organism_adapter.FilterPorts
 class Ledger(original_wave):
  def __init__(self,*args,**kw):super().__init__(*args,**kw);self.post_values=[]
  def add(self,times,rows,jumps,post_values=None):
   before=len(self.times);super().add(times,rows,jumps)
   if post_values is None:self.post_values.extend([None]*(len(self.times)-before))
   else:
    values=np.asarray(post_values,dtype=float);keep=np.asarray(jumps)>0
    if values.shape!=np.asarray(jumps).shape or not np.isfinite(values).all():raise ValueError('Reset metadata')
    self.post_values.extend(values[keep].tolist())
 event_coupling.Waveform=Ledger;organism_adapter.FilterPorts=ResetFilterPorts
 def restore():event_coupling.Waveform=original_wave;organism_adapter.FilterPorts=original_ports
 return restore

def attach(session):
 import kc_spatial_brain as spatial
 old=spatial.lif_events;events=session.events
 def record(*args,**kw):
  if events.active is None:return old(*args,**kw)
  clipped,t,j,posts=lif_record_with_post(*args,**kw);rr,cc=np.where(np.isfinite(t))
  events.active.add(t[rr,cc],events.other[rr].astype(np.int64),j[rr,cc],post_values=posts[rr,cc])
  return clipped
 spatial.lif_events=record
 def restore():spatial.lif_events=old
 return restore

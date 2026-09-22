"""Explicit bridge to the conserved organism; exactly one owner per process.

Both profiles retain mandatory event boundaries. This adapter is not a claim
that the inherited class-level installation scheme is a generic engine API.
"""
from pathlib import Path
import sys,threading
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
sys.path[:0]=[str(ROOT/'campanas/etapa3_motor_nuevo_20260922'),str(H.parent/'causal_runtime_20260922'),str(H.parent/'native_hybrid_20260922')]
PROFILES=('causal_cuda','reference_cuda')
_lock=threading.Lock()

class RuntimeSession:
 def __init__(self,brain,profile):
  if profile not in PROFILES:raise ValueError('Explicit supported engine required')
  if getattr(brain,'_operator_restore_invalid',False) or getattr(brain,'_native_rebuild_required',False):raise RuntimeError('Invalidated owner cannot start a runtime')
  b=brain._spatial_batch
  while hasattr(b,'base'):b=b.base
  if getattr(b,'_native_publication_invalid',False):raise RuntimeError('Invalidated physical publisher cannot start a runtime')
  if not _lock.acquire(blocking=False):raise RuntimeError('Legacy adapter permits one runtime per process')
  self.brain=brain;self.profile=profile;self.undo=[];self.closed=False;self.cell=None;self.events=None;self.adapter=None;self.pn=None;self.partition=None
  try:
   import kc_adaptive,event_coupling,block_midpoint,pn_execution,organism_adapter,rollback_guard
   self.undo.append(kc_adaptive.install(brain))
   self.events,u=event_coupling.install(brain);self.undo.append(u)
   self.adapter,u=organism_adapter.install(brain,self.events,event_boundaries=True);self.undo.append(u)
   self.pn,u=pn_execution.install(brain);self.undo.append(u)
   if profile=='causal_cuda':
    import device_cell
    self.cell,u=device_cell.install(brain,self.events)
   else:
    import native_cell
    self.cell,u=native_cell.install(brain,self.events,compressed=False)
   self.undo.append(u)
   self.partition,u=block_midpoint.install(brain,125000);self.undo.append(u)
   self.undo.append(rollback_guard.install(brain))
  except BaseException:
   self.close();raise
 def report(self):
  return {'profile':self.profile,'event_boundaries':True,'geometry_compression':False,'coupling_ns':125000,'cell':None if self.cell is None else self.cell.report,'CNS':None if self.adapter is None else self.adapter.report,'PN':self.pn,'events':None if self.events is None else self.events.report,'partition':self.partition}
 def close(self):
  if self.closed:return
  self.closed=True;errors=[]
  try:
   for undo in reversed(self.undo):
    try:undo()
    except BaseException as exc:errors.append(repr(exc))
  finally:_lock.release()
  if errors:raise RuntimeError('Runtime teardown failed: '+str(errors))

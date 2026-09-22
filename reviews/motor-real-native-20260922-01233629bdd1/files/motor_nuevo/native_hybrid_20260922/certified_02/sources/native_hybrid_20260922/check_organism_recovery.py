"""Actual full-brain rollback and explicit reconstruction of execution owners."""
from pathlib import Path
import sys,time,copy,json,gc
import numpy as np
HERE=Path(__file__).resolve().parent;OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix/work');P=HERE.parents[1]/'campanas/etapa3_motor_nuevo_20260922'
sys.path[:0]=[str(P),str(OLD/'motor14_20260922'),str(OLD/'motor13_20260922')]
from motor_runtime import load
import kc_adaptive,event_coupling,block_midpoint,pn_execution,organism_adapter,native_cell,rollback_guard
from verification_vendor.compare import flatten

def same(a,b):
 aa,bb=dict(flatten(a)),dict(flatten(b));changed=[]
 for k in set(aa)|set(bb):
  if k not in aa or k not in bb:changed.append(k);continue
  x,y=aa[k],bb[k]
  eq=(isinstance(y,np.ndarray) and x.shape==y.shape and x.dtype==y.dtype and x.tobytes()==y.tobytes()) if isinstance(x,np.ndarray) else type(x) is type(y) and x==y
  if not eq:changed.append(k)
 return sorted(changed)

def install(h):
 undo=[kc_adaptive.install(h)];events,u=event_coupling.install(h);undo.append(u);adapter,u=organism_adapter.install(h,events);undo.append(u)
 _,u=pn_execution.install(h);undo.append(u);_,u=native_cell.install(h,events,compressed=True);undo.append(u)
 _,u=block_midpoint.install(h,125000);undo.append(u);undo.append(rollback_guard.install(h));return events,adapter,undo

out=HERE/'recovery_01';out.mkdir(exist_ok=False);result={};t=time.perf_counter();expected=None
for fault in (False,True):
 obj,*_=load(out/('fault' if fault else 'control'));h=obj.core.hybrid;events,adapter,undo=install(h)
 try:
  if fault:
   before={};old_advance=h.advance
   def observed(*args,**kw):
    before['state']=copy.deepcopy(h.state_dict());return old_advance(*args,**kw)
   h.advance=observed;original=events.step
   def inject(*args,**kw):
    original(*args,**kw)
    if adapter.core.steps<1:raise RuntimeError('Fault fixture did not accept a native step')
    raise RuntimeError('INJECT_AFTER_ACCEPT')
   events.step=inject
   try:obj.step()
   except RuntimeError as e:
    if str(e)!='INJECT_AFTER_ACCEPT':raise
   else:raise RuntimeError('Fault not triggered')
   changes=same(before['state'],h.state_dict());result['rollback_changed_paths']=changes
   if changes:raise RuntimeError('Physical rollback differs: '+str(changes))
   try:h.advance(1000000,np.zeros(h.brain.n_neurons),np.zeros(len(h.photo_ids)))
   except RuntimeError as e:
    if 'rebuild native runtime' not in str(e):raise
    result['stale_runtime_rejected']=True
   else:raise RuntimeError('Stale runtime silently reused')
   for fn in reversed(undo):fn()
   events,adapter,undo=install(h)
  obj.step();state=h.state_dict();body={'qpos':obj.body.data.qpos.copy(),'qvel':obj.body.data.qvel.copy(),'pending':obj.core.pending_sensors.copy()}
  if not fault:expected=(copy.deepcopy(state),body)
  else:
   changes=same(expected[0],state);bodychanges=same(expected[1],body);result.update(retry_state_changed_paths=changes,retry_body_changed_paths=bodychanges)
   if changes or bodychanges:raise RuntimeError('Explicit reconstruction continuation differs')
 finally:
  for fn in reversed(undo):fn()
  obj.close()
 del obj,h,events,adapter,undo;gc.collect()
result.update(status='PASS',wall_s=time.perf_counter()-t,scope='First CNS block failure after accepted GPU epoch, full physical state restored, runtime reconstructed, next 1ms equals no-failure control. Not automatic recovery from every partial body interval.')
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

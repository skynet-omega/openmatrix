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

out=HERE/(sys.argv[1] if len(sys.argv)>1 else 'recovery_01');out.mkdir(exist_ok=False);result={};t=time.perf_counter();expected=None
for fault in (False,True):
 obj,*_=load(out/('fault' if fault else 'control'));h=obj.core.hybrid;events,adapter,undo=install(h)
 try:
  before={};old_advance=h.advance;body_before=obj.body.data.qpos.copy()
  def observed(*args,**kw):
   before['state']=copy.deepcopy(h.state_dict());before['args']=copy.deepcopy(args);before['kw']=copy.deepcopy(kw)
   value=old_advance(*args,**kw);before['after']=copy.deepcopy(h.state_dict());return value
  h.advance=observed
  if fault:
   original=events.step
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
  if not fault:
   obj.step();expected=before['after']
  else:
   try:obj.step()
   except ValueError as exc:
    if 'Failed or unsupported' not in str(exc):raise
    result['coupled_session_reuse_rejected']=True
   else:raise RuntimeError('Failed coupled session silently reused')
   h.advance(*before['args'],**before['kw'])
   changes=same(expected,h.state_dict());result['neural_retry_changed_paths']=changes
   from session_io import write_state
   write_state(out/'control_neural',expected);write_state(out/'retry_neural',h.state_dict())
   aa,bb=dict(flatten(expected)),dict(flatten(h.state_dict()));differences=[];material=[]
   for path in changes:
    x,y=aa.get(path),bb.get(path)
    numeric=isinstance(x,np.ndarray) and x.dtype.kind=='f' and isinstance(y,np.ndarray) and x.shape==y.shape and x.dtype==y.dtype
    scalar=type(x) is float and type(y) is float
    maximum=float(np.max(abs(x-y),initial=0)) if numeric else (abs(x-y) if scalar else None)
    exact_required=('manifest' in path or 'migration' in path or not (numeric or scalar))
    within=(not exact_required) and bool(np.allclose(x,y,rtol=1e-11,atol=1e-10))
    differences.append(dict(path=path,max_abs=maximum,within_predeclared_execution_bound=within))
    if not within:material.append(path)
   result['continuation_differences']=differences
   (out/'DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
   if material:raise RuntimeError('Explicit neural continuation differs materially: '+str(material))
   if not np.array_equal(obj.body.data.qpos,body_before):raise RuntimeError('Body advanced despite neural failure')
   result['body_unchanged_after_failed_step']=True
 finally:
  for fn in reversed(undo):fn()
  obj.close()
 del obj,h,events,adapter,undo;gc.collect()
result.update(status='PASS',wall_s=time.perf_counter()-t,scope='First CNS block failure after accepted GPU epoch: complete neural physical state restored; explicit runtime reconstruction and neural 1ms retry tested against predeclared execution bound; exact differences retained. Failed coupled session refuses reuse; body remains unadvanced. Automatic whole-organism resume is not implemented.')
(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

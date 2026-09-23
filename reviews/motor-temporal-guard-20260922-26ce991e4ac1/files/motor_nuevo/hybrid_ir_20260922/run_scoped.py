"""Bounded full-organism candidates; reuse the corrected runner's load contract."""
from pathlib import Path
import sys,json,argparse,time,types,traceback,resource,hashlib,shutil
import numpy as np
R=Path(__file__).resolve().parent;M=R.parent/'macro_abc_20260922';sys.path.insert(0,str(M));T=R.parent/'transient_localization_20260922';ROOT=R.parents[1];H=R.parent/'pipeline_review_20260922'
sys.path.insert(0,str(T))
sys.path.insert(0,str(H));import run_pipeline
from motor_runtime import load
from run_storage import RunStorage,atomic_json
from runtime_session import RuntimeSession

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['guard','fine1562'],required=True);p.add_argument('--ms',type=int,choices=[20,50],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['baseline','scoped'],required=True);a=p.parse_args()
 RunStorage(a.out);obj=session=None;undo_metadata=undo_ports=undo_graph=None;start=time.perf_counter();rows=[];times=[];status='INCOMPLETE';error=None
 try:
  import cupy as cp,pandas as pd
  obj,d,plan,Field,field,ports=load(a.out/'inputs');b=obj.core.hybrid
  # Preserve125us coupling; only the prospective scoped arm may remove global cuts.
  import reset_adapter
  undo_ports=reset_adapter.prepare()
  session=RuntimeSession(b,'causal_cuda');undo_metadata=reset_adapter.attach(session)
  import scoped_graph
  undo_graph=scoped_graph.install(b,session.events,a.mode,a.out/'EVENT_CONTRACT.json')
  from operator_inventory import inventory
  atomic_json(a.out/'OPERATOR_INVENTORY.json',inventory(b,session.events))
  if a.variant.startswith('guard'):
   import event_guard
   event_guard.attach(session)
  else:
   base=b._spatial_batch
   while hasattr(base,'base'):base=base.base
   old=base.advance
   def limited(self,ns,ge,gi,**kw):kw['inner_step_ns']=1562;return old(ns,ge,gi,**kw)
   base.advance=types.MethodType(limited,base)
  from session_io import write_state
  from operator_state import OperatorState,LEGACY_BINDINGS
  atomic_json(a.out/'OPERATOR.json',{'identity':OperatorState(b,LEGACY_BINDINGS).state_dict()['identity_sha256']})
  frozen=a.out/'executed_sources';frozen.mkdir()
  for root in (R,M,T,H,T.parent/'causal_runtime_20260922',T.parent/'native_hybrid_20260922',ROOT/'campanas/etapa3_motor_nuevo_20260922'):
   for f in root.iterdir():
    if f.suffix in ('.py','.cu','.cpp','.hpp'):
     dest=frozen/root.name/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
  atomic_json(a.out/'FROZEN.json',{str(f.relative_to(frozen)):hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen.rglob('*') if f.is_file()})
  table=pd.read_parquet(run_pipeline.OLD/'data/male_v10/nodes.parquet',columns=['type']);selected=np.flatnonzero(table['type'].fillna('').astype(str).str.match(r'^(DNa02|DNb05|PFL3|hDeltaK|PFG)(?:$|_)').to_numpy())
  yaw=d.yaw_grados(obj.body.data.qpos);origin=b.time_ns
  for i in range(a.ms):
   if time.perf_counter()-start>275:raise TimeoutError('Process budget')
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('RAM budget')
   used=obj.core.pending_sensors.copy();cp.cuda.get_current_stream().synchronize();t=time.perf_counter();obj.step();cp.cuda.get_current_stream().synchronize();advance=time.perf_counter()-t
   t=time.perf_counter();row=d.captura(obj,ports,'ensayo',i+1,used,yaw,cp);row['central_q']=b.release()[selected].copy();row['central_transmission']=b.state[b.transmission_start+selected].copy();cp.cuda.get_current_stream().synchronize();capture=time.perf_counter()-t
   rows.append(row);record={'step':i+1,'advance_s':advance,'observation_s':capture};times.append(record)
   with (a.out/'TIMES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   if i in (0,4,19,49):write_state(a.out/f'brain_{i+1:02d}ms',b.state_dict())
   print(json.dumps(record),flush=True)
  if b.time_ns!=origin+a.ms*1000000:raise ValueError('Clock mismatch')
  write_state(a.out/'brain_final',b.state_dict())
  np.savez_compressed(a.out/'body_final.npz',**{k:getattr(obj.body.data,k).copy() for k in ('qpos','qvel','qacc','qacc_warmstart','act','ctrl')},pending_sensors=obj.core.pending_sensors.copy(),pending_excitation=obj.core.pending_excitation.copy())
  np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([v[k] for v in rows]) for k in rows[0]})
  status='COMPLETE'
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'numerical_diagnostic':getattr(exc,'numerical_diagnostic',None)};raise
 finally:
  atomic_json(a.out/'RESULT.json',{'status':status,'error':error,'variant':a.variant,'mode':a.mode,'effective_event_policy':None if session is None or session.adapter.core is None else {'scoped_epochs':session.adapter.core.scoped_epochs,'global_cuts':session.adapter.core.global_cuts},'ms':len(rows),'requested_ms':a.ms,'wall_s':time.perf_counter()-start,'times':times,'runtime':None if session is None else session.report(),'stage3_admission':False})
  if undo_graph:undo_graph()
  if undo_metadata:undo_metadata()
  if session:session.close()
  if undo_ports:undo_ports()
  if obj:obj.close()

if __name__=='__main__':main()

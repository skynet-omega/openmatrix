"""One instrumented repeat of the known domain failure; no tolerance change."""
from pathlib import Path
import sys,time,types,json,traceback,hashlib,resource,shutil
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parents[1];H=R.parent/'pipeline_review_20260922'
sys.path.insert(0,str(H));import run_pipeline
from motor_runtime import load
from runtime_session import RuntimeSession
from run_storage import RunStorage,atomic_json
from diagnostic_graph import DiagnosticGraph

def main():
 import argparse,cupy as cp
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();RunStorage(a.out)
 start=time.perf_counter();obj=session=None;status='INCOMPLETE';error=None;times=[];boundaries=[]
 try:
  import organism_adapter
  organism_adapter.NativeGraph=DiagnosticGraph
  obj,d,plan,Field,field,ports=load(a.out/'inputs');b=obj.core.hybrid;session=RuntimeSession(b,'causal_cuda')
  from session_io import write_state
  from operator_state import OperatorState,LEGACY_BINDINGS
  atomic_json(a.out/'OPERATOR.json',{'identity':OperatorState(b,LEGACY_BINDINGS).state_dict()['identity_sha256']})
  frozen=a.out/'executed_sources';frozen.mkdir()
  sources=[Path(__file__).resolve(),R/'diagnostic_graph.py',ROOT/'campanas/etapa3_motor_nuevo_20260922/graph_core.py',H/'runtime_session.py']
  for f in sources:shutil.copy2(f,frozen/f.name)
  atomic_json(a.out/'FROZEN.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
  base=b._spatial_batch
  while hasattr(base,'base'):base=base.base
  old=base.advance
  def limited(self,ns,ge,gi,**kw):kw['inner_step_ns']=1562;return old(ns,ge,gi,**kw)
  base.advance=types.MethodType(limited,base)
  event=session.events;old_step=event.step
  def step(brain,ns,drive,light):
   w=event.active;record={'index':len(boundaries),'phase':'accepted' if ns==125000 else 'predictor','ns':int(ns),'clock_start_ns':int(brain.time_ns),'events':len(w.times)}
   try:
    result=old_step(brain,ns,drive,light);record['success']=True;return result
   except RuntimeError as exc:
    record['success']=False;core=session.adapter.core
    if hasattr(core,'last_failure'):
     arrays=core.failure_arrays();arrays.update(q0=np.asarray(w.q).copy(),s0=np.asarray(w.s).copy(),tau=np.asarray(w.tau).copy(),ts=np.asarray(w.ts),times=np.asarray(w.times,dtype=float),rows=np.asarray(w.rows,dtype=np.int64),jumps=np.asarray(w.jumps,dtype=float),event_rows=np.asarray(event.rows),port_qr=session.adapter.ports.qr.get(),port_sr=session.adapter.ports.sr.get(),drive=np.asarray(drive),light=np.asarray(light))
     np.savez_compressed(a.out/'failure_arrays.npz',**arrays)
     atomic_json(a.out/'FAILURE.json',{'boundary':record,'diagnostic':core.last_failure,'graph_class':type(core).__name__})
     # During a failed exchange, CNS and source clocks can intentionally differ.
     # A secondary checkpoint error must never replace the original exception.
     try:write_state(a.out/'failure_brain',brain.state_dict())
     except BaseException as snapshot_error:
      atomic_json(a.out/'CHECKPOINT_ERROR.json',{'type':type(snapshot_error).__name__,'message':str(snapshot_error),'full_checkpoint_available':False})
    raise
   finally:boundaries.append(record)
  event.step=step
  for i in range(20):
   if time.perf_counter()-start>215:raise TimeoutError('Diagnostic process budget')
   if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('RAM budget')
   t=time.perf_counter();obj.step();cp.cuda.get_current_stream().synchronize();record={'ms':i+1,'advance_s':time.perf_counter()-t};times.append(record);print(json.dumps(record),flush=True)
   if i in (0,4):write_state(a.out/f'brain_{i+1:02d}ms',b.state_dict())
  status='COMPLETE_NO_FAILURE'
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc),'diagnostic':getattr(exc,'numerical_diagnostic',None),'traceback':traceback.format_exc()};status='FAILURE_CAPTURED' if (a.out/'FAILURE.json').exists() else 'INCOMPLETE';print(json.dumps({'status':status,'error':error['message'],'diagnostic':error['diagnostic']}),flush=True)
 finally:
  atomic_json(a.out/'BOUNDARIES.json',boundaries)
  atomic_json(a.out/'RESULT.json',{'status':status,'error':error,'completed_ms':len(times),'wall_s':time.perf_counter()-start,'times':times,'runtime':None if session is None else session.report(),'stage3_admission':False})
  if session:session.close()
  if obj:obj.close()
if __name__=='__main__':main()

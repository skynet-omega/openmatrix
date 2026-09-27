"""One finite factorial arm from the same qualified scientific state."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,argparse,json,time,resource,signal,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];C49=H.parent/'etapa45_operands_20260927_49'
sys.path.insert(0,str(C49/'aporte_motor'))
from resume49 import build,save,sha,need,C48
from temporal_input import ARMS,install

def main():
 p=argparse.ArgumentParser();p.add_argument('arm',choices=ARMS);a=p.parse_args();out=H/a.arm;need(not out.exists(),'Immutable arm')
 plan=json.loads((H/'PLAN.json').read_text())
 for path,digest in plan['sources'].items():need(sha(path)==digest,'Changed frozen source '+path)
 wall=time.monotonic();cpu=time.process_time();run=None;undo=None
 status=dict(schema='temporal50_arm_v1',arm=a.arm,status='STARTED',attempted_ms=0,committed_ms=0,plan_sha256=sha(H/'PLAN.json'))
 def stop(*_):raise TimeoutError('Finite arm budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(600);resource.setrlimit(resource.RLIMIT_CPU,(500,510))
 try:
  run=build(Path(plan['source_checkpoint']),out)
  meta,undo=install(run.stimulus,a.arm);save(out/'TEMPORAL_OWNER.json',meta)
  from source_inventory import imported,verify
  current=imported();lock=json.loads((C48/'SOURCES.json').read_text())
  need(all(value==lock[name] for name,value in current.items() if name in lock),'Changed historical execution source')
  need(all(name in lock or Path(name).is_relative_to(H) or Path(name).is_relative_to(C49/'aporte_motor') for name in current),'Unexpected runtime source')
  save(out/'EXECUTED_SOURCES.json',current)
  print(json.dumps(dict(status='RESTORED_EXACT',arm=a.arm,wall_s=time.monotonic()-wall)),flush=True)
  rows=[];inputs=[];props=[];six=[]
  ids=np.array([10045,10056,10118,10065,523769,10360],np.int64);indices=np.searchsorted(run.obj.core.brain.node_ids,ids);need(np.array_equal(run.obj.core.brain.node_ids[indices],ids),'DN identities')
  for j in range(1,141):
   status['attempted_ms']+=1;save(out/'STATUS.json',status)
   props.append(run.obj.core.pending_proprioception['normalized_afferent_drive'].copy())
   row=run.step();rows.append(row);inputs.append(run.stimulus.current.copy());q=run.obj.core.hybrid.release()[indices].copy();six.append(q)
   need(np.array_equal(q[:4],row['DN_q_actual']),'DN observer alias')
   status['committed_ms']+=1
   if j==10:
    with np.load(C49/'sham_off_01/traces.npz',allow_pickle=False) as ref:
     checks={k:np.array_equal(np.array([x[k] for x in rows]),ref[k][:10]) for k in ref.files}
    need(all(checks.values()),'Common prefix differs from49');save(out/'PREFIX49.json',checks)
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<24*1024**3,'RAM limit')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<14*1024**3,'VRAM limit')
   if j%20==0:print(json.dumps(dict(status='STEP',arm=a.arm,ms=j,wall_s=time.monotonic()-wall)),flush=True)
  np.savez_compressed(out/'traces.npz',**{k:np.asarray([x[k] for x in rows]) for k in rows[0]})
  np.savez_compressed(out/'input_and_observers.npz',nominal_Hz=np.array(inputs),proprioception=np.array(props),DN_ids=ids,DN_q=np.array(six),ORN_ids=run.stimulus.spec['ids'],ORN_sides=run.stimulus.spec['sides'])
  run.observer.flush(out/'dng100_observed.npz');run.save(out/'final_state')
  # Explicit schema distinguishes a new stimulus experiment from a49 OFF-tail.
  endpoint=out/'final_state';save(endpoint/'temporal_owner.json',dict(**meta,consumed_ms=run.stimulus.k))
  m=json.loads((endpoint/'MANIFEST.json').read_text());m['parent_schema']=m['schema'];m['schema']='temporal50_scientific_state_v1';m['files']['temporal_owner.json']=sha(endpoint/'temporal_owner.json');save(endpoint/'MANIFEST.json',m)
  save(out/'EVENTS.json',run.session.events.audit);verify(current)
  status.update(status='COMPLETE',observer=run.observer.report(),runtime=run.session.report(),initial_exact=run.initial['exact'])
 except BaseException as exc:status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  if undo is not None:undo()
  if run is not None:
   try:run.close()
   except BaseException as exc:status.update(status='FAILED',cleanup_error=repr(exc))
  out.mkdir(exist_ok=True,parents=True);status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0)
  print(json.dumps({k:status[k] for k in ['status','arm','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

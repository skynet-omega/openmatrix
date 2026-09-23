"""Auditable stage3 runner; smoke is engineering only, not a stage3 experiment."""
from pathlib import Path
import argparse,sys,time,json,traceback,resource,signal,shutil,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
sys.path[:0]=[str(OLD/'work/motor14_20260922'),str(OLD/'work/motor13_20260922')]
from run_storage import RunStorage,atomic_json

def main(argv=None,*,loader=None):
 if not __debug__:raise RuntimeError('Unchanged historical loaders require normal Python')
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--odor',choices=['sham','uniform','odor_left','odor_right'],required=True)
 p.add_argument('--engine',choices=['causal_cuda','reference_cuda'],required=True);p.add_argument('--ms',type=int,default=1000);p.add_argument('--smoke-test',action='store_true');a=p.parse_args(argv)
 if a.smoke_test:
  if a.ms not in (1,2) or a.odor!='sham':raise ValueError('Smoke is sham1–2ms, no preparation')
 elif a.ms not in (100,1000):raise ValueError('Use the declared stage3 horizons100/1000ms')
 storage=RunStorage(a.out)  # Outside try/finally: never touch a pre-existing run.
 start=time.perf_counter();obj=session=None;rows=[];times=[];status='STARTING';error=None;checkpoint={'status':'NOT_ATTEMPTED'};cleanup=[];phase='loading'
 old_handler=signal.getsignal(signal.SIGTERM)
 def terminated(signum,frame):raise InterruptedError('Termination requested')
 signal.signal(signal.SIGTERM,terminated)
 def save_traces():
  if rows:np.savez_compressed(a.out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
 def snapshot(folder):
  from session_io import write_state
  from operator_state import OperatorState,LEGACY_BINDINGS
  h=obj.core.hybrid
  if getattr(obj.core,'failed',False) or getattr(h,'_operator_restore_invalid',False) or getattr(h,'_native_rebuild_required',False):raise RuntimeError('Failed owner cannot publish a coherent checkpoint')
  write_state(folder/'session',obj.core.state_dict());write_state(folder/'prosthesis',obj.state())
  write_state(folder/'published',{'rates':obj.core.brain.rates,'time_ns':obj.core.brain.time_ns,'rng':obj.core.brain.rng.bit_generator.state})
  write_state(folder/'effective_operator',OperatorState(h,LEGACY_BINDINGS).state_dict())
  atomic_json(folder/'boundary.json',obj.core.world.boundary.metadata())
 try:
  if loader is None:
   from motor_runtime import load
   loader=load
  obj,d,plan,Field,field,ports=loader(a.out/'preparation_inputs');h=obj.core.hybrid;initial=h.time_ns
  from runtime_session import RuntimeSession
  session=RuntimeSession(h,a.engine)
  frozen=a.out/'executed_sources';frozen.mkdir()
  roots=(HERE,HERE.parent/'causal_runtime_20260922',ROOT/'campanas/etapa3_motor_nuevo_20260922',HERE.parent/'native_hybrid_20260922')
  for base in roots:
   for f in base.iterdir():
    if f.is_file() and f.suffix in ('.py','.cu','.cpp','.hpp'):
     dest=frozen/base.name/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
  atomic_json(a.out/'FROZEN.json',{str(f.relative_to(frozen)):hashlib.sha256(f.read_bytes()).hexdigest() for f in frozen.rglob('*') if f.is_file()})
  import cupy as cp,pandas as pd
  table=pd.read_parquet(OLD/'data/male_v10/nodes.parquet',columns=['bodyId','type','somaSide']);types=table['type'].fillna('').astype(str)
  selected=np.flatnonzero(types.str.match(r'^(DNa02|DNb05|PFL3|hDeltaK|PFG)(?:$|_)').to_numpy())
  atomic_json(a.out/'CENTRAL_ROWS.json',{'rows':selected.tolist(),'ids':table.bodyId.to_numpy()[selected].tolist(),'types':types.to_numpy()[selected].tolist()})
  prep=0 if a.smoke_test else 40
  atomic_json(a.out/'RUN_CONTRACT.json',{'condition':a.odor,'smoke_only':a.smoke_test,'engine':a.engine,'event_boundaries':True,'geometry_compression':False,'preparation_ms':prep,'trial_ms':a.ms,'initial_clock_ns':initial,'checkpoint':str(OLD/plan['checkpoint']),'checkpoint_manifest_sha256':hashlib.sha256((OLD/plan['checkpoint']/'manifest.json').read_bytes()).hexdigest(),'interventions':str(a.out/'preparation_inputs/INTERVENCIONES.json'),'decoder_gain':plan['decoder_gain'],'budget_s':240 if a.smoke_test else 1200,'stage3_admission':False})
  yaw0=d.yaw_grados(obj.body.data.qpos)
  for phase,duration in [('preparacion',prep),('ensayo',a.ms)]:
   if phase=='ensayo':
    if not a.smoke_test:storage.snapshot('prepared_state',snapshot)
    d.instalar_campo(obj,Field,field,a.odor,0.);yaw0=d.yaw_grados(obj.body.data.qpos)
   for k in range(duration):
    if time.perf_counter()-start>(190 if a.smoke_test else 1150):raise TimeoutError('Process budget; preserving results')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2>18:raise MemoryError('RAM budget exceeded')
    used=obj.core.pending_sensors.copy();t=time.perf_counter();obj.step();cp.cuda.runtime.deviceSynchronize()
    row=d.captura(obj,ports,phase,k+1,used,yaw0,cp);row['central_q']=h.release()[selected].copy();row['central_transmission']=h.state[h.transmission_start+selected].copy();rows.append(row)
    record={'phase':phase,'step':k+1,'clock_ns':int(h.time_ns),'wall_s':time.perf_counter()-t};times.append(record)
    with (a.out/'TIMES.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if phase=='preparacion' and np.any(row['sensores_usados']!=0):raise ValueError('Preparation consumed odor')
    if phase=='ensayo' and k+1==100:storage.snapshot('state_100ms',snapshot)
    if (k+1)%10==0:save_traces();print(json.dumps(record),flush=True)
  if h.time_ns!=initial+(prep+a.ms)*1000000:raise ValueError('End clock mismatch')
  status='COMPLETE'
  from session_io import write_state
  write_state(a.out/'brain_final',h.state_dict())
  np.savez_compressed(a.out/'body_final.npz',qpos=obj.body.data.qpos,qvel=obj.body.data.qvel,qacc=obj.body.data.qacc,qacc_warmstart=obj.body.data.qacc_warmstart,act=obj.body.data.act,ctrl=obj.body.data.ctrl,pending_sensors=obj.core.pending_sensors,pending_excitation=obj.core.pending_excitation)
 except BaseException as exc:
  error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc(),'numerical_diagnostic':getattr(exc,'numerical_diagnostic',None)};status='INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'INCOMPLETE'
 finally:
  try:save_traces()
  except BaseException as exc:cleanup.append('traces: '+repr(exc));status='INCOMPLETE'
  if obj is not None:
   try:checkpoint=storage.snapshot('final_state',snapshot)
   except BaseException as exc:
    checkpoint={'status':'FAILED_NOT_RESUMABLE','type':type(exc).__name__,'message':str(exc)}
    atomic_json(a.out/'CHECKPOINT_ERROR.json',checkpoint)
    if status=='COMPLETE':status='INCOMPLETE_CHECKPOINT'
  report=None if session is None else session.report()
  for closer in ([session.close] if session else [])+([obj.close] if obj else []):
   try:closer()
   except BaseException as exc:cleanup.append(repr(exc))
  if cleanup and status=='COMPLETE':status='INCOMPLETE_CLEANUP'
  signal.signal(signal.SIGTERM,old_handler)
  trial=[r for r in rows if r['fase']=='ensayo']
  atomic_json(a.out/'RESULT.json',{'status':status,'phase_at_stop':phase,'error':error,'cleanup_errors':cleanup,'engine':a.engine,'smoke_only':a.smoke_test,'odor':a.odor,'requested_trial_ms':a.ms,'completed_trial_ms':len(trial),'completed_preparation_ms':len(rows)-len(trial),'wall_total_s':time.perf_counter()-start,'step_wall_s':sum(t['wall_s'] for t in times),'checkpoint':checkpoint,'runtime':report,'stage3_pass':None,'reason':'No biological admission. Smoke validates wiring only; snapshots have no automatic full-organism resume qualification.'})
 return 0 if status=='COMPLETE' else 2

if __name__=='__main__':raise SystemExit(main())

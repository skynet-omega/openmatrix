"""Bounded read-only observation of natural PN output in the qualified spatial setup."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,argparse,signal,resource,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
C55=H.parent/'etapa45_transferencia_causal_20260928_55';C54=H.parent/'etapa45_fuente_cuerpo_20260928_54';C52=H.parent/'etapa45_reparacion_observada_20260927_52';C49=H.parent/'etapa45_operands_20260927_49'
sys.path[:0]=[str(H),str(C55),str(C54/'repair02'),str(C52),str(C49/'aporte_motor')]
from resume49 import build,sha,need
from continuation54 import SpatialOwner,install_motor,Intervals,step
from pilot52_owners import AirOwner
from observer52 import Panel,witness
import pn_probe55,observe57

def save(path,value):
 path=Path(path);temporary=path.with_name(path.name+'.tmp')
 with temporary.open('w',encoding='utf-8') as f:
  json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 temporary.replace(path)
def snapshot(run,air):
 a=run.auditor;run.auditor=a.previous
 try:return witness(run,air)
 finally:run.auditor=a
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');args=ap.parse_args();plan=json.loads((H/'PLAN.json').read_text());spec=plan['arms'][args.arm];out=H/args.arm
 need(not out.exists(),'immutable arm');wall=time.monotonic();cpu=time.process_time();run=air=spatial=None;undo_motor=None;rows=[];consume=[];orn_consume=[];neural=[];rates=[];orn_audit=[];clock=[]
 status=dict(status='STARTED',arm=args.arm,attempted_ms=0,committed_ms=0,contract_sha256=sha(H/'PLAN.json'))
 cap=plan['qualification'] if spec['duration_ms']==2 else plan['per_arm']
 def stop(*_):raise TimeoutError('finite natural-signal budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(cap['wall_s']);resource.setrlimit(resource.RLIMIT_CPU,(cap['CPU_s'],cap['CPU_s']+5))
 try:
  for path,d in json.loads((H/'SOURCES.json').read_text()).items():need(sha(path)==d,'source changed '+path)
  source=R/plan['source_checkpoint'];run=build(source,out,observer_installer=observe57.installer)
  # The inherited loader still qualifies the same3000ms state within its3200ms
  # cold-load horizon. Only the new peripheral duration guard extends prospectively;
  # no constructor predicate, solver, time value or timestep is changed.
  initial_body_time=float(run.obj.body.data.time);expected_body_time=initial_body_time;initial_body_steps=int(run.obj.body.steps)
  if spec['duration_ms']>200:
   old_limit=run.stimulus.plan['duration_ms'];need(old_limit==3200,'inherited duration')
   run.stimulus.plan['duration_ms']=3000+spec['duration_ms']
   need(run.obj.core.world.boundary.plan is run.stimulus.plan,'shared duration owner')
   save(out/'HORIZON_EXTENSION.json',dict(old_duration_ms=old_limit,new_duration_ms=3000+spec['duration_ms'],cold_loaded_ms=3000,clock_values_changed=False,dt_changed=False,scope='Peripheral interval guard only; exact40 additions of25us checked at every committed millisecond. No portable final-state restart qualification.'))
  air=AirOwner(run,dict(mode='parent',air=0,odor=True),plan['prefix_ms']);spatial=SpatialOwner(run,spec['source'],plan['prefix_ms']);undo_motor=install_motor(run,True);run.auditor=Intervals(run,True)
  from source_inventory import imported,verify
  sources=imported();save(out/'EXECUTED_SOURCES.json',sources)
  h=run.obj.core.hybrid;need(run.session.adapter.core is None,'configure before captured graph')
  with np.load(H/'reference/TERMINALS.npz',allow_pickle=False) as z:pn_rows=z['rows'].copy();pn_ids=z['ids'].copy()
  need(np.array_equal(h.brain.node_ids[pn_rows],pn_ids),'PN support')
  panel=Panel(h.brain.node_ids,True,run.stimulus.spec['ids']);initial_q=h.release().copy()
  save(out/'SCIENTIFIC_INITIAL.json',snapshot(run,air));save(out/'SPATIAL_OWNER_INITIAL.json',spatial.state())
  print(json.dumps(dict(status='RESTORED',arm=args.arm,wall_s=time.monotonic()-wall)),flush=True)
  reference=np.load(H/'reference'/('parent_'+spec['source'])/'traces.npz',allow_pickle=False) if spec['duration_ms']>2 else None
  for j in range(1,spec['duration_ms']+1):
   status['attempted_ms']+=1;save(out/'STATUS.json',status);observe57.reset(run)
   row=step(run,spatial);air.check();rows.append(row);consume.append(pn_probe55.sample(run));orn_consume.append(observe57.sample(run));rates.append(run.stimulus.current.copy())
   for _ in range(40):expected_body_time+=2.5e-5
   need(float(run.obj.body.data.time)==expected_body_time and int(run.obj.body.steps)==initial_body_steps+40*j,'exact body clock recurrence')
   clock.append((float(run.obj.body.data.time),int(run.obj.body.steps)))
   orn_audit.append(np.array([int(run.stimulus.calls.get()[0]),int(run.stimulus.errors.get()[0])],dtype=np.uint64))
   q=h.release();neural.append(q[pn_rows].copy());panel.record(q,row['CNS_time_ns']);status['committed_ms']+=1
   if reference is not None and j<=89:
    for k in reference.files:need(np.array_equal(row[k],reference[k][j-1]),'54 prefix changed '+k+' at '+str(j))
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<plan['RAM_bytes'],'RAM')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<plan['VRAM_bytes'],'VRAM')
   if j%32==0 or j==spec['duration_ms']:print(json.dumps(dict(status='STEP',arm=args.arm,ms=j,wall_s=time.monotonic()-wall)),flush=True)
  if reference is not None:reference.close()
  panel.flush(out/'wide_observation.npz');save(out/'SCIENTIFIC_WITNESS.json',snapshot(run,air));save(out/'EVENTS.json',run.session.events.audit);save(out/'SPATIAL_OWNER_FINAL.json',spatial.state())
  np.savez_compressed(out/'neural_and_inputs.npz',initial_q=initial_q,final_q=h.release(),PN_ids=pn_ids,PN_rows=pn_rows,PN_q=neural,nominal_ORN_Hz=rates,ORN_ids=run.stimulus.spec['ids'],ORN_sides=run.stimulus.spec['sides'],baseline_Hz=run.stimulus.spec['baseline'],delta_profile_Hz=run.stimulus.spec['delta']['profile'],source_geometry=spatial.geometry['sources_mm'],sigma_mm=spatial.sigma)
  np.savez_compressed(out/'ORN_AUDIT.npz',counts_and_errors=orn_audit)
  if spec['duration_ms']>2:
   a=run.auditor;run.auditor=a.previous
   try:run.save(out/'final_state')
   finally:run.auditor=a
   save(out/'final_state/spatial57.json',dict(schema='natural_PN57_v1',spec=spec,source_checkpoint=str(source),spatial=spatial.state(),air_field_world=air.field.tolist(),reader='protocol48 unchanged',neural_law='parent',observation_only=True,resume_GPU_qualified=False))
   manifest=json.loads((out/'final_state/MANIFEST.json').read_text());manifest.update(schema='natural57_scientific_state_v1',parent_schema=manifest['schema']);manifest['files']['spatial57.json']=sha(out/'final_state/spatial57.json');save(out/'final_state/MANIFEST.json',manifest)
  status.update(status='COMPLETE',initial_exact=run.initial['exact'],observer=run.observer.report(),runtime=run.session.report());verify(sources)
 except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  out.mkdir(exist_ok=True,parents=True)
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
  if consume:np.savez_compressed(out/'PN_consumed.npz',**{k:np.asarray([r[k] for r in consume]) for k in consume[0]})
  if orn_consume:np.savez_compressed(out/'ORN_generic_consumed.npz',**{k:np.asarray([r[k] for r in orn_consume]) for k in orn_consume[0]})
  if clock:np.savez_compressed(out/'CLOCK_BODY.npz',initial_time_s=initial_body_time,initial_steps=np.int64(initial_body_steps),time_s=np.asarray(clock)[:,0],steps=np.asarray(clock)[:,1].astype(np.int64))
  if run is not None and run.observer.epochs:run.observer.flush(out/'dng100_observed.npz')
  for fn in [None if spatial is None else spatial.close,undo_motor,None if air is None else air.close,None if run is None else run.close]:
   if fn is not None:
    try:fn()
    except BaseException as e:status.update(status='FAILED',cleanup_error=repr(e))
  status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0);print(json.dumps({k:status[k] for k in ['status','arm','attempted_ms','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

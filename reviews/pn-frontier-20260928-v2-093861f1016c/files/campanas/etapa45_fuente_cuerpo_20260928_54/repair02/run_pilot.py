"""One bounded live source/body arm; preserve scientific state and failures."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,argparse,json,time,resource,signal,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2];C52=H.parents[1]/'etapa45_reparacion_observada_20260927_52';C49=H.parents[1]/'etapa45_operands_20260927_49'
sys.path[:0]=[str(H),str(C52),str(C49/'aporte_motor')]
from resume49 import build,save,sha,need,C48
from pilot52_owners import AirOwner,installer
from observer52 import Panel,witness
from continuation54 import SpatialOwner,install_motor,Intervals,step

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');ap.add_argument('--qualification',action='store_true');a=ap.parse_args();p=json.loads((H/'PILOT_PLAN.json').read_text());q=a.qualification
 spec=p['qualifications'][a.arm] if q else p['arms'][a.arm];setting={'mode':spec['mode'],'air':spec['air'] if q else 0,'odor':True}
 out=H/(('qual_' if q else '')+a.arm);duration=p['qualification_ms'] if q else p['duration_ms'];prefix=0 if q else p['prefix_ms']
 need(not out.exists(),'immutable arm')
 for path,digest in json.loads((H/'SOURCES.json').read_text()).items():need(sha(path)==digest,'frozen source '+path)
 wall=time.monotonic();cpu=time.process_time();run=air=spatial=None;undo_motor=None;rows=[];rates=[];status={'status':'STARTED','arm':a.arm,'qualification':q,'attempted_ms':0,'committed_ms':0,'plan_sha256':sha(H/'PILOT_PLAN.json')}
 b=p['budgets'];wallcap=b['per_qualification_wall_s' if q else 'per_arm_wall_s'];cpucap=b['per_qualification_CPU_s' if q else 'per_arm_CPU_s']
 def stop(*_):raise TimeoutError('finite arm budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(wallcap);resource.setrlimit(resource.RLIMIT_CPU,(cpucap,cpucap+5))
 try:
  run=build(R/p['source_checkpoint'],out,observer_installer=installer(setting['mode']));air=AirOwner(run,setting,prefix)
  from source_inventory import imported,verify
  sources=imported();historic=json.loads((C48/'SOURCES.json').read_text());need(all(v==historic[k] for k,v in sources.items() if k in historic),'historical source changed');save(out/'EXECUTED_SOURCES.json',sources)
  spatial=SpatialOwner(run,'L' if q else spec['source'],prefix,uniform=q);undo_motor=install_motor(run,not q);run.auditor=Intervals(run,not q)
  panel=Panel(run.obj.core.brain.node_ids,True,run.stimulus.spec['ids']);initial_q=run.obj.core.hybrid.release().copy()
  save(out/'SPATIAL_OWNER_INITIAL.json',spatial.state());print(json.dumps({'status':'RESTORED_EXACT','arm':a.arm,'wall_s':time.monotonic()-wall}),flush=True)
  for j in range(1,duration+1):
   status['attempted_ms']+=1;save(out/'STATUS.json',status);row=step(run,spatial);air.check();rows.append(row);rates.append(run.stimulus.current.copy());panel.record(run.obj.core.hybrid.release(),row['CNS_time_ns']);status['committed_ms']+=1
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<b['RAM_bytes'],'RAM')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<b['VRAM_bytes'],'VRAM')
   if q:
    with np.load(C52/('qual_'+spec['reference'])/'traces.npz') as ref:
     for key in ref.files:
      newkey='neural_yaw_raw_rad_s' if key=='neural_yaw_unapplied_rad_s' else key
      need(np.array_equal(row[newkey],ref[key][j-1]),'qualification differs '+key+' at '+str(j))
   if j%15==0 or j==duration:print(json.dumps({'status':'STEP','arm':a.arm,'ms':j,'wall_s':time.monotonic()-wall}),flush=True)
  panel.flush(out/'wide_observation.npz');save(out/'SPATIAL_OWNER_FINAL.json',spatial.state())
  np.savez_compressed(out/'neural_and_inputs.npz',initial_q=initial_q,final_q=run.obj.core.hybrid.release(),ORN_ids=run.stimulus.spec['ids'],ORN_sides=run.stimulus.spec['sides'],baseline_Hz=run.stimulus.spec['baseline'],delta_profile_Hz=run.stimulus.spec['delta']['profile'],nominal_Hz=rates,source_geometry=spatial.geometry['sources_mm'],sigma_mm=spatial.sigma)
  candidate=run.observer.candidate
  if candidate is not None:np.savez_compressed(out/'candidate_owner.npz',q0=candidate.q0.get(),S=candidate.scales.get(),gE0=candidate.ge0.get(),min_raw=candidate.low.get(),max_raw=candidate.high.get())
  if not q:
   wrapped=run.auditor;run.auditor=wrapped.previous
   try:run.save(out/'final_state')
   finally:run.auditor=wrapped
   final=out/'final_state';save(final/'closed_loop_owner.json',dict(schema='source_body54_owner_v1',spatial=spatial.state(),air_field_world=air.field.tolist(),applied_yaw=True,reader='protocol48 unchanged',legacy_world_zero=True,mode=setting['mode'],resume_GPU_qualified=False))
   if candidate is not None:
    import shutil;shutil.copy2(out/'candidate_owner.npz',final/'candidate_owner.npz')
   manifest=json.loads((final/'MANIFEST.json').read_text());manifest['parent_schema']=manifest['schema'];manifest['schema']='source_body54_scientific_state_v1';manifest['files']['closed_loop_owner.json']=sha(final/'closed_loop_owner.json')
   if candidate is not None:manifest['files']['candidate_owner.npz']=sha(final/'candidate_owner.npz')
   save(final/'MANIFEST.json',manifest)
  save(out/'EVENTS.json',run.session.events.audit);status.update(status='COMPLETE',initial_exact=run.initial['exact'],observer=run.observer.report(),runtime=run.session.report());verify(sources)
 except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  out.mkdir(exist_ok=True,parents=True)
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
  if run is not None and run.observer.epochs:run.observer.flush(out/'dng100_observed.npz')
  for action in [None if spatial is None else spatial.close,undo_motor,None if air is None else air.close,None if run is None else run.close]:
   if action is not None:
    try:action()
    except BaseException as e:status.update(status='FAILED',cleanup_error=repr(e))
  status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0);print(json.dumps({k:status[k] for k in ['status','arm','attempted_ms','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

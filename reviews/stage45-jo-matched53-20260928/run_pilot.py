"""One prospectively bounded arm, using the qualified48 restoration."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,argparse,json,time,resource,signal,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];C49=H.parent/'etapa45_operands_20260927_49'
sys.path.insert(0,str(C49/'aporte_motor'))
from resume49 import build,save,sha,need,C48
from jo_matched import AirOwner,installer,PARENT
from observer52 import Panel,witness

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');ap.add_argument('--qualification',action='store_true');args=ap.parse_args();p=json.loads((H/'PILOT_PLAN.json').read_text())
 if args.qualification:
  need(args.arm in p['qualifications'],'Known qualification');spec=p['qualifications'][args.arm]
  out=H/('qual_'+args.arm);duration=spec['duration_ms'];prefix=spec['prefix_ms'];wide=spec['wide']
  setting={k:spec[k] for k in ['mode','air','odor']}
 else:
  need(args.arm in p['arms'],'Known arm');out=H/args.arm;duration=p['duration_ms'];prefix=p['prefix_ms'];wide=True;setting=p['arms'][args.arm]
 need(not out.exists(),'Immutable arm');lock=json.loads((H/'PILOT_SOURCES.json').read_text())
 for path,digest in lock.items():need(sha(path)==digest,'Frozen source changed '+path)
 wall=time.monotonic();cpu=time.process_time();run=air=None;rows=[];inputs=[];air_inputs=[];neural=[];panel=None;status=dict(status='STARTED',arm=args.arm,qualification=args.qualification,wide_observer=wide,attempted_ms=0,committed_ms=0,setting=setting,plan_sha256=sha(H/'PILOT_PLAN.json'))
 def stop(*_):raise TimeoutError('Finite pilot arm budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);armwall=p['budgets']['per_qualification_wall_s' if args.qualification else 'per_arm_wall_s'];armcpu=p['budgets']['per_qualification_CPU_s' if args.qualification else 'per_arm_CPU_s'];signal.alarm(armwall);resource.setrlimit(resource.RLIMIT_CPU,(armcpu,armcpu+5))
 try:
  run=build(R/p['source_checkpoint'],out,observer_installer=installer);air=AirOwner(run,setting,prefix,matched=not args.qualification)
  from source_inventory import imported,verify
  current=imported();historic=json.loads((C48/'SOURCES.json').read_text());need(all(v==historic[k] for k,v in current.items() if k in historic),'Historical source changed');save(out/'EXECUTED_SOURCES.json',current)
  ids=np.array([10045,10056,10118,10065,10442,10760,523769,10360,10888,11067,11074,512006,11960,11702,523640,10371],np.int64);idx=np.searchsorted(run.obj.core.brain.node_ids,ids);need(np.array_equal(run.obj.core.brain.node_ids[idx],ids),'Observe IDs')
  panel=Panel(run.obj.core.brain.node_ids,wide,run.stimulus.spec['ids']);initial_velocity=run.obj.body.data.qvel[:3].copy();initial_qpos=run.obj.body.data.qpos.copy();initial_q=run.obj.core.hybrid.release().copy();print(json.dumps(dict(status='RESTORED_EXACT',arm=args.arm,wall_s=time.monotonic()-wall)),flush=True)
  for j in range(1,duration+1):
   status['attempted_ms']+=1;save(out/'STATUS.json',status);row=run.step();air.check();rows.append(row);inputs.append(run.stimulus.current.copy());air_inputs.append(air.last.copy());committed_release=run.obj.core.hybrid.release();neural.append(committed_release[idx].copy());panel.record(committed_release,row['CNS_time_ns']);status['committed_ms']+=1
   need(np.array_equal(neural[-1][:4],row['DN_q_actual']),'DN native observer')
   if args.qualification and args.arm=='unused_legacy_identity':
    with np.load(C49/'sham_off_01/traces.npz') as ref:need(all(np.array_equal(row[k],ref[k][j-1]) for k in ref.files),'Identity parity with49')
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<p['budgets']['RAM_bytes'],'RAM budget')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<p['budgets']['VRAM_bytes'],'VRAM budget')
   if j%15==0 or j==duration:print(json.dumps(dict(status='STEP',arm=args.arm,ms=j,wall_s=time.monotonic()-wall)),flush=True)
  final_q=run.obj.core.hybrid.release();np.savez_compressed(out/'neural_and_inputs.npz',ids=ids,q=np.array(neural),initial_q=initial_q,final_q=final_q,ORN_ids=run.stimulus.spec['ids'],nominal_Hz=np.array(inputs),JO_ids=air.api.ids,JO_rows=air.api.rows,JO_drive=np.array(air_inputs),air_kinematics=np.array(air.kinematics),initial_native_velocity=initial_velocity,initial_qpos=initial_qpos,JO_raw=np.asarray(air.raw_history),JO_consumed_FP32=np.asarray(air.post_history),JO_kernel_audit=np.asarray(air.call_history,dtype=np.uint64),JO_target_total=np.asarray(air.total),JO_reference_totals=np.asarray(air.reference_totals))
  panel.flush(out/'wide_observation.npz')
  if args.qualification:save(out/'OWNERS_FINAL.json',witness(run,air))
  c=run.observer.candidate
  if c is not None:
   np.savez_compressed(out/'candidate_owner.npz',q0=c.q0.get(),S=c.scales.get(),gE0=c.ge0.get(),min_raw=c.low.get(),max_raw=c.high.get())
  if not args.qualification:
   run.save(out/'final_state');d=out/'final_state';save(d/'pilot_owner.json',dict(schema='jo_matched53_owner_v1',setting=setting,completed_ms=duration,field_world=air.field.tolist(),origin_rotation=air.origin_rotation.tolist(),duration_ms=duration,prefix_ms=prefix,body_velocity_conversion=10.0,wide_panel_sha256=sha(PARENT/'PANEL.npz'),JO_target_total=air.total,JO_reference_totals=air.reference_totals,matched=air.matched,postcast_tolerance=1e-7,readout='unchanged forward; yaw observed unapplied'))
   if c is not None:
    import shutil;shutil.copy2(out/'candidate_owner.npz',d/'candidate_owner.npz')
   m=json.loads((d/'MANIFEST.json').read_text());m['parent_schema']=m['schema'];m['schema']='jo_matched53_scientific_state_v1';m['files']['pilot_owner.json']=sha(d/'pilot_owner.json')
   if c is not None:m['files']['candidate_owner.npz']=sha(d/'candidate_owner.npz')
   save(d/'MANIFEST.json',m)
  save(out/'EVENTS.json',run.session.events.audit);status.update(status='COMPLETE',initial_exact=run.initial['exact'],observer=run.observer.report(),runtime=run.session.report());verify(current)
 except BaseException as exc:status.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  out.mkdir(exist_ok=True,parents=True)
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([x[k] for x in rows]) for k in rows[0]})
  if run is not None and run.observer.epochs:run.observer.flush(out/'dng100_observed.npz')
  if air is not None:air.close()
  if run is not None:
   try:run.close()
   except BaseException as exc:status.update(status='FAILED',cleanup_error=repr(exc))
  status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0);print(json.dumps({k:status[k] for k in ['status','arm','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

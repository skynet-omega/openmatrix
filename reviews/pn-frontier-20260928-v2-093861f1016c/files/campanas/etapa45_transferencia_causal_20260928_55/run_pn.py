"""Bounded partial-PN intervention. State belongs to the experimental evaluator."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,argparse,signal,resource,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1];C54=H.parent/'etapa45_fuente_cuerpo_20260928_54';C52=H.parent/'etapa45_reparacion_observada_20260927_52';C49=H.parent/'etapa45_operands_20260927_49'
sys.path[:0]=[str(H/'aporte_motor'),str(H),str(C54/'repair02'),str(C52),str(C49/'aporte_motor')]
from resume49 import build,save,sha,need
from continuation54 import SpatialOwner,install_motor,Intervals,step
from pilot52_owners import AirOwner
from observer52 import Panel,witness
from observations import digest
from pn_boundary55 import apply_between_committed_ms
import pn_probe55

def snapshot(run,air):
 a=run.auditor;run.auditor=a.previous
 try:return witness(run,air)
 finally:run.auditor=a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');args=ap.parse_args();p=json.loads((H/'PN_PLAN.json').read_text());spec=p['arms'][args.arm];out=H/args.arm
 need(not out.exists(),'immutable arm')
 for path,d in json.loads((H/'PN_SOURCES.json').read_text()).items():need(sha(path)==d,'source changed '+path)
 wall=time.monotonic();cpu=time.process_time();run=air=spatial=None;undo_motor=None;rows=[];consume=[];neural=[];rates=[];status=dict(status='STARTED',arm=args.arm,attempted_ms=0,committed_ms=0,contract_sha256=sha(H/'PN_PLAN.json'))
 cap=p['qualification'] if spec['duration_ms']==2 else p['per_arm']
 def stop(*_):raise TimeoutError('finite PN budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(cap['wall_s']);resource.setrlimit(resource.RLIMIT_CPU,(cap['CPU_s'],cap['CPU_s']+5))
 try:
  source=R/'campanas/etapa45_composicion_20260927_48'/spec['receiver']/'final_state'
  run=build(source,out,observer_installer=pn_probe55.installer);air=AirOwner(run,dict(mode='parent',air=0,odor=True),p['prefix_ms'])
  spatial=SpatialOwner(run,'none',p['prefix_ms']);undo_motor=install_motor(run,True);run.auditor=Intervals(run,True)
  from source_inventory import imported,verify
  sources=imported();save(out/'EXECUTED_SOURCES.json',sources)
  meta=json.loads((H/'reference/PN_donors.json').read_text());h=run.obj.core.hybrid
  with np.load(H/'reference/PN_donors.npz') as z:
   selected=z['selected_indices'].copy();pn_rows=z['rows'].copy();pn_ids=z['PN_ids'].copy()
   payload=dict(meta['payload'],pn_ids=pn_ids,selected_indices=selected,values=z[spec['donor']].copy())
  need(run.session.adapter.core is None,'patch before captured CNS graph')
  before=h.state.copy();owner_before=snapshot(run,air);fine_before=digest(h._online_source.state_dict())
  if spec['apply']:surgery=apply_between_committed_ms(run,payload)
  else:surgery=dict(schema='untouched_control',changed_slots=0)
  after=h.state.copy();owner_after=snapshot(run,air);fine_after=digest(h._online_source.state_dict())
  outside=np.ones(len(before),bool);outside[selected]=False
  need(np.array_equal(before[outside],after[outside]),'outside PN state modified')
  need(fine_before==fine_after,'fine PN or input memories changed')
  need(all(owner_before[k]==owner_after[k] for k in owner_before if k not in ['session','published']),'unselected scientific owner changed')
  if spec['receiver']==spec['donor']:need(owner_before==owner_after and np.array_equal(before,after),'identity surgery changed state')
  if spec['apply']:need(np.array_equal(after[selected],payload['values']),'donor values missing')
  np.savez_compressed(out/'INTERVENTION.npz',selected_indices=selected,pn_rows=pn_rows,pn_ids=pn_ids,before_state=before,after_state=after,donor_values=payload['values'])
  save(out/'INTERVENTION.json',dict(spec=spec,operation=surgery,before=owner_before,after=owner_after,fine_before=fine_before,fine_after=fine_after,scope=p['intervention']))
  panel=Panel(h.brain.node_ids,True,run.stimulus.spec['ids']);initial_q=h.release().copy()
  print(json.dumps(dict(status='RESTORED_AND_PATCHED',arm=args.arm,changed=surgery['changed_slots'],wall_s=time.monotonic()-wall)),flush=True)
  for j in range(1,spec['duration_ms']+1):
   status['attempted_ms']+=1;save(out/'STATUS.json',status);pn_probe55.reset(run)
   row=step(run,spatial);air.check();rows.append(row);consume.append(pn_probe55.sample(run));rates.append(run.stimulus.current.copy());q=h.release();neural.append(q[pn_rows].copy());panel.record(q,row['CNS_time_ns']);status['committed_ms']+=1
   if spec['receiver']=='sham' and spec['donor']=='sham' and j<=89:
    with np.load(C54/'repair02/parent_none/traces.npz') as ref:
     for key in ref.files:need(np.array_equal(row[key],ref[key][j-1]),'sham no-op54 differs '+key+' at '+str(j))
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<p['RAM_bytes'],'RAM')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<p['VRAM_bytes'],'VRAM')
   if j%32==0 or j==spec['duration_ms']:print(json.dumps(dict(status='STEP',arm=args.arm,ms=j,wall_s=time.monotonic()-wall)),flush=True)
  panel.flush(out/'wide_observation.npz');save(out/'SCIENTIFIC_WITNESS.json',snapshot(run,air));save(out/'EVENTS.json',run.session.events.audit)
  np.savez_compressed(out/'neural_and_inputs.npz',initial_q=initial_q,final_q=h.release(),PN_ids=pn_ids,PN_q=neural,nominal_ORN_Hz=rates,ORN_ids=run.stimulus.spec['ids'],baseline_Hz=run.stimulus.spec['baseline'])
  if spec['duration_ms']>2:
   a=run.auditor;run.auditor=a.previous
   try:run.save(out/'final_state')
   finally:run.auditor=a
   save(out/'final_state/intervention55.json',dict(schema='partial_PN55_v1',spec=spec,source_checkpoint=str(source),spatial=spatial.state(),air_field_world=air.field.tolist(),resume_GPU_qualified=False))
   manifest=json.loads((out/'final_state/MANIFEST.json').read_text());manifest.update(schema='PN55_scientific_state_v1',parent_schema=manifest['schema']);manifest['files']['intervention55.json']=sha(out/'final_state/intervention55.json');save(out/'final_state/MANIFEST.json',manifest)
  status.update(status='COMPLETE',initial_exact=run.initial['exact'],observer=run.observer.report(),runtime=run.session.report());verify(sources)
 except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  out.mkdir(exist_ok=True,parents=True)
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
  if consume:np.savez_compressed(out/'PN_consumed.npz',**{k:np.asarray([r[k] for r in consume]) for k in consume[0]})
  if run is not None and run.observer.epochs:run.observer.flush(out/'dng100_observed.npz')
  for fn in [None if spatial is None else spatial.close,undo_motor,None if air is None else air.close,None if run is None else run.close]:
   if fn is not None:
    try:fn()
    except BaseException as e:status.update(status='FAILED',cleanup_error=repr(e))
  status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0);print(json.dumps({k:status[k] for k in ['status','arm','attempted_ms','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

"""Finite factorial evaluation of a declared output cut; no neural law change."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,argparse,signal,resource,traceback
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
C55=H.parent/'etapa45_transferencia_causal_20260928_55';C54=H.parent/'etapa45_fuente_cuerpo_20260928_54';C52=H.parent/'etapa45_reparacion_observada_20260927_52';C49=H.parent/'etapa45_operands_20260927_49'
sys.path[:0]=[str(H),str(C55),str(C54/'repair02'),str(C52),str(C49/'aporte_motor')]
from resume49 import build,save,sha,need
from continuation54 import SpatialOwner,install_motor,Intervals,step
from pilot52_owners import AirOwner
from observer52 import Panel,witness
from observations import digest
import pn_probe55,hold_probe56

def snapshot(run,air):
 a=run.auditor;run.auditor=a.previous
 try:return witness(run,air)
 finally:run.auditor=a
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm');args=ap.parse_args();plan=json.loads((H/'A_PLAN.json').read_text());spec=plan['arms'][args.arm];out=H/args.arm
 need(not out.exists(),'immutable arm')
 wall=time.monotonic();cpu=time.process_time();run=air=spatial=None;undo_motor=None;rows=[];consume=[];native=[];neural=[];rates=[];orn_audit=[]
 status=dict(status='STARTED',arm=args.arm,attempted_ms=0,committed_ms=0,contract_sha256=sha(H/'A_PLAN.json'))
 cap=plan['qualification'] if spec['duration_ms']==2 else plan['per_arm']
 def stop(*_):raise TimeoutError('finite terminal budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(cap['wall_s']);resource.setrlimit(resource.RLIMIT_CPU,(cap['CPU_s'],cap['CPU_s']+5))
 try:
  for path,d in json.loads((H/'A_SOURCES.json').read_text()).items():need(sha(path)==d,'source changed '+path)
  source=R/'campanas/etapa45_composicion_20260927_48'/spec['receiver']/'final_state'
  run=build(source,out,observer_installer=hold_probe56.installer(spec));air=AirOwner(run,dict(mode='parent',air=0,odor=True),plan['prefix_ms'])
  spatial=SpatialOwner(run,'none',plan['prefix_ms']);undo_motor=install_motor(run,True);run.auditor=Intervals(run,True)
  from source_inventory import imported,verify
  sources=imported();save(out/'EXECUTED_SOURCES.json',sources)
  h=run.obj.core.hybrid;need(run.session.adapter.core is None,'configure before captured graph')
  with np.load(H/'reference/TERMINALS.npz') as z:pn_rows=z['rows'].copy();pn_ids=z['ids'].copy();value=z[spec['donor']].copy()
  panel=Panel(h.brain.node_ids,True,run.stimulus.spec['ids']);initial_q=h.release().copy();initial_state=h.state.copy();initial_owners=snapshot(run,air)
  np.savez_compressed(out/'INTERVENTION.npz',pn_rows=pn_rows,pn_ids=pn_ids,fixed_FP32=value,initial_state=initial_state)
  save(out/'INTERVENTION.json',dict(spec=spec,initial_owners=initial_owners,scope=plan['intervention'],source_state_untouched=True))
  print(json.dumps(dict(status='RESTORED',arm=args.arm,wall_s=time.monotonic()-wall)),flush=True)
  for j in range(1,spec['duration_ms']+1):
   status['attempted_ms']+=1;save(out/'STATUS.json',status);hold_probe56.reset(run,spec,j)
   row=step(run,spatial);air.check();rows.append(row);consume.append(pn_probe55.sample(run));native.append(hold_probe56.sample(run));rates.append(run.stimulus.current.copy())
   orn_audit.append(np.array([int(run.stimulus.calls.get()[0]),int(run.stimulus.errors.get()[0])],dtype=np.uint64))
   q=h.release();neural.append(q[pn_rows].copy());panel.record(q,row['CNS_time_ns']);status['committed_ms']+=1
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<plan['RAM_bytes'],'RAM')
   import cupy as cp
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<plan['VRAM_bytes'],'VRAM')
   if j%32==0 or j==spec['duration_ms']:print(json.dumps(dict(status='STEP',arm=args.arm,ms=j,wall_s=time.monotonic()-wall)),flush=True)
  panel.flush(out/'wide_observation.npz');save(out/'SCIENTIFIC_WITNESS.json',snapshot(run,air));save(out/'EVENTS.json',run.session.events.audit)
  np.savez_compressed(out/'neural_and_inputs.npz',initial_q=initial_q,final_q=h.release(),PN_ids=pn_ids,PN_q=neural,nominal_ORN_Hz=rates,ORN_ids=run.stimulus.spec['ids'],baseline_Hz=run.stimulus.spec['baseline'])
  np.savez_compressed(out/'ORN_AUDIT.npz',counts_and_errors=orn_audit)
  if spec['duration_ms']>2:
   a=run.auditor;run.auditor=a.previous
   try:run.save(out/'final_state')
   finally:run.auditor=a
   save(out/'final_state/intervention56.json',dict(schema='generic_CSR_terminal56_v1',spec=spec,source_checkpoint=str(source),terminal_sha256=sha(H/'reference/TERMINALS.npz'),elapsed_intervention_ms=spec['duration_ms'],spatial=spatial.state(),air_field_world=air.field.tolist(),resume_GPU_qualified=False))
   manifest=json.loads((out/'final_state/MANIFEST.json').read_text());manifest.update(schema='terminal56_scientific_state_v1',parent_schema=manifest['schema']);manifest['files']['intervention56.json']=sha(out/'final_state/intervention56.json');save(out/'final_state/MANIFEST.json',manifest)
  status.update(status='COMPLETE',initial_exact=run.initial['exact'],observer=run.observer.report(),runtime=run.session.report());verify(sources)
 except BaseException as e:status.update(status='FAILED',error=repr(e),traceback=traceback.format_exc());print(status['traceback'],flush=True)
 finally:
  out.mkdir(exist_ok=True,parents=True)
  if rows:np.savez_compressed(out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
  if consume:np.savez_compressed(out/'PN_consumed.npz',**{k:np.asarray([r[k] for r in consume]) for k in consume[0]})
  if native:np.savez_compressed(out/'PN_native.npz',**{k:np.asarray([r[k] for r in native]) for k in native[0]})
  if run is not None and run.observer.epochs:run.observer.flush(out/'dng100_observed.npz')
  for fn in [None if spatial is None else spatial.close,undo_motor,None if air is None else air.close,None if run is None else run.close]:
   if fn is not None:
    try:fn()
    except BaseException as e:status.update(status='FAILED',cleanup_error=repr(e))
  status.update(CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-wall,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);save(out/'RESULT.json',status);signal.alarm(0);print(json.dumps({k:status[k] for k in ['status','arm','attempted_ms','committed_ms','CPU_s','wall_s']}),flush=True)
 if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

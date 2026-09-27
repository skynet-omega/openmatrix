"""Bounded OFF continuation with same-call scalar and CSR operand capture."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,argparse,json,time,signal,resource,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'aporte_motor'))
from resume49 import build,save,sha,need,C48

def observer_factory(brain,stimulus):
 from operand_observer import install
 return install(brain,stimulus)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--ms',type=int,choices=[4,50],required=True);ap.add_argument('--contract',type=Path,required=True);args=ap.parse_args()
 need(not args.out.exists(),'Preserve previous run');contract=json.loads(args.contract.read_text());need(contract['frozen_before_acquisition'],'Unfrozen acquisition')
 for name,digest in contract['sources'].items():need(sha(name)==digest,'Changed acquisition source '+name)
 wall=time.monotonic();cpu=time.process_time();run=None
 report=dict(schema='capture49_run_v1',status='STARTED',attempted_ms=0,committed_ms=0,source=str(args.source),contract_sha256=sha(args.contract))
 def stop(*_):raise TimeoutError('Finite capture budget')
 signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(1800);resource.setrlimit(resource.RLIMIT_CPU,(1600,1610))
 try:
  run=build(args.source,args.out,observer_installer=observer_factory)
  import cupy as cp
  from source_inventory import imported,verify
  current=imported();lock=json.loads((C48/'SOURCES.json').read_text())
  changed=[p for p,v in current.items() if p in lock and lock[p]!=v]
  unknown=[p for p in current if p not in lock and not Path(p).is_relative_to(HERE)]
  need(not changed and not unknown,'Unexpected executed sources '+str(changed+unknown));save(args.out/'EXECUTED_SOURCES.json',current)
  detail=set(range(1,args.ms+1)) if args.ms==4 else set(contract['detail_tail_ms'])
  rows=[];proprio=[];capture=[]
  print(json.dumps(dict(status='RESTORED_EXACT',wall_s=time.monotonic()-wall)),flush=True)
  for j in range(1,args.ms+1):
   report['attempted_ms']+=1;save(args.out/'STATUS.json',report)
   # The sample is the pending physical input consumed by this macrostep.
   pending=run.obj.core.pending_proprioception
   proprio.append({k:np.asarray(pending[k]).copy() for k in ('angles_rad','angular_velocity_rad_s','normalized_afferent_drive')})
   rows.append(run.step());report['committed_ms']+=1
   ob=run.observer
   if j in detail:
    file=args.out/f'operands_{j:03d}ms.npz';ob.flush(file);capture.append(dict(tail_ms=j,file=file.name,sha256=sha(file),bytes=file.stat().st_size))
   else:ob.epochs.clear();ob.arrays.clear();ob.edge_arrays.clear()
   need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<24*1024**3,'RAM budget')
   free,total=cp.cuda.runtime.memGetInfo();need(total-free<14*1024**3,'VRAM budget')
   if j%5==0 or j==args.ms:print(json.dumps(dict(status='STEP',tail_ms=j,wall_s=time.monotonic()-wall)),flush=True)
  np.savez_compressed(args.out/'traces.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
  np.savez_compressed(args.out/'proprioception_consumed.npz',**{k:np.stack([r[k] for r in proprio]) for k in proprio[0]})
  run.save(args.out/'final_state');save(args.out/'EVENTS.json',run.session.events.audit)
  report.update(initial=run.initial,appendix=run.appendix,runtime=run.session.report(),observer=run.observer.report(),detail=capture,status='COMPLETE')
  verify(current)
 except BaseException as exc:
  report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());print(report['traceback'],flush=True)
 finally:
  if run is not None:
   try:run.close()
   except BaseException as exc:report.update(status='FAILED',cleanup_error=repr(exc))
  report.update(wall_s=time.monotonic()-wall,CPU_s=time.process_time()-cpu,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  args.out.mkdir(parents=True,exist_ok=True);save(args.out/'RESULT.json',report);signal.alarm(0)
  print(json.dumps({k:report[k] for k in ('status','attempted_ms','committed_ms','wall_s','CPU_s','peak_RSS_bytes')}),flush=True)
 if report['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib,time,subprocess,sys,os
H=Path(__file__).resolve().parent

def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 p=json.loads((H/'PILOT_PLAN.json').read_text());b=p['budgets'];freeze=json.loads((H/'FREEZE.json').read_text())
 if hashlib.sha256((H/'PILOT_PLAN.json').read_bytes()).hexdigest()!=freeze['plan_sha256']:raise ValueError('contract changed')
 if (H/'QUEUE_START.json').exists():raise ValueError('immutable queue; no retry')
 start=time.monotonic();save(H/'QUEUE_START.json',{'unix_s':time.time(),'pid':os.getpid()});reports=[];used_ms=used_cpu=0;status='RUNNING';error=None;active=None
 try:
  schedule=[(True,n,p['qualification_ms']) for n in p['qualifications']]+[(False,n,p['duration_ms']) for n in p['arms']]
  for qual,name,duration in schedule:
   active=('qual_' if qual else '')+name;cpucap=b['per_qualification_CPU_s' if qual else 'per_arm_CPU_s'];wallcap=b['per_qualification_wall_s' if qual else 'per_arm_wall_s']
   if used_cpu+cpucap>b['queue_CPU_s'] or time.monotonic()-start+wallcap>b['queue_wall_s'] or used_ms+duration>b['CNS_ms']:raise RuntimeError('insufficient remaining finite budget')
   save(H/'QUEUE_STATUS.json',{'status':'RUNNING','active':active,'completed':reports,'CPU_s':used_cpu,'attempted_ms':used_ms})
   cmd=[sys.executable,'-B',str(H/'run_pilot.py'),name]+(['--qualification'] if qual else [])
   with (H/(active+'.log')).open('x') as log:job=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=wallcap+10)
   result=json.loads((H/active/'RESULT.json').read_text());rec={k:result[k] for k in ['status','arm','qualification','attempted_ms','committed_ms','CPU_s','wall_s']};reports.append(rec);used_cpu+=result['CPU_s'];used_ms+=result['attempted_ms'];print(json.dumps(rec),flush=True)
   if job.returncode or result['status']!='COMPLETE':raise RuntimeError('arm failed '+active)
   if sum(x.stat().st_size for x in H.rglob('*') if x.is_file())>b['new_disk_bytes']:raise RuntimeError('disk cap')
  status='COMPLETE'
 except BaseException as e:status='STOPPED';error=repr(e)
 finally:
  out={'status':status,'error':error,'active':active,'arms':reports,'CPU_s':used_cpu,'attempted_ms':used_ms,'committed_ms':sum(x['committed_ms'] for x in reports),'queue_wall_s':time.monotonic()-start}
  if active and not (H/active/'RESULT.json').exists():out['unclosed_worker_accounting']={'arm':active,'reserved_CPU_s':cpucap,'reserved_ms':duration}
  save(H/'QUEUE_RESULT.json',out);save(H/'QUEUE_STATUS.json',out);print(json.dumps(out),flush=True)
 if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

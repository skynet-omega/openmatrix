"""Single finite GPU queue with live exact qualification and conservative failure accounting."""
from pathlib import Path
import json,time,subprocess,sys,os,hashlib,resource
import numpy as np
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def save(p,d):
 t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def qualify():
 actual=H/'qual';reference=H/'reference/qualification56'
 files=['traces.npz','PN_consumed.npz','wide_observation.npz','dng100_observed.npz']
 for name in files:
  with np.load(actual/name,allow_pickle=False) as a,np.load(reference/name,allow_pickle=False) as b:
   need(set(a.files)==set(b.files),'qualification keys '+name)
   for k in b.files:need(np.array_equal(a[k],b[k]),'qualification '+name+'/'+k)
 with np.load(actual/'neural_and_inputs.npz') as a,np.load(reference/'neural_and_inputs.npz') as b:
  for k in b.files:need(np.array_equal(a[k],b[k]),'neural qualification '+k)
 for name in ['SCIENTIFIC_WITNESS.json','EVENTS.json']:need(json.loads((actual/name).read_text())==json.loads((reference/name).read_text()),'qualification owners '+name)
 save(H/'LIVE_QUALIFICATION.json',dict(exact=True,reference='56/repair01/qual_disabled',scope='Read-only PN probe vs previously qualified run, all retained trace/panel/RHS/owner fields'))
def main():
 p=json.loads((H/'PLAN.json').read_text());f=json.loads((H/'FREEZE.json').read_text());need(hashlib.sha256((H/'PLAN.json').read_bytes()).hexdigest()==f['plan_sha256'],'contract changed');need(not (H/'START.json').exists(),'immutable queue')
 save(H/'START.json',dict(unix_s=time.time(),pid=os.getpid()));wall=time.monotonic();cpu=time.process_time();records=[];active=None;error=None;status='RUNNING';unclosed=None
 try:
  for index,(arm,spec) in enumerate(p['arms'].items()):
   if index==1:qualify()
   active=arm;cap=p['qualification'] if spec['duration_ms']==2 else p['per_arm']
   need(sum(x['CPU_s'] for x in records)+cap['CPU_s']<=p['queue_CPU_s'],'CPU reserve');need(time.monotonic()-wall+cap['wall_s']<=p['queue_wall_s'],'wall reserve');need(sum(x['attempted_ms'] for x in records)+spec['duration_ms']<=p['CNS_ms'],'CNS reserve')
   save(H/'QUEUE_STATUS.json',dict(status='RUNNING',active=active,arms=records))
   before=resource.getrusage(resource.RUSAGE_CHILDREN)
   with (H/(arm+'.log')).open('x',encoding='utf-8') as log:
    try:job=subprocess.run([sys.executable,'-B',str(H/'run_natural57.py'),arm],stdout=log,stderr=subprocess.STDOUT,timeout=cap['wall_s']+10)
    finally:
     after=resource.getrusage(resource.RUSAGE_CHILDREN);save(H/(arm+'_PROCESS_COST.json'),dict(child_user_system_CPU_s=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime)))
   r=json.loads((H/arm/'RESULT.json').read_text());records.append({k:r[k] for k in ['arm','status','attempted_ms','committed_ms','CPU_s','wall_s']});print(json.dumps(records[-1]),flush=True);need(job.returncode==0 and r['status']=='COMPLETE','failed '+arm)
  status='COMPLETE'
 except BaseException as e:status='STOPPED';error=repr(e)
 finally:
  if active and not (H/active/'RESULT.json').exists():unclosed=dict(CPU_s=cap['CPU_s']+5,CNS_ms=spec['duration_ms'])
  d=dict(status=status,error=error,active=active,arms=records,queue_wall_s=time.monotonic()-wall,CPU_s=sum(x['CPU_s'] for x in records),attempted_ms=sum(x['attempted_ms'] for x in records),committed_ms=sum(x['committed_ms'] for x in records),supervisor_CPU_s=time.process_time()-cpu,unclosed_reserved=unclosed)
  save(H/'QUEUE_RESULT.json',d);save(H/'QUEUE_STATUS.json',d);print(json.dumps(d),flush=True)
 if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

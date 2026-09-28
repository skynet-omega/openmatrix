"""One finite GPU queue; require live transparency before scientific arms."""
from pathlib import Path
import json,time,subprocess,sys,os,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def qualify():
 reference=H/'reference/qualification55';results={}
 for arm in ['qual_disabled','qual_identity']:
  for name in ['traces.npz','PN_consumed.npz','neural_and_inputs.npz','wide_observation.npz','dng100_observed.npz']:
   with np.load(H/arm/name) as a,np.load(reference/name) as b:
    need(set(a.files)==set(b.files),'qualification keys '+name)
    for k in a.files:need(np.array_equal(a[k],b[k]),'qualification difference '+arm+'/'+name+'/'+k)
  for name in ['SCIENTIFIC_WITNESS.json','EVENTS.json']:need(json.loads((H/arm/name).read_text())==json.loads((reference/name).read_text()),'qualification owners '+arm+'/'+name)
  with np.load(H/arm/'PN_native.npz') as a,np.load(H/arm/'PN_consumed.npz') as b:
   for k in ['first','last','lo','hi','counts']:need(np.array_equal(a[k],b[k]),'passthrough differs '+arm+'/'+k)
   need(not np.any(a['errors']),'passthrough errors')
  results[arm]=dict(exact=True,reference='55/qual_sham_untouched')
 save(H/'LIVE_QUALIFICATION.json',results)
def main():
 p=json.loads((H/'A_PLAN.json').read_text());f=json.loads((H/'A_FREEZE.json').read_text());need(hashlib.sha256((H/'A_PLAN.json').read_bytes()).hexdigest()==f['contract_sha256'],'contract changed');need(not (H/'START.json').exists(),'immutable queue')
 save(H/'START.json',dict(unix_s=time.time(),pid=os.getpid()));wall=time.monotonic();records=[];active=None;error=None;status='RUNNING'
 try:
  for index,(arm,spec) in enumerate(p['arms'].items()):
   if index==2:qualify()
   active=arm;cap=p['qualification'] if spec['duration_ms']==2 else p['per_arm']
   need(sum(x['CPU_s'] for x in records)+cap['CPU_s']<=p['queue_CPU_s'],'CPU reserve');need(time.monotonic()-wall+cap['wall_s']<=p['queue_wall_s'],'wall reserve');need(sum(x['attempted_ms'] for x in records)+spec['duration_ms']<=p['CNS_ms'],'CNS reserve')
   save(H/'QUEUE_STATUS.json',dict(status='RUNNING',active=active,arms=records))
   with (H/(arm+'.log')).open('x') as log:job=subprocess.run([sys.executable,'-B',str(H/'run_hold56.py'),arm],stdout=log,stderr=subprocess.STDOUT,timeout=cap['wall_s']+10)
   r=json.loads((H/arm/'RESULT.json').read_text());records.append({k:r[k] for k in ['arm','status','attempted_ms','committed_ms','CPU_s','wall_s']});print(json.dumps(records[-1]),flush=True);need(job.returncode==0 and r['status']=='COMPLETE','failed '+arm)
  status='COMPLETE'
 except BaseException as e:status='STOPPED';error=repr(e)
 finally:
  d=dict(status=status,error=error,active=active,arms=records,queue_wall_s=time.monotonic()-wall,CPU_s=sum(x['CPU_s'] for x in records),attempted_ms=sum(x['attempted_ms'] for x in records),committed_ms=sum(x['committed_ms'] for x in records))
  if active and not (H/active/'RESULT.json').exists():d['unclosed_reserved']=dict(CPU_s=cap['CPU_s'],CNS_ms=spec['duration_ms'])
  save(H/'QUEUE_RESULT.json',d);save(H/'QUEUE_STATUS.json',d);print(json.dumps(d),flush=True)
 if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

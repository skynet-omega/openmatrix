"""Qualify both live identities before either reciprocal intervention."""
from pathlib import Path
import json,time,subprocess,sys,os,hashlib
import numpy as np
H=Path(__file__).resolve().parent
def need(x,m):
 if not x:raise ValueError(m)
def save(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def qualify():
 result={}
 for receiver in ['sham','profile']:
  a=H/('qual_'+receiver+'_untouched');b=H/('qual_'+receiver+'_self')
  files=['traces.npz','PN_consumed.npz','neural_and_inputs.npz','wide_observation.npz','dng100_observed.npz']
  for name in files:
   with np.load(a/name) as x,np.load(b/name) as y:
    need(set(x.files)==set(y.files),'qualification keys '+name)
    for k in x.files:need(np.array_equal(x[k],y[k]),'qualification differs '+receiver+' '+name+' '+k)
  for name in ['SCIENTIFIC_WITNESS.json','EVENTS.json']:
   need(json.loads((a/name).read_text())==json.loads((b/name).read_text()),'qualification owner differs '+receiver+' '+name)
  result[receiver]=dict(exact=True,array_files=files,owners=['scientific','events'])
 save(H/'PN_LIVE_QUALIFICATION.json',result)
def main():
 p=json.loads((H/'PN_PLAN.json').read_text());f=json.loads((H/'PN_FREEZE.json').read_text())
 need(hashlib.sha256((H/'PN_PLAN.json').read_bytes()).hexdigest()==f['contract_sha256'],'changed contract')
 need(not (H/'PN_START.json').exists(),'immutable queue')
 feedback=json.loads((H/'FEEDBACK_RESULT.json').read_text());need(feedback['status']=='COMPLETE','C did not close; no simultaneous GPU')
 save(H/'PN_START.json',dict(unix_s=time.time(),pid=os.getpid()));wall=time.monotonic();records=[];active=None;status='RUNNING';error=None
 try:
  for index,(arm,spec) in enumerate(p['arms'].items()):
   if index==4:qualify()
   active=arm;cap=p['qualification'] if spec['duration_ms']==2 else p['per_arm']
   need(sum(x['CPU_s'] for x in records)+cap['CPU_s']<=p['queue_CPU_s'],'CPU reserve')
   need(time.monotonic()-wall+cap['wall_s']<=p['queue_wall_s'],'wall reserve')
   need(sum(x['attempted_ms'] for x in records)+spec['duration_ms']<=p['CNS_ms'],'exposure reserve')
   save(H/'PN_STATUS.json',dict(status='RUNNING',active=active,arms=records))
   with (H/(arm+'.log')).open('x') as log:job=subprocess.run([sys.executable,'-B',str(H/'run_pn.py'),arm],stdout=log,stderr=subprocess.STDOUT,timeout=cap['wall_s']+10)
   r=json.loads((H/arm/'RESULT.json').read_text());records.append({k:r[k] for k in ['arm','status','attempted_ms','committed_ms','CPU_s','wall_s']});print(json.dumps(records[-1]),flush=True)
   need(job.returncode==0 and r['status']=='COMPLETE','failed '+arm)
  status='COMPLETE'
 except BaseException as e:status='STOPPED';error=repr(e)
 finally:
  d=dict(status=status,error=error,active=active,arms=records,queue_wall_s=time.monotonic()-wall,CPU_s=sum(x['CPU_s'] for x in records),attempted_ms=sum(x['attempted_ms'] for x in records),committed_ms=sum(x['committed_ms'] for x in records))
  if active and not (H/active/'RESULT.json').exists():d['unclosed_reserved']=dict(CPU_s=cap['CPU_s'],CNS_ms=spec['duration_ms'])
  save(H/'PN_RESULT.json',d);save(H/'PN_STATUS.json',d);print(json.dumps(d),flush=True)
 if status!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':main()

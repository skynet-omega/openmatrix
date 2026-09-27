"""Finite sequential GPU campaign; failures stop, never auto-retry."""
from pathlib import Path
import sys,json,time,subprocess
H=Path(__file__).resolve().parent;p=json.loads((H/'PILOT_PLAN.json').read_text());freeze=json.loads((H/'PILOT_FREEZE.json').read_text());qual=json.loads((H/'QUALIFICATION.json').read_text())
if qual['status']!='PASS':raise ValueError('Qualification absent')
start=time.monotonic();reports=[];qreports=[json.loads((H/('qual_'+a)/'RESULT.json').read_text()) for a in p['qualifications']];used_cpu=sum(r['CPU_s'] for r in qreports);used_ms=sum(r['attempted_ms'] for r in qreports);status='RUNNING'
def save(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def compact(r):return {k:r[k] for k in ['arm','status','attempted_ms','committed_ms','CPU_s','wall_s']}
try:
 for arm in p['arms']:
  elapsed=time.time()-freeze['created_unix_s']
  if elapsed+400>p['budgets']['wall_s'] or used_cpu+440>p['budgets']['CPU_s'] or used_ms+90>p['budgets']['neural_ms_total']:raise RuntimeError('Insufficient remaining fixed budget')
  save(H/'QUEUE_STATUS.json',dict(status='RUNNING',active=arm,completed=[compact(r) for r in reports],pilot_elapsed_s=elapsed,CPU_s=used_cpu,attempted_ms=used_ms))
  with (H/(arm+'.log')).open('x') as log:r=subprocess.run([sys.executable,'-O',str(H/'run_pilot.py'),arm],stdout=log,stderr=subprocess.STDOUT,timeout=410)
  result=json.loads((H/arm/'RESULT.json').read_text());reports.append(result);used_cpu+=result['CPU_s'];used_ms+=result['attempted_ms'];print(json.dumps(compact(result)),flush=True)
  if r.returncode or result['status']!='COMPLETE':raise RuntimeError('Arm failed '+arm)
  disk=sum(x.stat().st_size for x in H.rglob('*') if x.is_file())
  if disk>p['budgets']['new_disk_bytes']:raise RuntimeError('Disk budget')
 status='COMPLETE'
except BaseException as exc:status='STOPPED';error=repr(exc)
finally:
 out=dict(status=status,arms=[compact(r) for r in reports],qualification_CPU_s=sum(r['CPU_s'] for r in qreports),CPU_s=used_cpu,attempted_ms=used_ms,committed_ms=sum(r['committed_ms'] for r in reports+qreports),queue_wall_s=time.monotonic()-start,pilot_elapsed_s=time.time()-freeze['created_unix_s'])
 if status!='COMPLETE':out['error']=error
 save(H/'QUEUE_RESULT.json',out);save(H/'QUEUE_STATUS.json',out);print(json.dumps(out),flush=True)
if status!='COMPLETE':raise SystemExit(1)

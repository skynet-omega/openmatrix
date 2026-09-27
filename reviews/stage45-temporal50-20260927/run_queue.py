"""Sequential GPU queue with fixed aggregate limits; stop on first failure."""
import subprocess,time,json,resource,sys
from pathlib import Path
from temporal_input import ARMS,need
H=Path(__file__).resolve().parent
plan=json.loads((H/'PLAN.json').read_text());start=time.monotonic();reports=[];status='RUNNING'
try:
 for arm in ARMS:
  elapsed=time.monotonic()-start;used=sum(x.get('CPU_s',0) for x in reports)
  need(elapsed+600<=plan['budget']['wall_s_max'],'Insufficient aggregate wall reserve')
  need(used+500<=plan['budget']['CPU_s_max'],'Insufficient aggregate CPU reserve')
  with (H/(arm+'.log')).open('x') as log:p=subprocess.run([sys.executable,'-O',str(H/'run_arm.py'),arm],stdout=log,stderr=subprocess.STDOUT,timeout=610)
  r=json.loads((H/arm/'RESULT.json').read_text());reports.append(r)
  (H/'QUEUE_STATUS.json').write_text(json.dumps(dict(status='RUNNING',arms=[{k:x[k] for k in ['arm','status','committed_ms','CPU_s','wall_s']} for x in reports],wall_s=time.monotonic()-start),indent=2)+'\n')
  need(p.returncode==0 and r['status']=='COMPLETE','Arm failure '+arm)
 status='COMPLETE'
except BaseException as exc:
 status='STOPPED';error=repr(exc)
finally:
 out=dict(status=status,arms=[{k:x[k] for k in ['arm','status','committed_ms','CPU_s','wall_s']} for x in reports],wall_s=time.monotonic()-start,total_neural_ms=sum(x['committed_ms'] for x in reports),CPU_s=sum(x['CPU_s'] for x in reports))
 if status!='COMPLETE':out['error']=error
 (H/'QUEUE_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if status!='COMPLETE':raise SystemExit(1)

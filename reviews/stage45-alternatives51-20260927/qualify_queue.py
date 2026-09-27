from pathlib import Path
import subprocess,sys,json,time
H=Path(__file__).resolve().parent;start=time.monotonic();reports=[]
for arm in ['air','conductance','current']:
 with (H/('qual_'+arm+'.log')).open('x') as f:
  p=subprocess.run([sys.executable,'-O',str(H/'run_pilot.py'),arm,'--qualification'],stdout=f,stderr=subprocess.STDOUT,timeout=410)
 r=json.loads((H/('qual_'+arm)/'RESULT.json').read_text());reports.append({k:r[k] for k in ['arm','status','attempted_ms','committed_ms','CPU_s','wall_s']});print(json.dumps(reports[-1]),flush=True)
 if p.returncode or r['status']!='COMPLETE':break
(H/'QUAL_QUEUE.json').write_text(json.dumps({'reports':reports,'wall_s':time.monotonic()-start},indent=2)+'\n')
if len(reports)!=3 or any(x['status']!='COMPLETE' for x in reports):raise SystemExit(1)

"""One finite sequential continuation, never a recurring automation."""
from pathlib import Path
import os,sys,time,subprocess,json
HERE=Path(__file__).resolve().parent
running=1848121
record={'waits_for_original_sham_pid':running,'conditions':['uniform','odor_left','odor_right'],'horizon_ms':100,'runs':[],'state':'WAITING_FOR_SHAM'}
def save():
 (HERE/'CAMPAIGN_QUEUE.json').write_text(json.dumps(record,indent=2)+'\n')
save()
while Path('/proc/'+str(running)).exists():
 cmd=Path('/proc/'+str(running)+'/cmdline').read_bytes()
 if b'run_pilot.py' not in cmd:raise RuntimeError('PID ownership changed')
 time.sleep(5)
if not (HERE/'pilot_sham_01/RESULT.json').is_file():raise RuntimeError('Missing original attempt receipt; inspect before continuation')
start=time.perf_counter();record['state']='RUNNING';save()
for condition in record['conditions']:
 out=HERE/('pilot_'+condition+'_01')
 attempts=[p for p in HERE.iterdir() if p.is_dir() and p.name.startswith(('transport_','pilot_'))]
 if out.exists() or len(attempts)>=14:raise RuntimeError('Unique result / scientific attempt budget')
 row={'condition':condition,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};record['runs'].append(row);save()
 t=time.perf_counter()
 with (HERE/(out.name+'.log')).open('x') as log:
  try:
   p=subprocess.run([sys.executable,'-B',str(HERE/'run_pilot.py'),'--out',str(out),'--odor',condition,'--ms','100'],stdout=log,stderr=subprocess.STDOUT,timeout=1200)
   row['exit_code']=p.returncode
  except subprocess.TimeoutExpired:row['exit_code']='TIMEOUT'
 row['wall_s']=time.perf_counter()-t;save()
 if row['exit_code']!=0:
  record['state']='STOPPED_ON_FAILURE';save();raise SystemExit(2)
 if time.perf_counter()-start>3600:raise RuntimeError('Remaining aggregate budget')
record['state']='COMPLETE';save()

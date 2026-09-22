from pathlib import Path
import subprocess,time,json,sys
H=Path(__file__).resolve().parent
jobs=[]
for case in ['hh','mixed']:jobs.append((case+'_reference',['--case',case,'--reference']))
for case in ['clamp','stiff','hh','mixed']:
    for profile in ['fast','precise']:
        if case=='stiff' and profile=='fast':continue
        args=['--case',case,'--profile',profile]
        if case in ['hh','mixed']:args+=['--reference-dir','pilots/'+case+'_reference']
        jobs.append((case+'_'+profile,args))
ledger=[]
for name,args in jobs:
    start=time.perf_counter()
    try:
        with (H/('pilot_'+name+'.log')).open('w') as log:r=subprocess.run([sys.executable,'-B','-O','pilot.py','pilots/'+name,*args],cwd=H,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        code=r.returncode
    except subprocess.TimeoutExpired:code='timeout180'
    row={'name':name,'subprocess_wall_s':time.perf_counter()-start,'exit':code};ledger.append(row)
    (H/'PILOT_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(row),flush=True)

from pathlib import Path
import subprocess,sys,json,time,threading
H=Path(__file__).resolve().parent
jobs=[]
for case in ['clamp','stiff','hh','mixed']:
    for profile in ['fast','precise']:
        args=['pilot.py','runs/'+case+'_'+profile,'--case',case,'--profile',profile]
        if case in ['hh','mixed']:args+=['--reference-dir','references/'+case+'_reference']
        jobs.append((case+'_'+profile,args))
for profile in ['fast','precise']:
    jobs.append(('hh_python_'+profile,['pilot.py','runs/hh_python_'+profile,'--case','hh','--profile',profile,'--implementation','python','--reference-dir','references/hh_reference']))
for profile in ['fast','precise']:
    for implementation in ['python','native']:
        name='one_'+implementation+'_'+profile
        jobs.append((name,['bench.py','runs/'+name,'--implementation',implementation,'--profile',profile,'--reference-dir','references/large_fine']))
for implementation in ['reference','reference_fine']:
    name='ten_'+implementation;jobs.append((name,['bench.py','runs/'+name,'--implementation',implementation,'--seconds','10']))
for profile in ['fast','precise']:
    name='ten_native_'+profile;jobs.append((name,['bench.py','runs/'+name,'--implementation','native','--seconds','10','--profile',profile,'--reference-dir','runs/ten_reference_fine']))
(H/'runs').mkdir(exist_ok=False);ledger=[];campaign_start=time.perf_counter()
for name,args in jobs:
    if time.perf_counter()-campaign_start>3200:raise RuntimeError('campaign execution budget')
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.wait(.5):
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if r.returncode==0:
                try:samples.append([float(x.strip()) for x in r.stdout.strip().split(',')])
                except ValueError:pass
    th=threading.Thread(target=monitor,daemon=True);th.start();start=time.perf_counter();code=None
    with (H/(name+'.log')).open('w') as log:
        try:code=subprocess.run([sys.executable,'-B','-O',*args],cwd=H,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        except subprocess.TimeoutExpired:code='timeout240'
    stop.set();th.join(timeout=5)
    row={'name':name,'args':args,'exit':code,'subprocess_wall_s':time.perf_counter()-start,'device_wide_peak_MiB':max([x[0] for x in samples],default=None),'device_wide_mean_gpu_percent':sum(x[1] for x in samples)/len(samples) if samples else None,'monitor':'device-wide including other GPU work, 0.5s sampling; not precise per-process allocation peak'};ledger.append(row)
    (H/'CAMPAIGN.json').write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(row),flush=True)
    if code!=0:break

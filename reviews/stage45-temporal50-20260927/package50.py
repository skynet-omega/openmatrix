"""Small public capsule: all eight recorded arms and independent CPU verifiers."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time,zipfile
H=Path(__file__).resolve().parent
OUT=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
ARMS=('p00','p01','p10','p11','m00','m01','m10','m11')
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')
def need(ok,m):
    if not ok:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
start=time.process_time();name='ETAPA45_TEMPORAL_20260927_50';archive=OUT/'salida'/(name+'.zip')
need(not archive.exists() and not (H/'MANIFEST.json').exists(),'Preserve prior capsule')
files=[]
excluded={'LOCAL_DELIVERY.json','REMOTE_DELIVERY.json','QUEUE_STATUS.json','STATUS.json','CAPSULE_PREFLIGHT.json'}
for p in H.iterdir():
    if p.is_file() and p.suffix in ('.py','.json','.md','.png','.csv','.npz') and p.name not in excluded:
        files.append(p)
for directory in ['reference','aporte_motor50']:
    files.extend(p for p in (H/directory).iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.npz','.cu'))
for arm in ARMS:
    files.extend(H/arm/n for n in ['traces.npz','input_and_observers.npz','dng100_observed.npz','TEMPORAL_OWNER.json','RESULT.json','EXECUTED_SOURCES.json','PREFIX49.json','APPENDIX.json','CLOCK_OVERLAY.json'])
    files.append(H/(arm+'.log'))
    files.append(H/arm/'final_state/MANIFEST.json')
m=dict(schema='temporal50_recorded_evidence_v1',scope='All eight recorded arms and analysis, not full CNS replay; full checkpoint/dependency ZIP is separate',files={})
for p in sorted(set(files)):
    b=p.read_bytes()
    if p.suffix in ('.py','.json','.md','.log'):
        need(not SECRET.search(b),'Credential pattern')
    m['files'][str(p.relative_to(H))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
(H/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for n in list(m['files'])+['MANIFEST.json']:
        z.write(H/n,name+'/'+n)
need(archive.stat().st_size<95*1024**2,'Public ZIP bound')
clean=OUT/'recibos'/(name+'_LOCAL');clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
root=clean/name;checks=[]
for args in [['verify_complete50.py'],['-O','verify_complete50.py'],['-O','test_analysis50.py','--real'],['-O','test_law_supplement50.py'],['-O','verify_supplements50.py']]:
    r=subprocess.run([sys.executable,*args],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
    need(r.returncode==0,'Cold verification '+r.stderr);checks.append(json.loads(r.stdout))
out=dict(zip=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(m['files']),checks=checks,
    CPU_s=time.process_time()-start,scope=m['scope'])
(H/'LOCAL_DELIVERY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['zip','bytes','sha256','files','CPU_s','scope']}))

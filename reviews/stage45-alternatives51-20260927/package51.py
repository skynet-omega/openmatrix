"""Compact public capsule with all recorded arms and independent CPU verifiers."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys,time,zipfile
H=Path(__file__).resolve().parent
OUT=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
ARMS=tuple(json.loads((H/'PILOT_PLAN.json').read_text())['arms'])
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')
def need(ok,m):
 if not ok:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 start=time.process_time();wall=time.monotonic();name='ETAPA45_ALTERNATIVAS_20260927_51';archive=OUT/'salida'/(name+'.zip')
 need(not archive.exists() and not (H/'MANIFEST.json').exists(),'Preserve prior capsule')
 need(json.loads((H/'QUEUE_RESULT.json').read_text())['status']=='COMPLETE','All scientific arms complete')
 excluded={'LOCAL_DELIVERY.json','REMOTE_DELIVERY.json','QUEUE_STATUS.json','STATUS.json','MANIFEST.json','FULL_CAPSULE.json','package.log','full_capsule.log','publish.log'}
 files=[p for p in H.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.png','.csv','.npz','.cu','.log','.patch') and p.name not in excluded]
 files.extend(p for p in (H/'aporte_motor').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json','.npz','.cu','.csv','.log'))
 for arm in [*ARMS,*('qual_'+k for k in json.loads((H/'PILOT_PLAN.json').read_text())['qualifications'])]:
  files.extend(p for p in (H/arm).iterdir() if p.is_file() and p.name!='STATUS.json' and p.suffix in ('.npz','.json','.md','.log'))
  if (H/arm/'final_state/MANIFEST.json').exists():files.append(H/arm/'final_state/MANIFEST.json')
 m=dict(schema='alternatives51_recorded_evidence_v1',scope='Ten complete recorded90ms arms,16ms qualification and ten bounded alternative screens; CPU reproduction of evidence, not full CNS replay. Full checkpoints and dependencies are in separate lossless archive.',files={})
 for p in sorted(set(files)):
  b=p.read_bytes()
  if p.suffix in ('.py','.json','.md','.log','.cu','.patch'):need(not SECRET.search(b),'Credential pattern')
  m['files'][str(p.relative_to(H))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
 (H/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n in list(m['files'])+['MANIFEST.json']:z.write(H/n,name+'/'+n)
 need(archive.stat().st_size<95*1024**2,'Public ZIP bound')
 clean=OUT/'recibos'/(name+'_LOCAL');clean.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:z.extractall(clean)
 root=clean/name;checks=[]
 for args in [['verify_complete51.py'],['-O','verify_complete51.py'],['test_corruptions51.py'],['-O','test_corruptions51.py']]:
  r=subprocess.run([sys.executable,*args],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=120)
  need(r.returncode==0,'Cold verification '+r.stderr);checks.append(json.loads(r.stdout))
 out=dict(zip=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(m['files']),checks=checks,CPU_s=time.process_time()-start,wall_s=time.monotonic()-wall,scope=m['scope'])
 (H/'LOCAL_DELIVERY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['zip','bytes','sha256','files','CPU_s','wall_s','scope']}))

if __name__=='__main__':main()

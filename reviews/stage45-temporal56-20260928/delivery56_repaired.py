"""Repair closure limits and recover the existing immutable archive; no new CNS."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,sys,os,re,time,resource,urllib.request,argparse
H=Path(__file__).resolve().parent;R=H.parents[1]
X=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO');STEM='ETAPA45_TRANSFERENCIA_TEMPORAL_20260928_56';LEAF='stage45-temporal56-20260928';PREFIX='reviews/'+LEAF
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')
SKIP={'MANIFEST.json','LOCAL_DELIVERY.json','REMOTE_DELIVERY.json','PUBLICATION_SCOPE.json','FULL_CAPSULE.json','ARCHIVE_PARTS.json','pack.log','publish.log','QUEUE_STATUS.json','STATUS.json'}
def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def scan(p):
 if p.suffix.lower() not in ['.py','.json','.md','.log','.txt','.cu','.cpp','.hpp','.html','.diff','.patch','.tmp']:return
 with p.open('rb') as f:
  tail=b''
  for b in iter(lambda:f.read(1048576),b''):
   part=tail+b;lower=part.lower()
   if b'apikey_' in lower or b'sk-' in lower or b'private key' in lower:need(not SECRET.search(part),'Credential pattern in '+str(p))
   tail=b[-200:]
def info(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def files_at(full=False):
 files={}
 for p in H.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(H)
  if any(x in rel.parts for x in ['__pycache__','cache','continuidad_previa']):continue
  if not full and 'final_state' in rel.parts:continue
  if p.suffix in ['.pyc','.zip']:continue
  if p.name in SKIP and not (p.name=='MANIFEST.json' and rel.parent!=Path('.')):continue
  # Keep the failed UTF-8 attempt's partial .tmp files as evidence.
  files[str(rel)]=p
 return files
def disk_current():
 total=sum(p.stat().st_size for p in H.rglob('*') if p.is_file())
 for base in [X/'salida',X/'recibos']:
  for p in base.glob(STEM+'*'):
   total+=p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
 baseline=H/'DISK_BASELINE.json'
 if baseline.exists():
  v=json.loads(baseline.read_text());now=sum(p.stat().st_size for p in Path(v['git_objects']).rglob('*') if p.is_file());total+=max(0,now-v['git_objects_bytes_before'])
 return total
def extract_verify(path,tag):
 d=X/'recibos'/(STEM+'_'+tag);d.mkdir(exist_ok=False)
 with zipfile.ZipFile(path) as z:
  need(len(z.namelist())==len(set(z.namelist())) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'archive paths');z.extractall(d)
 root=d/STEM
 run=subprocess.run([sys.executable,'-B','-O','check_delivery56.py','--corruptions'],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,encoding='utf-8',timeout=80)
 need(run.returncode==0,run.stdout+'\n'+run.stderr)
 return {'extraction':str(root),'verification':json.loads(run.stdout),'new_CNS_ms':0}
def pack():
 start=time.process_time();child0=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()
 need(json.loads((H/'repair01/QUEUE_RESULT.json').read_text())['status']=='COMPLETE','incomplete queue')
 need((H/'A_RESULTADOS.json').is_file() and (H/'RESULTADOS.md').is_file(),'unverified analysis')
 inventories=[R/'campanas/etapa45_composicion_20260927_48/SOURCES.json',H/'A_SOURCES.json',H/'repair01/A_SOURCES.json']+list(H.rglob('EXECUTED_SOURCES.json'));sources={}
 for inventory in inventories:
  for p,digest in json.loads(inventory.read_text()).items():
   if p in sources:need(sources[p]==digest,'source identity conflict')
   sources[p]=digest
 for p in [R/'campanas/etapa45_postwind_diagnosis_20260925_41/inputs/body.mjb',R/'campanas/etapa45_postwind_diagnosis_20260925_41/inputs/reference.npz']:sources[str(p)]=sha(p)
 source_index={};logical={}
 for path,digest in sources.items():
  p=Path(path);need(sha(p)==digest,'source changed '+path);name='sources/'+digest+p.suffix;logical[name]=p;source_index[path]=name
 dump(H/'SOURCE_INDEX.json',source_index)
 logical.update({'campaign56/'+n:p for n,p in files_at(True).items()})
 for donor in ['sham','profile']:
  for p in (R/'campanas/etapa45_composicion_20260927_48'/donor/'final_state').iterdir():
   if p.is_file():logical['initial48'+donor+'/'+p.name]=p
 entries={};blobs={};paths={};seen_paths={}
 for name,p in logical.items():
  key=str(p)
  if key not in seen_paths:scan(p);seen_paths[key]=info(p)
  item=seen_paths[key];entries[name]=item;digest=item['sha256'];blobs[digest]={'bytes':item['bytes']};paths[digest]=p
 need(disk_current()<10*1024**3,'disk already exceeds budget')
 full=X/'salida'/(STEM+'_ESTADOS_Y_FUENTES.zip')
 index={'schema':'temporal56_content_addressed_v1','files':entries,'blobs':blobs,'restore_sha256':sha(H/'restore_capsule56.py'),'scope':'Complete retained initial/final scientific states and frozen transitive sources, stored once per SHA256. CPU analysis is portable. A GPU run from remapped paths is unqualified.'}
 with zipfile.ZipFile(full,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  z.writestr('FULL_INDEX.json',json.dumps(index,indent=2));z.write(H/'restore_capsule56.py','RESTORE.py')
  for digest,p in paths.items():
   method=zipfile.ZIP_STORED if p.suffix in ['.npz','.npy','.png','.so','.pdf'] else zipfile.ZIP_DEFLATED
   z.write(p,'blobs/'+digest,compress_type=method,compresslevel=None if method==zipfile.ZIP_STORED else 1)
 need(disk_current()<10*1024**3,'full capsule exceeds aggregate disk budget')
 # Read every stored unique blob and reconstruct its correspondence to all logical files.
 sample=X/'recibos'/(STEM+'_FULL_EXTRACTION')
 proof=subprocess.run([sys.executable,'-B','-O',str(H/'restore_capsule56.py'),str(full),'--restore-to',str(sample),'--prefix','campaign56/reference/TERMINALS.npz'],capture_output=True,text=True,encoding='utf-8',timeout=120);need(proof.returncode==0,proof.stdout+'\n'+proof.stderr)
 need(sha(sample/'campaign56/reference/TERMINALS.npz')==sha(H/'reference/TERMINALS.npz'),'materialized capsule sample')
 dump(H/'FULL_CAPSULE.json',dict(path=str(full),**info(full),verification=json.loads(proof.stdout),sample_extraction=str(sample),logical_bytes=sum(x['bytes'] for x in entries.values()),unique_bytes=sum(x['bytes'] for x in blobs.values()),scope='Streaming integrity of every blob and logical mapping, plus one materialized terminal file checked against its source; no new GPU restore.'))
 files=files_at();files['FULL_CAPSULE.json']=H/'FULL_CAPSULE.json'
 for p in files.values():scan(p)
 dump(H/'MANIFEST.json',{'schema':'temporal56_compact_v1','files':{n:info(p) for n,p in files.items()},'scope':'Portable reconstruction of eight recorded scientific arms, two identity qualifications, preserved encoding failure, and CPU temporal bench; full states are in the local capsule.'})
 archive=X/'salida'/(STEM+'.zip')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n,p in {**files,'MANIFEST.json':H/'MANIFEST.json'}.items():z.write(p,STEM+'/'+n)
 need(disk_current()+4*archive.stat().st_size+2*sum(p.stat().st_size for p in files.values())<10*1024**3,'reserve download, split parts and two extractions')
 proof=extract_verify(archive,'LOCAL');child1=resource.getrusage(resource.RUSAGE_CHILDREN)
 receipt=dict(path=str(archive),**info(archive),files=len(files)+1,verification=proof,CPU_s=time.process_time()-start,child_CPU_s=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime,wall_s=time.monotonic()-wall,aggregate_new_disk_bytes=disk_current())
 dump(H/'LOCAL_DELIVERY.json',receipt);print(json.dumps({k:receipt[k] for k in ['path','bytes','sha256','files','CPU_s','child_CPU_s','aggregate_new_disk_bytes']}),flush=True)
def publish():
 start=time.process_time();child0=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic();receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());archive=Path(receipt['path']);need(sha(archive)==receipt['sha256'],'compact changed')
 names=['RESULTADOS.md','DECISION.md','DATOS_PARA_SIGUIENTE_RONDA.md','A_RESULTADOS.json','B_RESULTADOS.json','COMPARACION.png','COMPARACION.pdf','PLAN_INICIAL.md','A_DESIGN.md','FUENTES_Y_DECISIONES.md','REPARACION.md','A_PLAN.json','A_FREEZE.json','B_PLAN.json','A_VERIFY_COST.json','MANIFEST.json','LOCAL_DELIVERY.json','FULL_CAPSULE.json','verify56.py','report56.py','check_delivery56.py','delivery56.py','REPRODUCIR.md']
 names+=['delivery56_repaired.py','CIERRE_REPARACION.md','PACK_FAILURE_ACCOUNTING.json'];files={n:H/n for n in names};parts=[]
 if archive.stat().st_size<90*1024**2:files[archive.name]=archive;parts=[dict(name=archive.name,**info(archive))]
 else:
  with archive.open('rb') as f:
   k=1
   while True:
    block=f.read(48*1024**2)
    if not block:break
    part=archive.with_name(archive.name+'.part%03d'%k)
    with part.open('xb') as out:out.write(block)
    files[part.name]=part;parts.append(dict(name=part.name,**info(part)));k+=1
 dump(H/'ARCHIVE_PARTS.json',dict(archive=archive.name,archive_sha256=receipt['sha256'],bytes=receipt['bytes'],parts=parts));files['ARCHIVE_PARTS.json']=H/'ARCHIVE_PARTS.json'
 for p in files.values():need(p.is_file(),'publication file '+str(p));scan(p)
 dump(H/'PUBLICATION_SCOPE.json',{n:info(p) for n,p in files.items()});files['PUBLICATION_SCOPE.json']=H/'PUBLICATION_SCOPE.json'
 repo=R/'intercambio/post48_plan_20260927_06.git'
 def git(*args,data=None):
  p=subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,capture_output=True,timeout=60,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_NO_LAZY_FETCH='1'))
  need(p.returncode==0,p.stderr.decode(errors='replace'));return p.stdout.decode().strip()
 need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','remote')
 git('fetch','--depth=1','--filter=blob:none','origin','main');parent=git('rev-parse','FETCH_HEAD');leaf=[]
 for name,p in files.items():leaf.append('100644 blob '+git('hash-object','-w','--',str(p))+'\t'+name)
 tree=git('mktree','--missing',data=('\n'.join(leaf)+'\n').encode());reviews=git('ls-tree',parent+':reviews').splitlines();need(not any(s.endswith('\t'+LEAF) for s in reviews),'review exists');reviews.append('040000 tree '+tree+'\t'+LEAF)
 rt=git('mktree','--missing',data=('\n'.join(reviews)+'\n').encode());entries=git('ls-tree',parent).splitlines();entries=[('040000 tree '+rt+'\treviews') if s.endswith('\treviews') else s for s in entries];root=git('mktree','--missing',data=('\n'.join(entries)+'\n').encode());commit=git('commit-tree',root,'-p',parent,data=b'Publish bounded temporal-transfer campaign56\n')
 changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines();need(changed and all(s.startswith(PREFIX+'/') for s in changed),'publication scope')
 git('push','origin',commit+':refs/heads/'+LEAF)
 dump(H/'PUBLICATION_COMMIT.json',dict(commit=commit,parent=parent,branch=LEAF,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX))
 download=X/'recibos'/(STEM+'_DESCARGADO.zip')
 with download.open('xb') as out:
  for part in parts:
   digest=hashlib.sha256();size=0;url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+part['name']
   with urllib.request.urlopen(url,timeout=45) as remote:
    for b in iter(lambda:remote.read(1048576),b''):digest.update(b);size+=len(b);out.write(b)
   need(digest.hexdigest()==part['sha256'] and size==part['bytes'],'download part changed')
 need(sha(download)==receipt['sha256'],'remote archive changed');proof=extract_verify(download,'REMOTO');child1=resource.getrusage(resource.RUSAGE_CHILDREN)
 need(disk_current()<10*1024**3,'aggregate disk budget')
 result=dict(commit=commit,parent=parent,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,archive_sha256=receipt['sha256'],verification=proof,CPU_s=time.process_time()-start,child_CPU_s=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime,wall_s=time.monotonic()-wall,aggregate_new_disk_bytes=disk_current())
 dump(H/'REMOTE_DELIVERY.json',result);print(json.dumps({k:result[k] for k in ['commit','url','CPU_s','child_CPU_s','aggregate_new_disk_bytes']}),flush=True)
def resume_local():
 start=time.process_time();child0=resource.getrusage(resource.RUSAGE_CHILDREN);wall=time.monotonic()
 archive=X/'salida'/(STEM+'.zip');folder=X/'recibos'/(STEM+'_LOCAL')/STEM
 need(archive.is_file() and folder.is_dir() and not (H/'LOCAL_DELIVERY.json').exists(),'existing clean extraction required')
 run=subprocess.run([sys.executable,'-B','-O','check_delivery56.py','--corruptions'],cwd=folder,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,encoding='utf-8',timeout=65)
 need(run.returncode==0,run.stdout+'\n'+run.stderr);proof={'extraction':str(folder),'verification':json.loads(run.stdout),'new_CNS_ms':0,'same_untouched_clean_extraction_after_launcher_limit_repair':True}
 child1=resource.getrusage(resource.RUSAGE_CHILDREN)
 receipt=dict(path=str(archive),**info(archive),files=len(json.loads((H/'MANIFEST.json').read_text())['files'])+1,verification=proof,CPU_s=time.process_time()-start,child_CPU_s=child1.ru_utime+child1.ru_stime-child0.ru_utime-child0.ru_stime,wall_s=time.monotonic()-wall,aggregate_new_disk_bytes=disk_current(),first_pack_accounting=json.loads((H/'PACK_FAILURE_ACCOUNTING.json').read_text()))
 dump(H/'LOCAL_DELIVERY.json',receipt);print(json.dumps({k:receipt[k] for k in ['path','bytes','sha256','files','CPU_s','child_CPU_s','aggregate_new_disk_bytes']}),flush=True)
if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CPU,(20,125));p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');p.add_argument('--resume-local',action='store_true');a=p.parse_args()
 need(a.publish!=a.resume_local,'choose exactly one closure action; do not repack')
 publish() if a.publish else resume_local()

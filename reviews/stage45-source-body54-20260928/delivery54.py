"""Preserve full scientific evidence; publish only the explicit compact review."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,sys,os,re,time,resource,urllib.request,argparse
H=Path(__file__).resolve().parent;R=H.parents[1]
X=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO');STEM='ETAPA45_FUENTE_CUERPO_20260928_54';LEAF='stage45-source-body54-20260928';PREFIX='reviews/'+LEAF
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')

def need(x,m):
 if not x:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def scan(p):
 if p.suffix in ['.py','.json','.md','.log','.txt','.cu','.cpp','.hpp','.html','.diff']:
  with p.open('rb') as f:
   tail=b''
   for b in iter(lambda:f.read(1024**2),b''):
    need(not SECRET.search(tail+b),'Credential pattern in '+str(p));tail=b[-200:]
def info(p):return {'bytes':p.stat().st_size,'sha256':sha(p)}
def member_files():
 files={}
 for p in H.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(H)
  if any(x in rel.parts for x in ['__pycache__','cache','continuidad_previa','final_state']):continue
  if p.suffix in ['.pyc','.tmp','.zip'] or p.name in ['MANIFEST.json','LOCAL_DELIVERY.json','REMOTE_DELIVERY.json','PUBLICATION_SCOPE.json','pack.log','publish.log','STATUS.json','QUEUE_STATUS.json']:continue
  files[str(rel)]=p
 return files

def extract_verify(path,tag):
 d=X/'recibos'/(STEM+'_'+tag);d.mkdir(exist_ok=False)
 with zipfile.ZipFile(path) as z:
  need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'archive paths');z.extractall(d)
 root=d/STEM
 run=subprocess.run([sys.executable,'-B','-O','check_delivery54.py','--corruptions'],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=65)
 need(run.returncode==0,run.stdout+'\n'+run.stderr)
 return {'extraction':str(root),'verification':json.loads(run.stdout),'new_CNS_ms':0}

def pack():
 start=time.process_time();need(json.loads((H/'repair02/QUEUE_RESULT.json').read_text())['status']=='COMPLETE','incomplete campaign')
 files=member_files()
 # Historical transitive inventory, actually imported modules, and new owners.
 inventories=[R/'campanas/etapa45_composicion_20260927_48/SOURCES.json',H/'SOURCES.json',H/'repair02/SOURCES.json']+list((H/'repair02').glob('*/EXECUTED_SOURCES.json'))
 sources={}
 for index in inventories:
  for p,digest in json.loads(index.read_text()).items():
   if p in sources:need(sources[p]==digest,'source identity conflict')
   sources[p]=digest
 extra=[R/'campanas/etapa45_postwind_diagnosis_20260925_41/inputs/body.mjb',R/'campanas/etapa45_postwind_diagnosis_20260925_41/inputs/reference.npz']
 sources.update({str(p):sha(p) for p in extra})
 full={};source_index={}
 for name,digest in sources.items():
  p=Path(name);need(sha(p)==digest,'source changed '+name);scan(p);logical='sources/'+digest+p.suffix;full[logical]=p;source_index[name]=logical
 dump(H/'SOURCE_INDEX.json',source_index);files['SOURCE_INDEX.json']=H/'SOURCE_INDEX.json'
 for p in H.rglob('*'):
  if not p.is_file():continue
  rel=p.relative_to(H)
  if any(x in rel.parts for x in ['__pycache__','cache','continuidad_previa']):continue
  if p.suffix in ['.pyc','.tmp','.zip'] or p.name in ['pack.log','publish.log']:continue
  full['campaign54/'+str(rel)]=p
 for p in (R/'campanas/etapa45_composicion_20260927_48/sham/final_state').iterdir():
  if p.is_file():full['initial48sham/'+p.name]=p
 manifest={}
 for name,p in full.items():scan(p);manifest[name]=info(p)
 archive=X/'salida'/(STEM+'_ESTADOS_Y_FUENTES.zip')
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
  for name,p in full.items():
   method=zipfile.ZIP_STORED if p.suffix in ['.npz','.npy','.png','.so'] else zipfile.ZIP_DEFLATED
   z.write(p,name,compress_type=method,compresslevel=None if method==zipfile.ZIP_STORED else 1)
  z.writestr('FULL_MANIFEST.json',json.dumps({'files':manifest,'scope':'Saved initial/final scientific states and frozen transitive sources. Analysis is portable; GPU restoration and absolute-path simulation loader are not qualified as portable.'},indent=2))
 with zipfile.ZipFile(archive) as z:
  for name,item in manifest.items():
   digest=hashlib.sha256();size=0
   with z.open(name) as f:
    for b in iter(lambda:f.read(1024**2),b''):digest.update(b);size+=len(b)
   need(digest.hexdigest()==item['sha256'] and size==item['bytes'],'full archive member '+name)
 dump(H/'FULL_CAPSULE.json',dict(path=str(archive),**info(archive),members_verified=len(manifest),CPU_s=time.process_time()-start,scope='Preservation checked by streaming every member; no new GPU resume.'))
 files['FULL_CAPSULE.json']=H/'FULL_CAPSULE.json'
 for p in files.values():scan(p)
 need(not (H/'MANIFEST.json').exists(),'preserve manifest')
 dump(H/'MANIFEST.json',{'schema':'source_body54_compact_v1','files':{n:info(p) for n,p in files.items()},'scope':'Portable reconstruction of all six recorded arms and qualifications; complete states in local full capsule.'})
 compact=X/'salida'/(STEM+'.zip')
 with zipfile.ZipFile(compact,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
  for n,p in {**files,'MANIFEST.json':H/'MANIFEST.json'}.items():z.write(p,STEM+'/'+n)
 receipt={'path':str(compact),**info(compact),'files':len(files)+1,'verification':extract_verify(compact,'LOCAL'),'CPU_s':time.process_time()-start}
 dump(H/'LOCAL_DELIVERY.json',receipt);print(json.dumps({k:receipt[k] for k in ['path','bytes','sha256','files','CPU_s']}),flush=True)

def publish():
 start=time.process_time();receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());archive=Path(receipt['path']);need(sha(archive)==receipt['sha256'],'compact changed');need(archive.stat().st_size<90*1024**2,'compact publication size')
 names=['RESULTADOS.md','COMPARACION.png','DECISION.md','PLAN.md','PILOT_PLAN.json','FREEZE.json','REPAIR_PLAN.json','ANALYSIS_PLAN.md','CORRUPTION_TESTS.json','MANIFEST.json','LOCAL_DELIVERY.json','FULL_CAPSULE.json','verify54.py','report54.py','check_delivery54.py','delivery54.py']
 files={n:H/n for n in names};files[archive.name]=archive;files['RESULTADOS.json']=H/'repair02/RESULTADOS.json';files['CONTINUACION_Y_GEMINI.md']=H/'research11/INFORME.md'
 for p in files.values():need(p.is_file(),'publication file '+str(p));scan(p)
 dump(H/'PUBLICATION_SCOPE.json',{n:info(p) for n,p in files.items()});files['PUBLICATION_SCOPE.json']=H/'PUBLICATION_SCOPE.json'
 repo=R/'intercambio/post48_plan_20260927_06.git'
 def git(*args,data=None):
  p=subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,capture_output=True,timeout=60,env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GIT_NO_LAZY_FETCH='1'))
  need(p.returncode==0,p.stderr.decode(errors='replace'));return p.stdout.decode().strip()
 need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','remote')
 git('fetch','--depth=1','--filter=blob:none','origin','main');parent=git('rev-parse','FETCH_HEAD')
 leaf=[]
 for name,p in files.items():leaf.append('100644 blob '+git('hash-object','-w','--',str(p))+'\t'+name)
 tree=git('mktree','--missing',data=('\n'.join(leaf)+'\n').encode());reviews=git('ls-tree',parent+':reviews').splitlines();need(not any(s.endswith('\t'+LEAF) for s in reviews),'review exists');reviews.append('040000 tree '+tree+'\t'+LEAF)
 reviews_tree=git('mktree','--missing',data=('\n'.join(reviews)+'\n').encode());entries=git('ls-tree',parent).splitlines();entries=[('040000 tree '+reviews_tree+'\treviews') if s.endswith('\treviews') else s for s in entries];root=git('mktree','--missing',data=('\n'.join(entries)+'\n').encode());commit=git('commit-tree',root,'-p',parent,data=b'Publish bounded spatial source/body campaign54 and adviser comparison\n')
 changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines();need(changed and all(s.startswith(PREFIX+'/') for s in changed),'publication scope')
 git('push','origin',commit+':refs/heads/'+LEAF)
 url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+archive.name;download=X/'recibos'/(STEM+'_DESCARGADO.zip')
 with urllib.request.urlopen(url,timeout=45) as remote,download.open('xb') as out:
  for b in iter(lambda:remote.read(1024**2),b''):out.write(b)
 need(sha(download)==receipt['sha256'],'remote archive changed')
 result={'commit':commit,'parent':parent,'url':'https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,'archive_sha256':receipt['sha256'],'verification':extract_verify(download,'REMOTO'),'CPU_s':time.process_time()-start}
 dump(H/'REMOTE_DELIVERY.json',result);print(json.dumps({k:result[k] for k in ['commit','url','CPU_s']}),flush=True)

if __name__=='__main__':
 resource.setrlimit(resource.RLIMIT_CPU,(115,120));p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args();publish() if a.publish else pack()

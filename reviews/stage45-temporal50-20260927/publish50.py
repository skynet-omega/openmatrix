from pathlib import Path
import os,subprocess,json,hashlib,urllib.request,zipfile,time,sys
H=Path(__file__).resolve().parent;ROOT=H.parents[1];REPO=ROOT/'intercambio/post48_plan_20260927_06.git';parent='f4b746c8fe56aee7e7ef6df17ecb5171e27e7311';prefix='reviews/stage45-temporal50-20260927';branch='stage45-temporal50-20260927'
index=REPO/'temporal50.index'
if index.exists():raise ValueError('Immutable index')
env=dict(os.environ,GIT_INDEX_FILE=str(index),GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
def git(*a,input=None):return subprocess.run(['git','--git-dir='+str(REPO),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*a],input=input,env=env,check=True,capture_output=True,timeout=120).stdout.decode().strip()
if git('remote','get-url','origin')!='git@github.com:skynet-omega/openmatrix.git':raise ValueError('Wrong destination')
m=json.loads((H/'MANIFEST.json').read_text());receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());zpath=Path(receipt['zip'])
if zpath.stat().st_size>95*1024**2:raise ValueError('Publication size bound')
files=[(n,H/n) for n in list(m['files'])+['MANIFEST.json']]+[(zpath.name,zpath)]
git('read-tree',parent)
for name,p in files:
 if name in m['files'] and hashlib.sha256(p.read_bytes()).hexdigest()!=m['files'][name]['sha256']:raise ValueError('Modified source '+name)
 oid=git('hash-object','-w','--',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+','+prefix+'/'+name)
tree=git('write-tree','--missing-ok');commit=git('commit-tree',tree,'-p',parent,input=b'Publish eight-arm temporal50 diagnostic; stages4/5 remain open\n')
paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
if not paths or any(not p.startswith(prefix+'/') for p in paths):raise ValueError('Publication scope')
git('push','origin',commit+':refs/heads/'+branch)
u='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+prefix+'/'+zpath.name
with urllib.request.urlopen(u,timeout=60) as r:b=r.read(zpath.stat().st_size+1)
if len(b)!=receipt['bytes'] or hashlib.sha256(b).hexdigest()!=receipt['sha256']:raise ValueError('Remote differs')
folder=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/recibos/ETAPA45_TEMPORAL_20260927_50_REMOTO');folder.mkdir(exist_ok=False);p=folder/zpath.name;p.write_bytes(b)
with zipfile.ZipFile(p) as z:
 if any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()):raise ValueError('ZIP path')
 z.extractall(folder)
root=folder/zpath.stem;checks=[]
for a in [['verify_complete50.py'],['-O','verify_complete50.py'],['-O','test_analysis50.py','--real'],['-O','test_law_supplement50.py'],['-O','verify_supplements50.py']]:
 r=subprocess.run([sys.executable,*a],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=60)
 if r.returncode:raise ValueError(r.stderr)
 checks.append(json.loads(r.stdout))
out=dict(commit=commit,branch=branch,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+prefix,zip=str(zpath),sha256=receipt['sha256'],bytes=len(b),checks=checks,scope=receipt['scope']);(H/'REMOTE_DELIVERY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['url','commit','bytes','scope']}))

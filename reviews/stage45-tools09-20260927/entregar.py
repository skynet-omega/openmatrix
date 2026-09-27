"""Package and publish only this research's explicit evidence manifest."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time, urllib.request, zipfile
H=Path(__file__).resolve().parent; R=H.parents[1]
BASE=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
ZIPNAME='ETAPA45_HERRAMIENTAS_20260927_09.zip'
PREFIX='reviews/stage45-tools09-20260927';BRANCH='stage45-tools09-20260927'
def need(x,m):
    if not x:raise ValueError(m)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def verify_archive(path,tag):
    d=BASE/'recibos'/('ETAPA45_HERRAMIENTAS_20260927_09_'+tag);d.mkdir(exist_ok=False)
    with zipfile.ZipFile(path) as z:
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'Zip paths')
        z.extractall(d)
    root=d/H.name;m=json.loads((root/'MANIFEST.json').read_text())
    for name,v in m['files'].items():need(sha(root/name)==v['sha256'],'File digest '+name)
    checks=[]
    for args in [['-O','verificar_criba.py'],['-O','aporte_motor/verify_projection.py'],['-O','criba_instrumentos.py','--verify']]:
        p=subprocess.run([sys.executable,*args],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=45)
        need(p.returncode==0,p.stderr);checks.append(json.loads(p.stdout))
    return dict(extraction=str(root),checks=checks)
def local():
    need(not (H/'MANIFEST.json').exists(),'Preserve manifest')
    files={}
    for p in sorted(H.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or 'fuentes' in p.relative_to(H).parts:continue
        name=str(p.relative_to(H));files[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
    dump(H/'MANIFEST.json',{'scope':'Own research, advice and portable recorded evidence. No CNS simulation or outside sources required for numerical reproduction. Third-party README retained locally, not mirrored.','files':files})
    path=BASE/'salida'/ZIPNAME;need(not path.exists(),'Preserve ZIP')
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in list(files)+['MANIFEST.json']:z.write(H/name,arcname=H.name+'/'+name)
    out={'zip':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'files':len(files)+1,'verification':verify_archive(path,'LOCAL')}
    dump(H/'LOCAL_DELIVERY.json',out);print(json.dumps({k:out[k] for k in ['zip','bytes','sha256','files']}))
def publish():
    receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());path=Path(receipt['zip']);need(sha(path)==receipt['sha256'],'ZIP changed')
    repo=R/'intercambio/post48_plan_20260927_06.git';idx=repo/'tools09.index';need(not idx.exists(),'Preserve publication index')
    env=dict(os.environ,GIT_INDEX_FILE=str(idx),GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
    def git(*args,data=None):
        return subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,env=env,capture_output=True,check=True,timeout=60).stdout.decode().strip()
    need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','Destination')
    parent='ae782f9e1c7976f4bb31e54783076e7a83a4042e';git('read-tree',parent)
    m=json.loads((H/'MANIFEST.json').read_text());files=[(n,H/n) for n in m['files']]+[(n,H/n) for n in ['MANIFEST.json','LOCAL_DELIVERY.json']]+[(ZIPNAME,path)]
    dump(H/'PUBLICATION_SCOPE.json',{'repository':'skynet-omega/openmatrix','prefix':PREFIX,'files':{n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files}})
    files.append(('PUBLICATION_SCOPE.json',H/'PUBLICATION_SCOPE.json'))
    for n,p in files:
        if n in m['files']:need(sha(p)==m['files'][n]['sha256'],'Changed source '+n)
        oid=git('hash-object','-w','--',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+','+PREFIX+'/'+n)
    tree=git('write-tree','--missing-ok');commit=git('commit-tree',tree,'-p',parent,data=b'Publish bounded causal tools review and retrospective comparison09\n')
    changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines();need(changed and all(n.startswith(PREFIX+'/') for n in changed),'Write scope')
    git('push','origin',commit+':refs/heads/'+BRANCH)
    url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+ZIPNAME
    with urllib.request.urlopen(url,timeout=30) as f:b=f.read(receipt['bytes']+1)
    need(len(b)==receipt['bytes'] and hashlib.sha256(b).hexdigest()==receipt['sha256'],'Remote identity')
    downloaded=H/'download_verificado.zip';downloaded.write_bytes(b)
    out={'commit':commit,'url':'https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,'branch':BRANCH,'bytes':len(b),'sha256':receipt['sha256'],'verification':verify_archive(downloaded,'REMOTO')}
    dump(H/'REMOTE_DELIVERY.json',out);print(json.dumps({k:out[k] for k in ['url','commit','bytes','sha256']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    publish() if args.publish else local()

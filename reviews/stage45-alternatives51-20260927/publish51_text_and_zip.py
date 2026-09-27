"""Publish the manifest-scoped compact campaign to the explicitly authorized repository."""
from pathlib import Path
import os,subprocess,json,hashlib,urllib.request,zipfile,sys,time
H=Path(__file__).resolve().parent;ROOT=H.parents[1];REPO=ROOT/'intercambio/post48_plan_20260927_06.git'
PARENT='fa91ceae43b3385b63cf795dedcbb470691de090';PREFIX='reviews/stage45-alternatives51-20260927';BRANCH='stage45-alternatives51-20260927'
def need(ok,message):
 if not ok:raise ValueError(message)

def main():
 start=time.monotonic();publication=json.loads((H/'PUBLICATION_SCOPE.json').read_text());index=REPO/'alternatives51.index';need(not index.exists(),'Immutable publication index')
 env=dict(os.environ,GIT_INDEX_FILE=str(index),GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
 def git(*a,input=None):return subprocess.run(['git','--git-dir='+str(REPO),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*a],input=input,env=env,check=True,capture_output=True,timeout=120).stdout.decode().strip()
 need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','Wrong destination')
 m=json.loads((H/'MANIFEST.json').read_text());receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());zpath=Path(receipt['zip'])
 need(zpath.stat().st_size<95*1024**2,'Publication size bound')
 need(hashlib.sha256(zpath.read_bytes()).hexdigest()==receipt['sha256'],'Compact archive changed')
 files=[(n,H/n) for n in list(m['files'])+['MANIFEST.json'] if Path(n).suffix!='.npz']+[(zpath.name,zpath)]
 for n in ['PUBLICATION_SCOPE.json','build_full_capsule51_v2.py','repair_full_packing51.py','PACKING_REPAIR.json','PACKING_DIAGNOSTIC.json','RAW_DATA_IN_ZIP.md','publish51_text_and_zip.py']:files.append((n,H/n))
 for n in ['LOCAL_DELIVERY.json','FULL_CAPSULE.json']:
  if (H/n).exists():files.append((n,H/n))
 git('read-tree',PARENT)
 for name,p in files:
  if name in publication['supplement_hashes']:need(hashlib.sha256(p.read_bytes()).hexdigest()==publication['supplement_hashes'][name],'Supplement changed')
  if name in m['files']:need(hashlib.sha256(p.read_bytes()).hexdigest()==m['files'][name]['sha256'],'Modified source '+name)
  oid=git('hash-object','-w','--',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+','+PREFIX+'/'+name)
 tree=git('write-tree','--missing-ok');commit=git('commit-tree',tree,'-p',PARENT,input=b'Publish bounded alternatives51 comparison and recorded evidence\n')
 paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
 need(paths and all(p.startswith(PREFIX+'/') for p in paths),'Publication scope')
 git('push','origin',commit+':refs/heads/'+BRANCH)
 u='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+zpath.name
 with urllib.request.urlopen(u,timeout=60) as r:b=r.read(zpath.stat().st_size+1)
 need(len(b)==receipt['bytes'] and hashlib.sha256(b).hexdigest()==receipt['sha256'],'Remote differs')
 folder=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/recibos/ETAPA45_ALTERNATIVAS_20260927_51_REMOTO');folder.mkdir(exist_ok=False);p=folder/zpath.name;p.write_bytes(b)
 with zipfile.ZipFile(p) as z:
  need(not any('..' in Path(n).parts or Path(n).is_absolute() for n in z.namelist()),'ZIP path');z.extractall(folder)
 root=folder/zpath.stem;checks=[]
 for a in [['verify_complete51.py'],['-O','verify_complete51.py'],['-O','test_corruptions51.py']]:
  r=subprocess.run([sys.executable,*a],cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=120)
  need(r.returncode==0,r.stderr);checks.append(json.loads(r.stdout))
 out=dict(commit=commit,branch=BRANCH,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,zip=str(zpath),sha256=receipt['sha256'],bytes=len(b),checks=checks,scope=receipt['scope'],wall_s=time.monotonic()-start)
 (H/'REMOTE_DELIVERY.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['url','commit','bytes','scope','wall_s']}))

if __name__=='__main__':main()

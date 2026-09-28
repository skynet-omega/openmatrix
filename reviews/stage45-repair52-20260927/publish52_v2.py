"""Manifest-scoped compact delivery and verified publication to OpenMatrix."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile

H=Path(__file__).resolve().parent
ROOT=H.parents[1]
EXCHANGE=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
STEM='ETAPA45_REPARACION_20260927_52'
PREFIX='reviews/stage45-repair52-20260927'
BRANCH='stage45-repair52-20260927'
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify_zip(path,tag):
    folder=EXCHANGE/'recibos'/(STEM+'_'+tag)
    folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(path) as z:
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'Unsafe archive path')
        z.extractall(folder)
    root=folder/STEM
    runs=[]
    # Independent verifier has its own canonical node mapping inside the ZIP.
    for args in [['-B','-O','verify_complete52.py'],['-B','-O','aporte_motor52/verify52.py','science','--root',str(root),'--out',str(root/'aporte_motor52/RECOMPUTE_DELIVERY.json')]]:
        command=[sys.executable,*args]
        run=subprocess.run(command,cwd=root,env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
        need(run.returncode==0,run.stdout+'\n'+run.stderr)
        runs.append(dict(command=command,output=run.stdout))
    return dict(extraction=str(root),checks=runs,scope='Recorded-data reconstruction only; CNS not rerun')


def compact():
    need(not (H/'MANIFEST.json').exists(),'Immutable manifest')
    plan=json.loads((H/'PILOT_PLAN.json').read_text())
    queue=json.loads((H/'QUEUE_RESULT.json').read_text())
    need(queue['status']=='COMPLETE','Only complete campaign can be delivered by this path')
    files={}
    for f in sorted(H.rglob('*')):
        if not f.is_file() or '__pycache__' in f.parts:continue
        rel=f.relative_to(H)
        if len(rel.parts)>1:
            if rel.parts[0] in plan['arms'] or rel.parts[0].startswith('qual_'):
                if len(rel.parts)>2:continue
            elif rel.parts[0] not in ('donors','aporte_motor52'):continue
        if f.suffix in ('.zip','.partial','.pyc') or f.name in ('STATUS.json','QUEUE_STATUS.json','LOCAL_DELIVERY.json','REMOTE_DELIVERY.json'):continue
        if f.suffix in ('.py','.json','.md','.txt','.log','.cu','.patch'):
            need(not SECRET.search(f.read_bytes()),'Credential pattern in explicit publication member')
        files[str(rel)]=dict(bytes=f.stat().st_size,sha256=sha(f))
    dump(H/'MANIFEST.json',dict(schema='repair52_compact_manifest_v1',files=files,
        scope='Complete recorded arrays for all ten arms and seven qualifications, exact51 projections, source code and portable verifiers. Full runtime dependencies/checkpoints in separate full capsule.'))
    out=EXCHANGE/'salida'/(STEM+'.zip')
    need(not out.exists(),'Preserve compact ZIP')
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in list(files)+['MANIFEST.json']:z.write(H/name,STEM+'/'+name)
    receipt=dict(zip=str(out),bytes=out.stat().st_size,sha256=sha(out),files=len(files)+1,
                 verification=verify_zip(out,'LOCAL'))
    dump(H/'LOCAL_DELIVERY.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['zip','bytes','sha256','files']}))


def publish():
    receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text())
    archive=Path(receipt['zip'])
    need(sha(archive)==receipt['sha256'],'Archive changed')
    manifest=json.loads((H/'MANIFEST.json').read_text())
    files=[(n,H/n) for n in manifest['files'] if Path(n).suffix not in ('.npz','.npy')]
    files += [(n,H/n) for n in ['MANIFEST.json','LOCAL_DELIVERY.json','FULL_CAPSULE.json'] if (H/n).exists()]
    supplements=['pack_full52.py','CAPSULE_ENCODING.json','CAPSULE_ATTEMPT1_FAILED.json',
        'recover_capsule52.py','build_full_capsule52_v2.py','pack_full52_v2.py',
        'CAPSULE_ENCODING_V2.json','COMPRESION_Y_ENTREGA.md','publish52_v2.py',
        'DELIVERY_CONTINUATION.json','cleanup_delivery52.py']
    for name in supplements:
        path=H/name
        need(path.is_file(),'Missing delivery supplement '+name)
        need(not SECRET.search(path.read_bytes()),'Credential pattern in delivery supplement')
        files.append((name,path))

    if archive.stat().st_size < 90*1024**2:
        parts=[archive]
    else:
        parts=[]
        with archive.open('rb') as f:
            i=1
            while True:
                data=f.read(80*1024**2)
                if not data:break
                part=archive.with_name(archive.name+'.part'+str(i).zfill(2))
                need(not part.exists(),'Preserve archive part')
                part.write_bytes(data);parts.append(part);i+=1
    files += [(p.name,p) for p in parts]
    for n,p in files:
        if n in manifest['files']:need(sha(p)==manifest['files'][n]['sha256'],'Frozen delivery file changed: '+n)
    publication=dict(repository='skynet-omega/openmatrix',prefix=PREFIX,
                     files={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in files},
                     archive_parts=[p.name for p in parts],archive_sha256=receipt['sha256'])
    dump(H/'PUBLICATION_SCOPE.json',publication)
    files.append(('PUBLICATION_SCOPE.json',H/'PUBLICATION_SCOPE.json'))
    repo=ROOT/'intercambio/post48_plan_20260927_06.git'
    index=repo/'repair52.index'
    need(not index.exists(),'Preserve publication transaction')
    env=dict(os.environ,GIT_INDEX_FILE=str(index),GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
    def git(*args,data=None):
        p=subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,env=env,capture_output=True,timeout=60)
        need(p.returncode==0,p.stderr.decode(errors='replace'))
        return p.stdout.decode().strip()
    need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','Wrong remote')
    parent='58adffb9e9ed3a71f08632a48a03bf72ea58774d'
    git('read-tree',parent)
    object_ids=[]
    for name,p in files:
        oid=git('hash-object','-w','--',str(p))
        object_ids.append(oid)
        git('update-index','--add','--cacheinfo','100644,'+oid+','+PREFIX+'/'+name)
    tree=git('write-tree','--missing-ok')
    commit=git('commit-tree',tree,'-p',parent,data=b'Publish corrected units replication52 and expanded neural observations\n')
    changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    need(changed and all(p.startswith(PREFIX+'/') for p in changed),'Publication write scope')
    git('push','origin',commit+':refs/heads/'+BRANCH)
    downloaded=EXCHANGE/'recibos'/(STEM+'_DESCARGADO.zip')
    need(not downloaded.exists(),'Preserve remote download')
    with downloaded.open('xb') as out:
        for part in parts:
            url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+part.name
            h=hashlib.sha256();length=0
            with urllib.request.urlopen(url,timeout=45) as r:
                while True:
                    b=r.read(4*1024**2)
                    if not b:break
                    out.write(b);h.update(b);length+=len(b)
            need(h.hexdigest()==sha(part) and length==part.stat().st_size,'Remote part identity')
    need(sha(downloaded)==receipt['sha256'] and downloaded.stat().st_size==receipt['bytes'],'Downloaded archive identity')
    object_sizes=git('cat-file','--batch-check=%(objectname) %(objectsize:disk)',data=('\n'.join(sorted(set(object_ids)))+'\n').encode()).splitlines()
    result=dict(commit=commit,branch=BRANCH,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,
                archive_sha256=receipt['sha256'],archive_bytes=receipt['bytes'],parts=[p.name for p in parts],
                publication_blob_disk_bytes=sum(int(line.split()[1]) for line in object_sizes),
                verification=verify_zip(downloaded,'REMOTO'))
    dump(H/'REMOTE_DELIVERY.json',result)
    print(json.dumps({k:result[k] for k in ['commit','url','archive_bytes','archive_sha256','parts']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');a=p.parse_args()
    publish() if a.publish else compact()

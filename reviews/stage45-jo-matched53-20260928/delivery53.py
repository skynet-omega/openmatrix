"""Bounded scientific archives and explicit OpenMatrix publication."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import resource
import urllib.request
import zipfile
from verify53 import sha,need

H=Path(__file__).resolve().parent
ROOT=H.parents[1]
EXCHANGE=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
STEM='ETAPA45_JO_EMPAREJADO_20260928_53'
PREFIX='reviews/stage45-jo-matched53-20260928'
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def compact_members():
    plan=json.loads((H/'PILOT_PLAN.json').read_text());files={}
    for f in sorted(H.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(H)
        if any(x in rel.parts for x in ['__pycache__','cache','continuidad_previa','source_snapshot']):continue
        if len(rel.parts)>1:
            if rel.parts[0] in plan['arms'] or rel.parts[0].startswith('qual_'):
                if len(rel.parts)>2:continue
            elif rel.parts[0]!='reference52':continue
        if f.suffix in ['.zip','.tmp','.pyc'] or f.name in ['STATUS.json','QUEUE_STATUS.json','LOCAL_DELIVERY.json','REMOTE_DELIVERY.json','MANIFEST.json','pack.log','publish.log']:continue
        if f.suffix in ['.py','.json','.md','.log','.txt','.cu']:
            need(not SECRET.search(f.read_bytes()),'Credential pattern in member '+str(rel))
        files[str(rel)]=dict(bytes=f.stat().st_size,sha256=sha(f))
    return files


def verify_zip(path,tag):
    destination=EXCHANGE/'recibos'/(STEM+'_'+tag)
    destination.mkdir(exist_ok=False)
    with zipfile.ZipFile(path) as z:
        need(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist()),'Unsafe archive')
        z.extractall(destination)
    root=destination/STEM
    run=subprocess.run([sys.executable,'-B','-O','check_delivery53.py','--corruptions'],cwd=root,
        env=dict(os.environ,PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),
        capture_output=True,text=True,timeout=75)
    need(run.returncode==0,run.stdout+'\n'+run.stderr)
    return dict(extraction=str(root),result=json.loads(run.stdout),new_CNS_ms=0)


def pack():
    start=time.process_time();queue=json.loads((H/'QUEUE_RESULT.json').read_text())
    need(queue['status']=='COMPLETE','Campaign incomplete')
    # Scientific states and every frozen local source are retained in the full archive.
    full_members={}
    for f in H.rglob('*'):
        if not f.is_file():continue
        rel=f.relative_to(H)
        if any(x in rel.parts for x in ['cache','__pycache__','continuidad_previa']):continue
        if f.suffix in ['.tmp','.pyc','.zip'] or f.name in ['pack.log','publish.log']:continue
        full_members['campaign53/'+str(rel)]=f
    source=ROOT/'campanas/etapa45_composicion_20260927_48/sham/final_state'
    for f in source.iterdir():
        if f.is_file():full_members['initial48sham/'+f.name]=f
    frozen=json.loads((H/'PILOT_SOURCES.json').read_text())
    source_index={}
    for path,digest in frozen.items():
        f=Path(path);need(sha(f)==digest,'Frozen source changed '+path)
        logical='sources/'+digest+f.suffix
        full_members[logical]=f
        source_index[path]=logical
    dump(H/'SOURCE_INDEX.json',source_index)
    full_members['campaign53/SOURCE_INDEX.json']=H/'SOURCE_INDEX.json'
    manifest={name:dict(bytes=f.stat().st_size,sha256=sha(f)) for name,f in full_members.items()}
    full=EXCHANGE/'salida'/(STEM+'_ESTADOS_Y_FUENTES.zip')
    with zipfile.ZipFile(full,'x',allowZip64=True,compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,f in full_members.items():
            method=zipfile.ZIP_STORED if f.suffix in ['.npz','.png','.npy','.so'] else zipfile.ZIP_DEFLATED
            z.write(f,name,compress_type=method,compresslevel=None if method==zipfile.ZIP_STORED else 1)
        z.writestr('FULL_MANIFEST.json',json.dumps(dict(files=manifest,scope='All saved initial/final scientific states, raw records and frozen sources. Historical absolute-path loader and GPU cold resume are not qualified as portable.'),indent=2))
    checked=0
    with zipfile.ZipFile(full) as z:
        for name,item in manifest.items():
            digest=hashlib.sha256();size=0
            with z.open(name) as f:
                for block in iter(lambda:f.read(1024**2),b''):digest.update(block);size+=len(block)
            need(digest.hexdigest()==item['sha256'] and size==item['bytes'],'Full archive member differs')
            checked+=1
    dump(H/'FULL_CAPSULE.json',dict(path=str(full),bytes=full.stat().st_size,sha256=sha(full),members_verified=checked,
        CPU_s=time.process_time()-start,scope='Complete saved scientific states/source preservation; streaming verification, no GPU resume claim'))
    files=compact_members();need(not (H/'MANIFEST.json').exists(),'Preserve compact manifest')
    dump(H/'MANIFEST.json',dict(schema='jo_matched53_compact_v1',files=files,scope='All recorded diagnostic arrays and qualification references; portable analysis. Full scientific states are in the local full archive.'))
    archive=EXCHANGE/'salida'/(STEM+'.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name in list(files)+['MANIFEST.json']:z.write(H/name,STEM+'/'+name)
    local=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)+1,
        verification=verify_zip(archive,'LOCAL'),CPU_s=time.process_time()-start)
    dump(H/'LOCAL_DELIVERY.json',local)
    print(json.dumps({k:local[k] for k in ['path','bytes','sha256','files','CPU_s']}))


def publish():
    start=time.process_time();receipt=json.loads((H/'LOCAL_DELIVERY.json').read_text());archive=Path(receipt['path'])
    need(sha(archive)==receipt['sha256'],'Archive changed');need(archive.stat().st_size<90*1024**2,'Compact exceeds one-file publication bound')
    names=['RESULTADOS.md','COMPARACION.png','DECISION.md','PLAN.md','PILOT_PLAN.json','PILOT_FREEZE.json','RESULTADOS.json',
        'QUALIFICATION.json','CORRUPTION_TESTS.json','MANIFEST.json','LOCAL_DELIVERY.json','FULL_CAPSULE.json',
        'jo_matched.py','run_pilot.py','run_queue.py','verify53.py','report53.py','check_delivery53.py','delivery53.py']
    files={name:H/name for name in names};files[archive.name]=archive
    for name,p in files.items():
        need(p.is_file(),'Missing publication member '+name)
        if p.suffix not in ['.png','.zip']:need(not SECRET.search(p.read_bytes()),'Credential pattern '+name)
    scope={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in files.items()}
    dump(H/'PUBLICATION_SCOPE.json',scope);files['PUBLICATION_SCOPE.json']=H/'PUBLICATION_SCOPE.json'
    repo=ROOT/'intercambio/post48_plan_20260927_06.git';index=H/'publication.index'
    need(not index.exists(),'Preserve publication transaction')
    env=dict(os.environ,GIT_INDEX_FILE=str(index),GIT_NO_LAZY_FETCH='1',GIT_TERMINAL_PROMPT='0')
    def git(*args,data=None):
        run=subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange','-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,env=env,capture_output=True,timeout=60)
        need(run.returncode==0,run.stderr.decode(errors='replace'));return run.stdout.decode().strip()
    need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','Wrong repository')
    parent='6d773209860ab0d52088f018d055aa8429be4b52';git('read-tree',parent)
    for name,p in files.items():
        oid=git('hash-object','-w','--',str(p));git('update-index','--add','--cacheinfo','100644,'+oid+','+PREFIX+'/'+name)
    tree=git('write-tree','--missing-ok');commit=git('commit-tree',tree,'-p',parent,data=b'Publish matched JO input campaign53 and reproducible results\n')
    changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    need(changed and all(n.startswith(PREFIX+'/') for n in changed),'Publication scope')
    git('push','origin',commit+':refs/heads/stage45-jo-matched53-20260928')
    download=EXCHANGE/'recibos'/(STEM+'_DESCARGADO.zip')
    url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+archive.name
    with urllib.request.urlopen(url,timeout=45) as source,download.open('xb') as dest:
        for block in iter(lambda:source.read(1024**2),b''):dest.write(block)
    need(sha(download)==receipt['sha256'],'Remote archive differs')
    result=dict(commit=commit,url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,
        archive_sha256=receipt['sha256'],verification=verify_zip(download,'REMOTO'),CPU_s=time.process_time()-start)
    dump(H/'REMOTE_DELIVERY.json',result);print(json.dumps({k:result[k] for k in ['commit','url','CPU_s']}))


if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(150,155))
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    publish() if args.publish else pack()

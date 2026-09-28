"""Publish only delivery receipts; preserve the previously verified science ZIP."""
from pathlib import Path
import hashlib,json,os,subprocess,time,urllib.request
import deliver52 as delivery

H=Path(__file__).resolve().parent;ROOT=H.parents[1]
EX=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
PREFIX=delivery.PREFIX
def main():
    cpu=time.process_time();wall=time.monotonic();times=os.times()
    first=json.loads((H/'REMOTE_DELIVERY.json').read_text());parent=first['commit']
    names=['FULL_CAPSULE.json','ARCHIVE_APPEND_PLAN.json','ARCHIVE_APPEND_EXECUTION.json',
        'resume_full52.py','resume_builder52.py','CLEANUP_LOCAL.json','CLEANUP_REMOTE.json',
        'REMOTE_DELIVERY.json','PUBLICATION_EXECUTION.json','COMPRESION_Y_ENTREGA.md',
        'CIERRE.md','STORAGE_CLOSURE.json','CPU_CLOSURE.json','FULL_EXTRACTION_CHECK.json',
        'publish_closure52.py','cierre/RECOMPUTE_LOCAL.json','cierre/RECOMPUTE_REMOTO.json']
    scope={}
    for name in names:
        p=H/name
        delivery.need(p.is_file() and not p.is_symlink(),'Missing closure '+name)
        delivery.need(not delivery.SECRET.search(p.read_bytes()),'Credential pattern in closure')
        scope[name]=dict(bytes=p.stat().st_size,sha256=delivery.sha(p))
    scope_path=H/'CLOSURE_SCOPE.json'
    delivery.need(not scope_path.exists(),'Immutable closure publication')
    delivery.dump(scope_path,dict(repository='skynet-omega/openmatrix',parent=parent,
        prefix=PREFIX,files=scope,science_archive_unchanged=first['archive_sha256']))
    names.append('CLOSURE_SCOPE.json')
    repo=ROOT/'intercambio/post48_plan_20260927_06.git';index=repo/'repair52closure.index'
    delivery.need(not index.exists(),'Preserve closure transaction')
    env=dict(os.environ,GIT_INDEX_FILE=str(index),GIT_TERMINAL_PROMPT='0',GIT_NO_LAZY_FETCH='1')
    def git(*args,data=None):
        r=subprocess.run(['git','--git-dir='+str(repo),'-c','user.name=AXIOMA exchange',
            '-c','user.email=axioma-exchange@users.noreply.github.com',*args],input=data,
            env=env,capture_output=True,timeout=60)
        delivery.need(r.returncode==0,r.stderr.decode(errors='replace'))
        return r.stdout.decode().strip()
    delivery.need(git('remote','get-url','origin')=='git@github.com:skynet-omega/openmatrix.git','Remote scope')
    delivery.need(git('ls-remote','origin','refs/heads/'+delivery.BRANCH).split()[0]==parent,'Remote branch moved')
    git('read-tree',parent);oids=[]
    for name in names:
        oid=git('hash-object','-w','--',str(H/name));oids.append(oid)
        git('update-index','--add','--cacheinfo','100644,'+oid+','+PREFIX+'/'+name)
    tree=git('write-tree','--missing-ok')
    commit=git('commit-tree',tree,'-p',parent,data=b'Close repair52 verified delivery and full evidence capsule\n')
    changed=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    delivery.need(all(p in {PREFIX+'/'+n for n in names} for p in changed),'Closure changed science or outside scope')
    for name in first['parts']:
        delivery.need(git('rev-parse',parent+':'+PREFIX+'/'+name)==git('rev-parse',commit+':'+PREFIX+'/'+name),'Archive part changed')
    git('push','origin',commit+':refs/heads/'+delivery.BRANCH)
    folder=EX/'recibos'/(delivery.STEM+'_CIERRE');folder.mkdir(exist_ok=False)
    downloaded={}
    for name in names:
        url='https://raw.githubusercontent.com/skynet-omega/openmatrix/'+commit+'/'+PREFIX+'/'+name
        p=folder/name;p.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=30) as r:p.write_bytes(r.read())
        delivery.need(delivery.sha(p)==delivery.sha(H/name),'Remote closure identity '+name)
        downloaded[name]=delivery.sha(p)
    sizes=git('cat-file','--batch-check=%(objectname) %(objectsize:disk)',data=('\n'.join(sorted(set(oids)))+'\n').encode()).splitlines()
    now=os.times();used=time.process_time()-cpu+now.children_user-times.children_user+now.children_system-times.children_system
    result=dict(commit=commit,parent_science_commit=parent,
        url='https://github.com/skynet-omega/openmatrix/tree/'+commit+'/'+PREFIX,
        verified_downloaded_files=downloaded,science_archive_parts_unchanged=True,
        new_blob_disk_bytes=sum(int(line.split()[1]) for line in sizes),CPU_s=used,wall_s=time.monotonic()-wall,
        extraction=str(folder),scope='Delivery receipts only; original scientific ZIP already downloaded and recomputed under parent commit')
    delivery.dump(H/'CLOSURE_REMOTE.json',result)
    print(json.dumps({k:result[k] for k in ['commit','url','CPU_s','new_blob_disk_bytes']}))
if __name__=='__main__':main()

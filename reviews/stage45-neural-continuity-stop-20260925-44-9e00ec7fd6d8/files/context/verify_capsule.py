"""Clean-extraction verification of archived evidence; never launches the CNS."""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path, PurePosixPath

HERE=Path(__file__).resolve().parent
SELF=Path('AXIOMA_ASTRA/campanas/etapa45_neural_contrast_20260925_44')

def need(ok,label):
    if not ok:raise ValueError(label)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def relative(value):
    p=PurePosixPath(value);need(not p.is_absolute() and '..' not in p.parts and '\\' not in value,'Unsafe capsule path')
    return Path(p)

def validate_member(path,item):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],
         'Changed member: '+item['path'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();m=json.loads((HERE/'MANIFEST.json').read_text());count=0
    for item in m['files']:
        payload=HERE/relative(item['storage']);validate_member(payload,item)
        target=HERE/relative(item['path'])
        if target!=payload:
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                try:os.link(payload,target)
                except OSError:shutil.copy2(payload,target)
            validate_member(target,item)
        count+=1
    corruption=[]
    with tempfile.TemporaryDirectory(prefix='stage44_cpu_verify_') as d:
        tmp=Path(d)
        for suffix in ('PLAN.json','sham_01/RESULT.json','sham_01/traces.npz'):
            item=next(x for x in m['files'] if x['path']==str(SELF/suffix))
            altered=tmp/(Path(suffix).name+'.corrupt')
            data=bytearray((HERE/relative(item['storage'])).read_bytes());data[len(data)//2]^=1;altered.write_bytes(data)
            try:validate_member(altered,item)
            except ValueError:corruption.append(suffix)
            else:raise ValueError('Corruption admitted: '+suffix)
        rebuilt=None
        if a.full:
            normal,optimized=tmp/'normal.json',tmp/'optimized.json'
            command=[str(HERE/SELF/'verify_sham.py'),'--folder',str(HERE/SELF/'sham_01')]
            subprocess.run([sys.executable,'-B',*command,'--out',str(normal)],cwd=HERE/'AXIOMA_ASTRA',check=True,capture_output=True)
            subprocess.run([sys.executable,'-O','-B',*command,'--out',str(optimized)],cwd=HERE/'AXIOMA_ASTRA',check=True,capture_output=True)
            need(normal.read_bytes()==optimized.read_bytes(),'Normal/O differs')
            recorded=HERE/SELF/'sham_01/VERIFIED_02.json'
            need(normal.read_bytes()==recorded.read_bytes(),'Sham recalculation differs')
            rebuilt=sha(normal)
            subprocess.run([sys.executable,'-O','-B',str(HERE/SELF/'test_budget_guard.py')],check=True,capture_output=True)
    result={'schema':'stage45_neural_sham_capsule44_verified_v1','hashed_file_count':count,
            'manifest_sha256':sha(HERE/'MANIFEST.json'),'corruption_checks_rejected':corruption,
            'full_sham_recomputation_sha256':rebuilt,'classification':'BLOQUEADO',
            'new_neural_steps':0,'scope':'Short: full member hashes. Full: complete initial-state equality plus120ms raw observable equality and resource-accounting tests. Neither mode reruns the CNS.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()

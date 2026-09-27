"""Archive explicit scientific dependencies and every full endpoint once by hash.

Read-only inventory. The capsule certifies recorded evidence, not an untested
portable GPU restart. No external service is used here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=Path('/home/daroch/AXIOMA_FLYWIRE/matrix')
C48=HERE.parent/'etapa45_composicion_20260927_48'
C49=HERE.parent/'etapa45_operands_20260927_49'
ARMS=tuple(json.loads((HERE/'PILOT_PLAN.json').read_text())['arms'])
SECRET=re.compile(rb'(?i)([a]pikey_[a-z0-9_]{20,}|[s]k-[a-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY)')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def digest(path):
    h=hashlib.sha256();carry=b''
    scan=path.suffix.lower() in ('.json','.py','.md','.log','.txt','.cpp','.cu','.h','.hpp','.cuh','.patch')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):
            h.update(block)
            if scan:
                need(not SECRET.search(carry+block),'Credential pattern in explicit inventory member')
                carry=block[-256:]
    return h.hexdigest()

def inventory():
    opened=json.loads((C48/'software_dependencies_01/OPENED_PROJECT_INPUTS.json').read_text())
    files={Path(p):h for p,h in opened.items()}
    for p,h in json.loads((HERE/'PILOT_SOURCES.json').read_text()).items():files[Path(p)]=h
    for p,h in json.loads((C48/'SOURCES.json').read_text()).items():files[Path(p)]=h
    for arm in ARMS:
        for p,h in json.loads((HERE/arm/'EXECUTED_SOURCES.json').read_text()).items():files[Path(p)]=h
    for arm in ['sham','profile']:
        for p in (C48/arm/'final_state').iterdir():
            if p.is_file():files[p]=None
        files[C49/(arm+'_off_01')/'operands_001ms.npz']=None
    for arm in ['sham','dm1','profile','permuted']:
        for p in (C48/arm/'blocks').glob('*/published.npz'):
            files[p]=None;files[p.parent/'MANIFEST.json']=None
    for p in [C48/'sham/initial_observation.npz',C49/'sham_off_01/traces.npz',C49/'frozen_selection.npz',C48/'ENVIRONMENT.json',C48/'software_dependencies_01/OPENED_PROJECT_INPUTS.json',OLD/'data/male_v10/nodes.parquet',OLD/'data/male_v10/counts_pre_post.npz']:
        files[p]=None
    for p in HERE.rglob('*'):
        rel=p.relative_to(HERE)
        if not p.is_file() or '__pycache__' in p.parts or 'cache' in p.parts:continue
        if len(rel.parts)>1:
            if rel.parts[0] in ARMS or rel.parts[0].startswith('qual_'):
                if len(rel.parts)>2 and rel.parts[1] not in ('final_state','static_inputs'):continue
            elif rel.parts[0]!='aporte_motor':continue
        if p.suffix in ('.zip','.partial','.pyc') or p.name in ('FULL_CAPSULE.json','full_capsule.log','QUEUE_STATUS.json','STATUS.json'):continue
        files[p]=None
    return files

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',type=Path,required=True);a=p.parse_args()
    need(not a.zip.exists(),'Immutable capsule')
    queue=json.loads((HERE/'QUEUE_RESULT.json').read_text());need(queue['status']=='COMPLETE','Ten complete arms required')
    start=time.monotonic();cpu=time.process_time();entries=[];seen={}
    # Scientific states, compact copies and full archive share the original8GiB cap.
    # Reserve the size of one downloaded compact ZIP and its extracted files.
    exchange=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO')
    stem='ETAPA45_ALTERNATIVAS_20260927_51'
    campaign_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    delivery_bytes=sum(p.stat().st_size for d in ['salida','recibos'] for top in (exchange/d).glob(stem+'*') for p in ([top] if top.is_file() else top.rglob('*')) if p.is_file())
    compact=exchange/'salida'/(stem+'.zip')
    compact_manifest=json.loads((HERE/'MANIFEST.json').read_text())
    remote_reserve=(compact.stat().st_size if compact.exists() else 95*1024**2)+sum(v['bytes'] for v in compact_manifest['files'].values())+2*1024**2
    remaining=8589934592-campaign_bytes-delivery_bytes-remote_reserve
    need(remaining>0,'Original aggregate disk bound before full archive')
    sources=inventory()
    for source in sources:
        need(source.is_file() and not source.is_symlink(),'Missing/symlink dependency '+str(source))
        need(source.is_relative_to(ROOT) or source.is_relative_to(OLD),'Outside project inventory')
    a.zip.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.zip,'x',allowZip64=True) as z:
        def payload(data,suffix,compressed):
            h=hashlib.sha256(data).hexdigest()
            if h not in seen:
                storage='payload/'+h+suffix
                z.writestr(storage,data,compress_type=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED,compresslevel=1 if compressed else None)
                seen[h]=storage
            return dict(storage=seen[h],bytes=len(data),sha256=h)
        for source,expected in sorted(sources.items(),key=lambda x:str(x[0])):
            h=digest(source);need(expected is None or h==expected,'Frozen dependency changed '+str(source))
            if source.is_relative_to(ROOT):dest=Path('AXIOMA_ASTRA')/source.relative_to(ROOT)
            else:dest=Path('AXIOMA_FLYWIRE/matrix')/source.relative_to(OLD)
            if source.suffix=='.npz':
                # Raw ZIP-member boundaries preserve every original byte while
                # sharing identical arrays across complete scientific states.
                with zipfile.ZipFile(source) as original:
                    offsets=sorted({0,*[i.header_offset for i in original.infolist()],original.start_dir,source.stat().st_size})
                segments=[]
                with source.open('rb') as f:
                    for lo,hi in zip(offsets[:-1],offsets[1:]):
                        f.seek(lo);segments.append(payload(f.read(hi-lo),'.npz_segment',False))
                entries.append(dict(path=dest.as_posix(),segments=segments,bytes=source.stat().st_size,sha256=h))
            else:
                part=payload(source.read_bytes(),source.suffix,source.suffix not in ('.zip','.png','.pdf','.jpg','.parquet','.gz'))
                entries.append(dict(path=dest.as_posix(),storage=part['storage'],bytes=source.stat().st_size,sha256=h))
            need(a.zip.stat().st_size<4096*1024**2,'Finite archive limit')
            need(a.zip.stat().st_size<remaining,'Original aggregate disk bound; preserve partial archive if exceeded')
        manifest=dict(schema='alternatives51_full_evidence_capsule_v1',files=entries,
            scope='All ten complete endpoints, four qualification records, source48 donors and explicit imported/opened local dependencies; recorded evidence reproducible, portable51 GPU restart not qualified')
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    # Verify EVERY unique payload from the finished ZIP, no working-tree input.
    with zipfile.ZipFile(a.zip) as z:
        m=json.loads(z.read('MANIFEST.json'));count=0
        for h,storage in seen.items():
            d=hashlib.sha256()
            with z.open(storage) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):d.update(block)
            need(d.hexdigest()==h,'Archived payload hash');count+=1
        reconstructed=0
        for item in m['files']:
            if 'segments' not in item:continue
            d=hashlib.sha256();length=0
            for part in item['segments']:
                data=z.read(part['storage']);d.update(data);length+=len(data)
            need(length==item['bytes'] and d.hexdigest()==item['sha256'],'Exact NPZ reconstruction')
            reconstructed+=1
    out=dict(zip=str(a.zip),bytes=a.zip.stat().st_size,sha256=digest(a.zip),paths=len(entries),unique_payloads=len(seen),
             payload_hashes_verified=count,NPZ_files_reconstructed_exact=reconstructed,CPU_s=time.process_time()-cpu,wall_s=time.monotonic()-start,
             scope=manifest['scope'],new_neural_ms=0,campaign_bytes_before=campaign_bytes,other_delivery_bytes_before=delivery_bytes,reserved_remote_bytes=remote_reserve,aggregate_disk_bound_bytes=8589934592)
    (HERE/'FULL_CAPSULE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()

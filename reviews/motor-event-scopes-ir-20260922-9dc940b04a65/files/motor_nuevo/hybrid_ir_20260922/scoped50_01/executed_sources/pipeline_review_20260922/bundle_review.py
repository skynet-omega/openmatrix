"""Freeze explicit review inputs before preparing or publishing a ZIP.

Does not recursively walk campaign outputs or reuse a mutable prepared tree.
"""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile,subprocess,os
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
C=H.parent/'causal_runtime_20260922';N=H.parent/'native_hybrid_20260922';P=ROOT/'campanas/etapa3_motor_nuevo_20260922'
sys.path.insert(0,str(ROOT/'instrumentos/openmatrix'))
import publish
selected={}
def add(path):
    if path.is_file() and path.suffix in publish.EXT and '__pycache__' not in path.parts:
        selected[str(path.relative_to(ROOT))]=path
excluded={'PUBLICATION.json','PUBLICATION_MANIFEST.json','PREPARED.json','DELIVERY.json','CLEAN_CHECK.json','CLEAN_CHECK.log'}
for path in H.iterdir():
    if path.is_file() and path.name not in excluded:add(path)
for path in (H/'jev_01').iterdir():add(path)
for base in (C,N,P,H.parent/'resident_pn_20260922'):
    for path in base.iterdir():
        if path.suffix in ('.py','.cpp','.cu','.hpp') and path.name!='package.py':add(path)
for base in (N/'vendor',N/'legacy_sources',P/'verification_vendor'):
    for path in base.iterdir():add(path)
for path in (N/'PETSC_LICENSE.txt',ROOT/'instrumentos/openmatrix/publish.py'):add(path)
data=('brain_final.json','brain_final.npz','body_final.npz','traces.npz')
for base in (H/'smoke_causal_01',H/'smoke_reference_01',H/'trace_smoke_01',H/'trace_smoke_02',C/'reference20_01',C/'device20_01',N/'baseline_01'):
    for name in (*data,'RESULT.json','RUN_CONTRACT.json','TIMES.jsonl','CENTRAL_ROWS.json','FROZEN.json'):
        add(base/name)
    for name in ('MANIFEST.json','effective_operator.json','boundary.json'):
        add(base/'final_state'/name)
    # Executed source versions are provenance, without full session snapshots.
    frozen=base/'executed_sources'
    if frozen.is_dir():
        for group in frozen.iterdir():
            if group.is_dir():
                for path in group.iterdir():add(path)
            else:add(group)
payload=H/'publication_sources';payload.mkdir(exist_ok=False)
files=[]
for relative,source in sorted(selected.items()):
    dest=payload/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    data=source.read_bytes();dest.write_bytes(data)
    if dest.read_bytes()!=data:raise RuntimeError('Freeze mismatch')
    files.append({'source':str(dest),'destination':relative})
spec={'label':'motor-pipeline-review-20260922','description':'Repaired runner, authenticated operator routes and failure handling; actual1ms transient fails unchanged tolerance. Exact instrumentation exposes CNS event/controller cuts. Review bundle with raw arrays and portable checks; no full static organism assets or complete session snapshots.','files':files}
(H/'PUBLICATION_MANIFEST.json').write_text(json.dumps(spec,indent=2)+'\n')
folder,manifest=publish.prepare(spec,H/'prepared')
meta=json.loads((folder/'ARCHIVE.json').read_text())
archive=Path('/mnt/f/Downloads/AXIOMA_INTERCAMBIO/salida/MATRIX_PIPELINE_REVIEW_2026-09-22.zip')
with archive.open('xb') as stream:
    for row in meta['parts']:
        data=(folder/row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise RuntimeError('Part hash mismatch')
        stream.write(data)
if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['archive_sha256']:raise RuntimeError('Archive hash mismatch')
clean=H/'clean_verification';clean.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:z.extractall(clean)
for row in manifest['files']:
    if hashlib.sha256((clean/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Extracted file hash mismatch')
env=os.environ.copy();env.pop('PYTHONPATH',None)
env.update(PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUPY_CACHE_DIR=str(clean/'cuda_cache'))
checks=[]
with (H/'CLEAN_CHECK.log').open('x') as log:
    for name,opt in [('test_repairs.py',True),('test_runner_failures.py',False),('check_coverage_after.py',True),('reconstruct.py',True),('check_publication_failure.py',False)]:
        command=[sys.executable,'-B']+(['-O'] if opt else [])+[str(clean/'motor_nuevo/pipeline_review_20260922'/name)]
        subprocess.run(command,cwd=clean,env=env,stdout=log,stderr=log,timeout=120,check=True)
        checks.append(name)
result={'archive':str(archive),'sha256':meta['archive_sha256'],'bytes':archive.stat().st_size,
    'files':len(manifest['files']),'clean_short_checks':checks,'full_organism_rerun':False,
    'scope':'Fresh extraction; CPU corruption/failure checks, evidence reconstruction, small CUDA publication/grouping fixture. Full-body resume not tested.',
    'prepared':str(folder)}
(H/'DELIVERY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
